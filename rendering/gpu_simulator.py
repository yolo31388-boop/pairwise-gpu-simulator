"""
游戏内GPU模拟器与渲染管线系统 - 核心模块

实现一条简化但完整的可编程渲染管线：

- 顶点处理器：模型-视图-投影矩阵变换 + 可编程顶点着色器
- 光栅化器：三角形 -> 片元，支持深度测试与背面剔除
- 片元处理器：可编程片元着色器 + 顶点属性重心插值
- 纹理采样器：最近邻 / 双线性过滤，重复 / 钳制环绕
- 帧缓冲：颜色缓冲 + 深度缓冲 + 模板缓冲

仅使用 Python 标准库。
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import math

Color = Tuple[float, float, float]
Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]
Mat4 = List[List[float]]


# ---------------------------------------------------------------------------
# 数学工具（4x4 矩阵，行主序，列向量约定：p' = M * p）
# ---------------------------------------------------------------------------

def mat4_identity() -> Mat4:
    return [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]


def mat4_multiply(a: Mat4, b: Mat4) -> Mat4:
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)]
            for i in range(4)]


def mat4_transform(m: Mat4, v: Sequence[float]) -> Tuple[float, float, float, float]:
    x, y, z, w = v
    return tuple(
        m[i][0] * x + m[i][1] * y + m[i][2] * z + m[i][3] * w
        for i in range(4)
    )


def translation_matrix(tx: float, ty: float, tz: float) -> Mat4:
    m = mat4_identity()
    m[0][3], m[1][3], m[2][3] = tx, ty, tz
    return m


def scale_matrix(sx: float, sy: float, sz: float) -> Mat4:
    m = mat4_identity()
    m[0][0], m[1][1], m[2][2] = sx, sy, sz
    return m


def perspective_matrix(fov_y: float, aspect: float, near: float, far: float) -> Mat4:
    f = 1.0 / math.tan(fov_y / 2.0)
    return [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (far + near) / (near - far), 2.0 * far * near / (near - far)],
        [0.0, 0.0, -1.0, 0.0],
    ]


# ---------------------------------------------------------------------------
# 顶点与片元
# ---------------------------------------------------------------------------

@dataclass
class Vertex:
    position: Vec3
    color: Color = (1.0, 1.0, 1.0)
    uv: Vec2 = (0.0, 0.0)


@dataclass
class Fragment:
    x: int
    y: int
    depth: float
    color: Color
    uv: Vec2


# ---------------------------------------------------------------------------
# 帧缓冲：颜色缓冲 + 深度缓冲 + 模板缓冲
# ---------------------------------------------------------------------------

class Framebuffer:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.color_buffer: List[List[Color]] = []
        self.depth_buffer: List[List[float]] = []
        self.stencil_buffer: List[List[int]] = []
        self.clear()

    def clear(self, color: Color = (0.0, 0.0, 0.0), depth: float = 1.0,
              stencil: int = 0):
        self.color_buffer = [[color for _ in range(self.width)]
                             for _ in range(self.height)]
        self.depth_buffer = [[depth for _ in range(self.width)]
                             for _ in range(self.height)]
        self.stencil_buffer = [[stencil for _ in range(self.width)]
                               for _ in range(self.height)]

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def write_pixel(self, x: int, y: int, color: Color, depth: float = 0.0,
                    stencil: Optional[int] = None, depth_test: bool = True) -> bool:
        """写入像素；启用深度测试时，只有更近的片元才能覆盖已有像素。"""
        if not self._in_bounds(x, y):
            return False
        if depth_test and depth >= self.depth_buffer[y][x]:
            return False
        self.depth_buffer[y][x] = depth
        self.color_buffer[y][x] = color
        if stencil is not None:
            self.stencil_buffer[y][x] = stencil
        return True

    def read_pixel(self, x: int, y: int) -> Color:
        return self.color_buffer[y][x]

    def read_depth(self, x: int, y: int) -> float:
        return self.depth_buffer[y][x]

    def read_stencil(self, x: int, y: int) -> int:
        return self.stencil_buffer[y][x]


# ---------------------------------------------------------------------------
# GPU 模拟器
# ---------------------------------------------------------------------------

class GPUSimulator:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        width = int(self.config.get("width", 64))
        height = int(self.config.get("height", 64))
        self.framebuffer = Framebuffer(width, height)
        self._state = {"time": 0.0, "frames": 0}
        self._history: List[int] = []

    # -- 模拟器生命周期 ----------------------------------------------------

    def update(self, dt: float):
        self._state["time"] += dt

    def reset(self):
        self._state = {"time": 0.0, "frames": 0}
        self._history = []
        self.framebuffer.clear()

    # -- 顶点处理器 ---------------------------------------------------------

    def vertex_processor(
        self,
        vertices: Sequence[Vertex],
        mvp: Optional[Mat4] = None,
        vertex_shader: Optional[Callable[[Vertex], Vertex]] = None,
    ) -> List[Vertex]:
        """应用 MVP 矩阵变换（含透视除法），再执行可选的顶点着色器。"""
        out: List[Vertex] = []
        for v in vertices:
            if mvp is not None:
                cx, cy, cz, cw = mat4_transform(mvp, (*v.position, 1.0))
                if cw != 0.0:
                    cx, cy, cz = cx / cw, cy / cw, cz / cw
                transformed = Vertex((cx, cy, cz), v.color, v.uv)
            else:
                transformed = Vertex(v.position, v.color, v.uv)
            if vertex_shader is not None:
                transformed = vertex_shader(transformed)
            out.append(transformed)
        return out

    # -- 光栅化器 -----------------------------------------------------------

    def rasterizer(
        self,
        triangle: Sequence[Vertex],
        width: Optional[int] = None,
        height: Optional[int] = None,
        depth_buffer: Optional[List[List[float]]] = None,
        depth_test: bool = True,
        backface_culling: bool = True,
    ) -> List[Fragment]:
        """将 NDC 空间的三角形光栅化为片元，重心插值颜色 / UV / 深度。"""
        if len(triangle) != 3:
            raise ValueError("rasterizer 只接受三角形（3 个顶点）")
        width = width if width is not None else self.framebuffer.width
        height = height if height is not None else self.framebuffer.height

        # NDC [-1,1] -> 屏幕坐标
        sx = [(v.position[0] + 1.0) * 0.5 * width for v in triangle]
        sy = [(v.position[1] + 1.0) * 0.5 * height for v in triangle]

        # 背面剔除：屏幕空间有向面积 <= 0 视为背面
        signed_area = ((sx[1] - sx[0]) * (sy[2] - sy[0])
                       - (sx[2] - sx[0]) * (sy[1] - sy[0]))
        if backface_culling and signed_area <= 0.0:
            return []
        if signed_area == 0.0:
            return []

        min_x = max(int(math.floor(min(sx))), 0)
        max_x = min(int(math.ceil(max(sx))), width - 1)
        min_y = max(int(math.floor(min(sy))), 0)
        max_y = min(int(math.ceil(max(sy))), height - 1)

        denom = ((sy[1] - sy[2]) * (sx[0] - sx[2])
                 + (sx[2] - sx[1]) * (sy[0] - sy[2]))
        if denom == 0.0:
            return []

        fragments: List[Fragment] = []
        for py in range(min_y, max_y + 1):
            for px in range(min_x, max_x + 1):
                cx, cy = px + 0.5, py + 0.5
                w0 = ((sy[1] - sy[2]) * (cx - sx[2])
                      + (sx[2] - sx[1]) * (cy - sy[2])) / denom
                w1 = ((sy[2] - sy[0]) * (cx - sx[2])
                      + (sx[0] - sx[2]) * (cy - sy[2])) / denom
                w2 = 1.0 - w0 - w1
                eps = -1e-9
                if w0 < eps or w1 < eps or w2 < eps:
                    continue

                weights = (w0, w1, w2)
                depth = sum(w * v.position[2] for w, v in zip(weights, triangle))

                # 深度测试：只有更近的片元通过，并更新深度缓冲
                if depth_buffer is not None and depth_test:
                    if depth >= depth_buffer[py][px]:
                        continue
                    depth_buffer[py][px] = depth

                color = tuple(
                    sum(w * v.color[c] for w, v in zip(weights, triangle))
                    for c in range(3)
                )
                uv = tuple(
                    sum(w * v.uv[i] for w, v in zip(weights, triangle))
                    for i in range(2)
                )
                fragments.append(Fragment(px, py, depth, color, uv))
        return fragments

    # -- 片元处理器 ---------------------------------------------------------

    def fragment_processor(
        self,
        fragments: Sequence[Fragment],
        fragment_shader: Optional[Callable[[Fragment], Color]] = None,
    ) -> List[Fragment]:
        """对每个片元执行片元着色器（光照 / 材质），颜色来自顶点属性插值。"""
        out: List[Fragment] = []
        for frag in fragments:
            color = fragment_shader(frag) if fragment_shader is not None else frag.color
            out.append(Fragment(frag.x, frag.y, frag.depth, color, frag.uv))
        return out

    # -- 纹理采样器 ---------------------------------------------------------

    @staticmethod
    def _wrap_coord(t: float, wrap_mode: str) -> float:
        if wrap_mode == "repeat":
            return t % 1.0
        if wrap_mode == "clamp":
            return min(max(t, 0.0), 1.0)
        raise ValueError(f"未知环绕模式: {wrap_mode}")

    def texture_sampler(
        self,
        texture: Sequence[Sequence[Color]],
        u: float,
        v: float,
        filter_mode: str = "nearest",
        wrap_mode: str = "repeat",
    ) -> Color:
        """按过滤模式与环绕模式采样纹理，UV 超出 [0,1] 不再报错。"""
        h = len(texture)
        w = len(texture[0])
        u = self._wrap_coord(u, wrap_mode)
        v = self._wrap_coord(v, wrap_mode)

        if filter_mode == "nearest":
            x = min(int(u * w), w - 1)
            y = min(int(v * h), h - 1)
            return texture[y][x]

        if filter_mode == "bilinear":
            fx = u * w - 0.5
            fy = v * h - 0.5
            x0 = math.floor(fx)
            y0 = math.floor(fy)
            tx = fx - x0
            ty = fy - y0
            x1, y1 = x0 + 1, y0 + 1
            if wrap_mode == "repeat":
                x0, x1 = x0 % w, x1 % w
                y0, y1 = y0 % h, y1 % h
            else:
                x0 = min(max(x0, 0), w - 1)
                x1 = min(max(x1, 0), w - 1)
                y0 = min(max(y0, 0), h - 1)
                y1 = min(max(y1, 0), h - 1)
            c00 = texture[y0][x0]
            c10 = texture[y0][x1]
            c01 = texture[y1][x0]
            c11 = texture[y1][x1]
            return tuple(
                (c00[c] * (1 - tx) + c10[c] * tx) * (1 - ty)
                + (c01[c] * (1 - tx) + c11[c] * tx) * ty
                for c in range(3)
            )

        raise ValueError(f"未知过滤模式: {filter_mode}")

    # -- 完整管线 -----------------------------------------------------------

    def render(
        self,
        triangles: Sequence[Sequence[Vertex]],
        mvp: Optional[Mat4] = None,
        vertex_shader: Optional[Callable[[Vertex], Vertex]] = None,
        fragment_shader: Optional[Callable[[Fragment], Color]] = None,
        depth_test: bool = True,
        backface_culling: bool = True,
    ) -> int:
        """跑完整条管线并把结果写入帧缓冲，返回写入的像素数。"""
        self.framebuffer.clear()
        written = 0
        for triangle in triangles:
            verts = self.vertex_processor(triangle, mvp=mvp, vertex_shader=vertex_shader)
            fragments = self.rasterizer(
                verts,
                depth_buffer=self.framebuffer.depth_buffer,
                depth_test=depth_test,
                backface_culling=backface_culling,
            )
            fragments = self.fragment_processor(fragments, fragment_shader=fragment_shader)
            for frag in fragments:
                if self.framebuffer.write_pixel(frag.x, frag.y, frag.color,
                                                depth=frag.depth, depth_test=False):
                    written += 1
        self._state["frames"] += 1
        self._history.append(written)
        return written
