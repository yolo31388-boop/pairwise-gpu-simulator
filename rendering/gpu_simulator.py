"""
游戏内GPU模拟器与渲染管线系统 - 核心模块

实现一个纯Python的软件渲染管线：
- 顶点处理器：MVP矩阵变换 + 可编程顶点着色器
- 光栅化器：三角形 -> 片元，支持深度测试与背面剔除
- 片元处理器：属性插值 + 可编程片元着色器
- 纹理采样器：最近邻/双线性过滤，重复/钳制/镜像环绕
- 帧缓冲：颜色缓冲 + 深度缓冲 + 模板缓冲
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import math
import time

Color = Tuple[float, float, float]
Vec3 = Tuple[float, float, float]
Vec4 = Tuple[float, float, float, float]
Mat4 = List[List[float]]


# ---------------------------------------------------------------------------
# 矩阵工具（4x4，行主序）
# ---------------------------------------------------------------------------

def mat4_identity() -> Mat4:
    return [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]


def mat4_multiply(a: Mat4, b: Mat4) -> Mat4:
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)]
            for i in range(4)]


def mat4_transform_vec4(m: Mat4, v: Vec4) -> Vec4:
    return tuple(sum(m[i][k] * v[k] for k in range(4)) for i in range(4))


def translation_matrix(x: float, y: float, z: float) -> Mat4:
    m = mat4_identity()
    m[0][3], m[1][3], m[2][3] = x, y, z
    return m


def perspective_matrix(fov_y: float, aspect: float, near: float, far: float) -> Mat4:
    f = 1.0 / math.tan(fov_y / 2.0)
    m = [[0.0] * 4 for _ in range(4)]
    m[0][0] = f / aspect
    m[1][1] = f
    m[2][2] = (far + near) / (near - far)
    m[2][3] = 2.0 * far * near / (near - far)
    m[3][2] = -1.0
    return m


# ---------------------------------------------------------------------------
# 帧缓冲：颜色 + 深度 + 模板
# ---------------------------------------------------------------------------

class Framebuffer:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.clear()

    def clear(self, color: Color = (0.0, 0.0, 0.0),
              depth: float = 1.0, stencil: int = 0):
        self.color = [[color for _ in range(self.width)]
                      for _ in range(self.height)]
        self.depth = [[depth for _ in range(self.width)]
                      for _ in range(self.height)]
        self.stencil = [[stencil for _ in range(self.width)]
                        for _ in range(self.height)]

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def read_pixel(self, x: int, y: int) -> Color:
        return self.color[y][x]

    def write_pixel(self, x: int, y: int, color: Color,
                    depth: Optional[float] = None,
                    stencil: Optional[int] = None,
                    depth_test: bool = True) -> bool:
        """写入像素。启用深度测试时，只有更近的片元才会写入。返回是否写入。"""
        if not self.in_bounds(x, y):
            return False
        if depth is not None and depth_test:
            if depth >= self.depth[y][x]:
                return False
            self.depth[y][x] = depth
        elif depth is not None:
            self.depth[y][x] = depth
        self.color[y][x] = color
        if stencil is not None:
            self.stencil[y][x] = stencil
        return True

    def read_depth(self, x: int, y: int) -> float:
        return self.depth[y][x]

    def write_stencil(self, x: int, y: int, value: int):
        self.stencil[y][x] = value

    def read_stencil(self, x: int, y: int) -> int:
        return self.stencil[y][x]


# ---------------------------------------------------------------------------
# GPU模拟器
# ---------------------------------------------------------------------------

@dataclass
class Config:
    pass


class GPUSimulator:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.width = int(self.config.get("width", 64))
        self.height = int(self.config.get("height", 64))
        self._state = {}
        self._history = []
        self._framebuffer = Framebuffer(self.width, self.height)

    # -- 生命周期 ----------------------------------------------------------

    def update(self, dt: float):
        self._history.append(("update", dt))

    def reset(self):
        self._state = {}
        self._history = []
        self._framebuffer.clear()

    # -- 帧缓冲 ------------------------------------------------------------

    def framebuffer(self, width: Optional[int] = None,
                    height: Optional[int] = None) -> Framebuffer:
        """返回当前帧缓冲；给定尺寸时重建帧缓冲。"""
        if width is not None and height is not None:
            if width != self._framebuffer.width or height != self._framebuffer.height:
                self.width, self.height = width, height
                self._framebuffer = Framebuffer(width, height)
        return self._framebuffer

    # -- 顶点处理器 ----------------------------------------------------------

    def vertex_processor(self,
                         vertices: Sequence[Dict],
                         mvp: Optional[Mat4] = None,
                         vertex_shader: Optional[Callable[[Dict, Optional[Mat4]], Vec4]] = None
                         ) -> List[Dict]:
        """
        顶点处理：应用MVP矩阵或自定义顶点着色器，做透视除法与视口变换。
        输入顶点至少包含 "position": (x, y, z)，可附带 "color"/"uv" 等属性。
        输出顶点包含 "screen": (sx, sy) 与 "depth"，其余属性原样保留。
        """
        out = []
        for v in vertices:
            if vertex_shader is not None:
                clip = vertex_shader(v, mvp)
            elif mvp is not None:
                clip = mat4_transform_vec4(mvp, (*v["position"], 1.0))
            else:
                clip = (*v["position"], 1.0)
            w = clip[3] if clip[3] != 0.0 else 1.0
            ndc_x, ndc_y, ndc_z = clip[0] / w, clip[1] / w, clip[2] / w
            sx = (ndc_x + 1.0) * 0.5 * self.width
            sy = (1.0 - ndc_y) * 0.5 * self.height
            nv = dict(v)
            nv["screen"] = (sx, sy)
            nv["depth"] = (ndc_z + 1.0) * 0.5
            out.append(nv)
        return out

    # -- 光栅化器 ------------------------------------------------------------

    @staticmethod
    def _signed_area(p0, p1, p2) -> float:
        return ((p1[0] - p0[0]) * (p2[1] - p0[1])
                - (p2[0] - p0[0]) * (p1[1] - p0[1]))

    def rasterizer(self,
                   triangle: Sequence[Dict],
                   depth_test: bool = True,
                   cull_backface: bool = True,
                   framebuffer: Optional[Framebuffer] = None) -> List[Dict]:
        """
        将三角形光栅化为片元列表。
        - 背面剔除：屏幕空间面积为非正（顺时针）的三角形被剔除
        - 深度测试：对帧缓冲深度做 early-z，通过者更新深度缓冲
        - 重心插值：depth/color/uv 逐像素插值
        """
        fb = framebuffer or self._framebuffer
        v0, v1, v2 = triangle
        p0, p1, p2 = v0["screen"], v1["screen"], v2["screen"]
        area = self._signed_area(p0, p1, p2)
        if cull_backface and area <= 0.0:
            return []
        if area == 0.0:
            return []

        min_x = max(0, int(math.floor(min(p0[0], p1[0], p2[0]))))
        max_x = min(fb.width - 1, int(math.ceil(max(p0[0], p1[0], p2[0]))))
        min_y = max(0, int(math.floor(min(p0[1], p1[1], p2[1]))))
        max_y = min(fb.height - 1, int(math.ceil(max(p0[1], p1[1], p2[1]))))

        fragments = []
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                px, py = x + 0.5, y + 0.5
                w0 = self._signed_area(p1, p2, (px, py)) / area
                w1 = self._signed_area(p2, p0, (px, py)) / area
                w2 = self._signed_area(p0, p1, (px, py)) / area
                if w0 < 0.0 or w1 < 0.0 or w2 < 0.0:
                    continue
                depth = w0 * v0["depth"] + w1 * v1["depth"] + w2 * v2["depth"]
                if depth_test and depth >= fb.depth[y][x]:
                    continue
                frag = {"x": x, "y": y, "depth": depth}
                for attr in ("color", "uv"):
                    if all(attr in v for v in (v0, v1, v2)):
                        frag[attr] = tuple(
                            w0 * v0[attr][i] + w1 * v1[attr][i] + w2 * v2[attr][i]
                            for i in range(len(v0[attr])))
                fragments.append(frag)
                if depth_test:
                    fb.depth[y][x] = depth
        return fragments

    # -- 片元处理器 ----------------------------------------------------------

    def fragment_processor(self,
                           fragments: Sequence[Dict],
                           fragment_shader: Optional[Callable[[Dict], Color]] = None,
                           texture: Optional[List[List[Color]]] = None,
                           filter_mode: str = "bilinear",
                           wrap_mode: str = "repeat") -> List[Dict]:
        """
        片元处理：默认输出插值颜色；提供片元着色器时由着色器计算颜色
        （可用于光照/材质）；提供纹理时按插值UV采样并调制颜色。
        """
        pixels = []
        for frag in fragments:
            if fragment_shader is not None:
                color = fragment_shader(frag)
            else:
                color = frag.get("color", (255.0, 255.0, 255.0))
                if texture is not None and "uv" in frag:
                    tex = self.texture_sampler(texture, frag["uv"][0], frag["uv"][1],
                                               filter_mode=filter_mode,
                                               wrap_mode=wrap_mode)
                    color = tuple(color[i] * tex[i] / 255.0 for i in range(3))
            pixels.append({"x": frag["x"], "y": frag["y"],
                           "depth": frag["depth"], "color": color})
        return pixels

    # -- 纹理采样器 ----------------------------------------------------------

    @staticmethod
    def _wrap_coord(t: float, wrap_mode: str) -> float:
        if wrap_mode == "repeat":
            return t % 1.0
        if wrap_mode == "clamp":
            return min(max(t, 0.0), 1.0)
        if wrap_mode == "mirror":
            t = t % 2.0
            return t if t <= 1.0 else 2.0 - t
        raise ValueError(f"未知环绕模式: {wrap_mode}")

    def texture_sampler(self,
                        texture: List[List[Color]],
                        u: float, v: float,
                        filter_mode: str = "bilinear",
                        wrap_mode: str = "repeat") -> Color:
        """
        纹理采样：支持 nearest/bilinear 过滤与 repeat/clamp/mirror 环绕，
        超出 [0,1] 的UV按环绕模式处理，不再报错。
        """
        h, w = len(texture), len(texture[0])
        u = self._wrap_coord(u, wrap_mode)
        v = self._wrap_coord(v, wrap_mode)

        def texel(ix: int, iy: int) -> Color:
            if wrap_mode == "clamp":
                ix = min(max(ix, 0), w - 1)
                iy = min(max(iy, 0), h - 1)
            elif wrap_mode == "mirror":
                ix = abs(ix) % (2 * w)
                iy = abs(iy) % (2 * h)
                ix = ix if ix < w else 2 * w - 1 - ix
                iy = iy if iy < h else 2 * h - 1 - iy
            else:
                ix %= w
                iy %= h
            return texture[iy][ix]

        if filter_mode == "nearest":
            ix = min(int(u * w), w - 1) if wrap_mode == "clamp" else int(u * w) % w
            iy = min(int(v * h), h - 1) if wrap_mode == "clamp" else int(v * h) % h
            return tuple(float(c) for c in texture[iy][ix])
        if filter_mode == "bilinear":
            fx, fy = u * w - 0.5, v * h - 0.5
            x0, y0 = math.floor(fx), math.floor(fy)
            tx, ty = fx - x0, fy - y0
            c00, c10 = texel(x0, y0), texel(x0 + 1, y0)
            c01, c11 = texel(x0, y0 + 1), texel(x0 + 1, y0 + 1)
            return tuple(
                (c00[i] * (1 - tx) + c10[i] * tx) * (1 - ty)
                + (c01[i] * (1 - tx) + c11[i] * tx) * ty
                for i in range(3))
        raise ValueError(f"未知过滤模式: {filter_mode}")

    # -- 完整管线 ------------------------------------------------------------

    def render_triangle(self,
                        vertices: Sequence[Dict],
                        mvp: Optional[Mat4] = None,
                        vertex_shader: Optional[Callable] = None,
                        fragment_shader: Optional[Callable] = None,
                        texture: Optional[List[List[Color]]] = None,
                        filter_mode: str = "bilinear",
                        wrap_mode: str = "repeat",
                        depth_test: bool = True,
                        cull_backface: bool = True) -> List[Dict]:
        """执行完整渲染管线并把结果写入帧缓冲，返回写入的像素。"""
        vs_out = self.vertex_processor(vertices, mvp=mvp, vertex_shader=vertex_shader)
        fragments = self.rasterizer(vs_out, depth_test=depth_test,
                                    cull_backface=cull_backface)
        pixels = self.fragment_processor(fragments, fragment_shader=fragment_shader,
                                         texture=texture, filter_mode=filter_mode,
                                         wrap_mode=wrap_mode)
        fb = self._framebuffer
        for p in pixels:
            fb.write_pixel(p["x"], p["y"], p["color"], depth=p["depth"],
                           depth_test=False)
        return pixels
