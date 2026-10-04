"""
游戏内GPU模拟器与渲染管线系统 - 验收测试
运行方式：python -m pytest tests/test_gpu_simulator.py -q
共 16 个测试用例
"""
import math

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rendering.gpu_simulator import (
    GPUSimulator,
    Vertex,
    perspective_matrix,
    scale_matrix,
    translation_matrix,
)


def make_triangle(z=0.0):
    """NDC 空间中的正面（逆时针）大三角形。"""
    return [
        Vertex((-0.8, -0.8, z), (1.0, 0.0, 0.0), (0.0, 0.0)),
        Vertex((0.8, -0.8, z), (0.0, 1.0, 0.0), (1.0, 0.0)),
        Vertex((0.0, 0.8, z), (0.0, 0.0, 1.0), (0.5, 1.0)),
    ]


class TestGPUSimulator:
    def setup_method(self):
        self.system = GPUSimulator({"width": 64, "height": 64})

    # -- 顶点处理器 ----------------------------------------------------

    def test_case_01(self):
        """顶点处理器应用 MVP 矩阵变换（平移生效）。"""
        verts = [Vertex((0.0, 0.0, 0.0))]
        out = self.system.vertex_processor(verts, mvp=translation_matrix(0.5, -0.25, 2.0))
        assert out[0].position == pytest.approx((0.5, -0.25, 2.0))

    def test_case_02(self):
        """顶点处理器应用缩放矩阵，位置不再原样通过。"""
        verts = [Vertex((1.0, 2.0, 3.0))]
        out = self.system.vertex_processor(verts, mvp=scale_matrix(2.0, 3.0, 4.0))
        assert out[0].position == pytest.approx((2.0, 6.0, 12.0))

    def test_case_03(self):
        """顶点处理器应用投影矩阵并做透视除法。"""
        mvp = perspective_matrix(math.radians(90.0), 1.0, 1.0, 10.0)
        verts = [Vertex((1.0, 0.0, -2.0))]
        out = self.system.vertex_processor(verts, mvp=mvp)
        # x_clip = 1, w = 2 -> x_ndc = 0.5
        assert out[0].position[0] == pytest.approx(0.5)
        assert 0.0 < out[0].position[2] < 1.0

    def test_case_04(self):
        """顶点处理器支持可编程顶点着色器。"""
        def shader(v):
            x, y, z = v.position
            return Vertex((x, y + 0.5, z), (1.0, 0.0, 0.0), v.uv)

        verts = [Vertex((0.0, 0.0, 0.0), (0.0, 1.0, 0.0))]
        out = self.system.vertex_processor(verts, vertex_shader=shader)
        assert out[0].position == pytest.approx((0.0, 0.5, 0.0))
        assert out[0].color == (1.0, 0.0, 0.0)

    # -- 光栅化器 ------------------------------------------------------

    def test_case_05(self):
        """光栅化器把三角形变成片元（像素）。"""
        fragments = self.system.rasterizer(make_triangle())
        assert len(fragments) > 0
        coords = {(f.x, f.y) for f in fragments}
        assert (32, 32) in coords  # 三角形覆盖屏幕中心

    def test_case_06(self):
        """光栅化器深度测试：近处物体覆盖远处物体。"""
        depth_buffer = [[1.0] * 64 for _ in range(64)]
        far = self.system.rasterizer(make_triangle(z=0.5), depth_buffer=depth_buffer)
        near = self.system.rasterizer(make_triangle(z=0.2), depth_buffer=depth_buffer)
        assert len(far) > 0 and len(near) > 0
        center_depth = depth_buffer[32][32]
        assert center_depth == pytest.approx(0.2)
        # 再画一次远处三角形，应被深度测试全部拒绝
        rejected = self.system.rasterizer(make_triangle(z=0.5), depth_buffer=depth_buffer)
        assert len(rejected) == 0

    def test_case_07(self):
        """光栅化器背面剔除：反面三角形不渲染，正面正常渲染。"""
        front = make_triangle()
        back = [front[0], front[2], front[1]]  # 翻转绕序 -> 背面
        assert len(self.system.rasterizer(front, backface_culling=True)) > 0
        assert len(self.system.rasterizer(back, backface_culling=True)) == 0
        # 关闭剔除时正反面都渲染
        assert len(self.system.rasterizer(back, backface_culling=False)) > 0

    def test_case_08(self):
        """光栅化器对顶点颜色做重心插值。"""
        fragments = self.system.rasterizer(make_triangle())
        center = next(f for f in fragments if (f.x, f.y) == (32, 32))
        # 中心点附近颜色应接近三个顶点颜色的混合，而非某个固定值
        r, g, b = center.color
        assert r > 0.1 and g > 0.1 and b > 0.1
        assert r + g + b == pytest.approx(1.0, abs=1e-6)

    # -- 片元处理器 ----------------------------------------------------

    def test_case_09(self):
        """片元处理器执行片元着色器（简单光照：按深度变暗）。"""
        fragments = self.system.rasterizer(make_triangle(z=0.4))

        def lighting(frag):
            factor = 1.0 - frag.depth
            return tuple(c * factor for c in frag.color)

        shaded = self.system.fragment_processor(fragments, fragment_shader=lighting)
        for frag, out in zip(fragments, shaded):
            assert out.color == pytest.approx(tuple(c * 0.6 for c in frag.color))

    def test_case_10(self):
        """片元处理器输出插值颜色，三角形上颜色平滑过渡而非固定色。"""
        fragments = self.system.rasterizer(make_triangle())
        shaded = self.system.fragment_processor(fragments)
        distinct_colors = {f.color for f in shaded}
        assert len(distinct_colors) > 10
        # 左侧偏红、右侧偏绿：验证渐变方向
        left = min(shaded, key=lambda f: f.x)
        right = max(shaded, key=lambda f: f.x)
        assert left.color[0] > left.color[1]
        assert right.color[1] > right.color[0]

    # -- 纹理采样器 ----------------------------------------------------

    TEXTURE = [
        [(255, 0, 0), (0, 255, 0)],
        [(0, 0, 255), (255, 255, 255)],
    ]

    def test_case_11(self):
        """纹理采样器最近邻过滤取最近纹素。"""
        assert self.system.texture_sampler(self.TEXTURE, 0.1, 0.1) == (255, 0, 0)
        assert self.system.texture_sampler(self.TEXTURE, 0.9, 0.1) == (0, 255, 0)
        assert self.system.texture_sampler(self.TEXTURE, 0.9, 0.9) == (255, 255, 255)

    def test_case_12(self):
        """纹理采样器双线性过滤在四纹素中心取平均。"""
        color = self.system.texture_sampler(self.TEXTURE, 0.5, 0.5, filter_mode="bilinear")
        assert color == pytest.approx((127.5, 127.5, 127.5))

    def test_case_13(self):
        """纹理采样器重复环绕：UV 超出 [0,1] 自动回绕，不报错。"""
        wrapped = self.system.texture_sampler(self.TEXTURE, 1.25, 2.75, wrap_mode="repeat")
        normal = self.system.texture_sampler(self.TEXTURE, 0.25, 0.75, wrap_mode="repeat")
        assert wrapped == normal

    def test_case_14(self):
        """纹理采样器钳制环绕：UV 超出范围钳到边缘纹素。"""
        assert self.system.texture_sampler(self.TEXTURE, 1.5, 0.1, wrap_mode="clamp") == (0, 255, 0)
        assert self.system.texture_sampler(self.TEXTURE, -0.5, 0.1, wrap_mode="clamp") == (255, 0, 0)

    # -- 帧缓冲 ---------------------------------------------------------

    def test_case_15(self):
        """帧缓冲保存渲染结果，并用深度缓冲做深度测试。"""
        fb = self.system.framebuffer
        assert fb.write_pixel(10, 10, (1.0, 0.0, 0.0), depth=0.5)
        assert fb.read_pixel(10, 10) == (1.0, 0.0, 0.0)
        # 更远的像素被深度缓冲拒绝
        assert not fb.write_pixel(10, 10, (0.0, 1.0, 0.0), depth=0.8)
        assert fb.read_pixel(10, 10) == (1.0, 0.0, 0.0)
        # 更近的像素覆盖成功
        assert fb.write_pixel(10, 10, (0.0, 0.0, 1.0), depth=0.3)
        assert fb.read_pixel(10, 10) == (0.0, 0.0, 1.0)
        assert fb.read_depth(10, 10) == pytest.approx(0.3)

    def test_case_16(self):
        """帧缓冲带模板缓冲，并且完整管线 render() 的结果可保存读取。"""
        fb = self.system.framebuffer
        fb.write_pixel(5, 5, (1.0, 1.0, 1.0), depth=0.1, stencil=1)
        assert fb.read_stencil(5, 5) == 1
        assert fb.read_stencil(6, 6) == 0

        written = self.system.render([make_triangle(z=0.0)])
        assert written > 0
        pixel = fb.read_pixel(32, 32)
        assert pixel != (0.0, 0.0, 0.0)  # 渲染结果已保存到颜色缓冲
        assert fb.read_depth(32, 32) < 1.0  # 深度缓冲已更新


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
