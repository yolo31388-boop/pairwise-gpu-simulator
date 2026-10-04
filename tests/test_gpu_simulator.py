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
    Framebuffer,
    mat4_identity,
    translation_matrix,
    perspective_matrix,
)


def make_triangle(gpu, z=0.0):
    """NDC空间中的大三角形，经顶点处理器转为屏幕空间。"""
    verts = [
        {"position": (-0.5, -0.5, z), "color": (255.0, 0.0, 0.0), "uv": (0.0, 0.0)},
        {"position": (0.0, 0.5, z), "color": (0.0, 0.0, 255.0), "uv": (0.5, 1.0)},
        {"position": (0.5, -0.5, z), "color": (0.0, 255.0, 0.0), "uv": (1.0, 0.0)},
    ]
    return gpu.vertex_processor(verts)


class TestGPUSimulator:
    def setup_method(self):
        self.system = GPUSimulator({"width": 64, "height": 64})

    # -- 顶点处理器 --------------------------------------------------------

    def test_case_01(self):
        """顶点处理器：应用MVP矩阵变换（平移生效）"""
        verts = [{"position": (0.0, 0.0, 0.0)}]
        base = self.system.vertex_processor(verts)[0]
        moved = self.system.vertex_processor(
            verts, mvp=translation_matrix(0.5, 0.0, 0.0))[0]
        assert base["screen"][0] == pytest.approx(32.0)
        assert moved["screen"][0] == pytest.approx(48.0)
        assert moved["screen"][1] == pytest.approx(base["screen"][1])

    def test_case_02(self):
        """顶点处理器：透视投影矩阵生效（透视除法+视口变换）"""
        mvp = perspective_matrix(math.radians(90.0), 1.0, 1.0, 100.0)
        out = self.system.vertex_processor(
            [{"position": (0.5, 0.0, -1.0)}], mvp=mvp)[0]
        # fov=90 -> f=1, clip=(0.5, 0, -1, 1) -> ndc=(0.5, 0, -1)
        assert out["screen"][0] == pytest.approx(48.0)
        assert out["screen"][1] == pytest.approx(32.0)
        assert out["depth"] == pytest.approx(0.0)

    def test_case_03(self):
        """顶点处理器：支持自定义顶点着色器"""
        def double_shader(vertex, mvp):
            x, y, z = vertex["position"]
            return (x * 2.0, y * 2.0, z, 1.0)
        out = self.system.vertex_processor(
            [{"position": (0.25, 0.25, 0.0)}], vertex_shader=double_shader)[0]
        assert out["screen"][0] == pytest.approx(48.0)
        assert out["screen"][1] == pytest.approx(16.0)

    # -- 光栅化器 ----------------------------------------------------------

    def test_case_04(self):
        """光栅化器：三角形被转换为覆盖其区域的像素片元"""
        frags = self.system.rasterizer(make_triangle(self.system))
        assert len(frags) > 0
        xs = {f["x"] for f in frags}
        ys = {f["y"] for f in frags}
        assert min(xs) >= 0 and max(xs) < 64
        assert min(ys) >= 0 and max(ys) < 64
        # 三角形质心附近应被覆盖
        assert any(abs(f["x"] - 32) <= 1 and abs(f["y"] - 32) <= 1 for f in frags)

    def test_case_05(self):
        """光栅化器：重心插值深度与顶点属性"""
        verts = [
            {"screen": (0.5, 0.5), "depth": 0.2, "color": (255.0, 0.0, 0.0)},
            {"screen": (6.5, 0.5), "depth": 0.4, "color": (0.0, 255.0, 0.0)},
            {"screen": (0.5, 6.5), "depth": 0.8, "color": (0.0, 0.0, 255.0)},
        ]
        frags = self.system.rasterizer(verts)
        # 质心 (2.5,2.5) 即像素(2,2)中心，重心权重各为 1/3
        frag = next(f for f in frags if f["x"] == 2 and f["y"] == 2)
        assert frag["depth"] == pytest.approx((0.2 + 0.4 + 0.8) / 3, abs=0.05)
        assert frag["color"][0] == pytest.approx(85.0, abs=1.0)
        assert frag["color"][1] == pytest.approx(85.0, abs=1.0)
        assert frag["color"][2] == pytest.approx(85.0, abs=1.0)

    def test_case_06(self):
        """光栅化器：深度测试，近处物体覆盖远处物体"""
        fb = self.system.framebuffer()
        far = self.system.rasterizer(make_triangle(self.system, z=0.5))
        near = self.system.rasterizer(make_triangle(self.system, z=-0.5))
        assert len(far) > 0 and len(near) > 0
        # 再画一次远处三角形，所有片元都应被深度测试丢弃
        blocked = self.system.rasterizer(make_triangle(self.system, z=0.5))
        assert blocked == []
        # 深度缓冲中保存的是近处深度
        assert fb.read_depth(32, 32) == pytest.approx(0.25)

    def test_case_07(self):
        """光栅化器：背面剔除，顺时针三角形被剔除"""
        ccw = make_triangle(self.system)
        assert len(self.system.rasterizer(ccw, cull_backface=True)) > 0
        cw = [ccw[0], ccw[2], ccw[1]]  # 交换顶点顺序 -> 背面
        assert self.system.rasterizer(cw, cull_backface=True) == []
        # 关闭剔除时背面也能渲染（关掉深度测试避免被前一次绘制挡住）
        assert len(self.system.rasterizer(cw, cull_backface=False, depth_test=False)) > 0

    # -- 片元处理器 ----------------------------------------------------------

    def test_case_08(self):
        """片元处理器：顶点颜色在三角形内部平滑插值"""
        verts = [
            {"screen": (0.0, 0.0), "depth": 0.5, "color": (255.0, 0.0, 0.0)},
            {"screen": (32.0, 0.0), "depth": 0.5, "color": (0.0, 255.0, 0.0)},
            {"screen": (0.0, 32.0), "depth": 0.5, "color": (0.0, 0.0, 255.0)},
        ]
        frags = self.system.rasterizer(verts)
        pixels = self.system.fragment_processor(frags)
        colors = {tuple(round(c) for c in p["color"]) for p in pixels}
        # 平滑过渡：应出现多种中间色而非固定颜色
        assert len(colors) > 10
        p = next(p for p in pixels if p["x"] == 10 and p["y"] == 10)
        assert p["color"][0] == pytest.approx(255.0 * (1 - 10.5 / 32 - 10.5 / 32), abs=2.0)

    def test_case_09(self):
        """片元处理器：自定义片元着色器计算光照"""
        def lighting_shader(frag):
            base = frag.get("color", (255.0, 255.0, 255.0))
            intensity = 0.5  # 模拟漫反射光照
            return tuple(c * intensity for c in base)
        frags = self.system.rasterizer(make_triangle(self.system))
        pixels = self.system.fragment_processor(frags, fragment_shader=lighting_shader)
        assert len(pixels) > 0
        for p in pixels:
            assert all(0.0 <= c <= 128.0 for c in p["color"])

    # -- 纹理采样器 ----------------------------------------------------------

    TEXTURE = [
        [(0.0, 0.0, 0.0), (255.0, 255.0, 255.0)],
        [(255.0, 0.0, 0.0), (0.0, 255.0, 0.0)],
    ]

    def test_case_10(self):
        """纹理采样：最近邻过滤返回精确的纹素颜色"""
        c = self.system.texture_sampler(self.TEXTURE, 0.25, 0.25, filter_mode="nearest")
        assert c == (0.0, 0.0, 0.0)
        c = self.system.texture_sampler(self.TEXTURE, 0.75, 0.25, filter_mode="nearest")
        assert c == (255.0, 255.0, 255.0)
        c = self.system.texture_sampler(self.TEXTURE, 0.75, 0.75, filter_mode="nearest")
        assert c == (0.0, 255.0, 0.0)

    def test_case_11(self):
        """纹理采样：双线性过滤在纹素间平滑混合"""
        c = self.system.texture_sampler(self.TEXTURE, 0.5, 0.5, filter_mode="bilinear")
        assert c[0] == pytest.approx(127.5)
        assert c[1] == pytest.approx(127.5)
        assert c[2] == pytest.approx(63.75)

    def test_case_12(self):
        """纹理采样：repeat环绕，超出UV范围不再报错"""
        c = self.system.texture_sampler(self.TEXTURE, 1.25, 2.25,
                                        filter_mode="nearest", wrap_mode="repeat")
        assert c == (0.0, 0.0, 0.0)
        c = self.system.texture_sampler(self.TEXTURE, -0.25, 0.25,
                                        filter_mode="nearest", wrap_mode="repeat")
        assert c == (255.0, 255.0, 255.0)

    def test_case_13(self):
        """纹理采样：clamp环绕，边缘颜色被钳制延伸"""
        c = self.system.texture_sampler(self.TEXTURE, 1.5, 0.25,
                                        filter_mode="nearest", wrap_mode="clamp")
        assert c == (255.0, 255.0, 255.0)
        c = self.system.texture_sampler(self.TEXTURE, -0.5, 0.75,
                                        filter_mode="nearest", wrap_mode="clamp")
        assert c == (255.0, 0.0, 0.0)

    # -- 帧缓冲 --------------------------------------------------------------

    def test_case_14(self):
        """帧缓冲：颜色缓冲与深度缓冲可读写，深度测试生效"""
        fb = Framebuffer(8, 8)
        assert fb.write_pixel(3, 3, (255.0, 0.0, 0.0), depth=0.5)
        assert fb.read_pixel(3, 3) == (255.0, 0.0, 0.0)
        assert fb.read_depth(3, 3) == pytest.approx(0.5)
        # 更远的片元被深度测试拒绝
        assert not fb.write_pixel(3, 3, (0.0, 255.0, 0.0), depth=0.9)
        assert fb.read_pixel(3, 3) == (255.0, 0.0, 0.0)
        # 更近的片元写入成功
        assert fb.write_pixel(3, 3, (0.0, 0.0, 255.0), depth=0.2)
        assert fb.read_pixel(3, 3) == (0.0, 0.0, 255.0)

    def test_case_15(self):
        """帧缓冲：模板缓冲可读写，clear重置所有缓冲"""
        fb = Framebuffer(8, 8)
        fb.write_pixel(2, 2, (1.0, 1.0, 1.0), depth=0.1, stencil=3)
        assert fb.read_stencil(2, 2) == 3
        fb.write_stencil(4, 4, 7)
        assert fb.read_stencil(4, 4) == 7
        fb.clear()
        assert fb.read_stencil(2, 2) == 0
        assert fb.read_depth(2, 2) == pytest.approx(1.0)
        assert fb.read_pixel(2, 2) == (0.0, 0.0, 0.0)

    # -- 完整管线 --------------------------------------------------------------

    def test_case_16(self):
        """完整管线：渲染三角形到帧缓冲，颜色被保存且远处物体不覆盖近处"""
        gpu = self.system
        red = [
            {"position": (-0.8, -0.8, 0.0), "color": (255.0, 0.0, 0.0)},
            {"position": (0.0, 0.8, 0.0), "color": (255.0, 0.0, 0.0)},
            {"position": (0.8, -0.8, 0.0), "color": (255.0, 0.0, 0.0)},
        ]
        pixels = gpu.render_triangle(red)
        assert len(pixels) > 0
        fb = gpu.framebuffer()
        center = fb.read_pixel(32, 32)
        assert center[0] == pytest.approx(255.0)
        assert center[1] == pytest.approx(0.0)
        # 渲染一个更远的蓝色三角形，不应覆盖近处的红色
        blue = [dict(v, color=(0.0, 0.0, 255.0), position=(p[0], p[1], 0.5))
                for v in red for p in [v["position"]]]
        gpu.render_triangle(blue)
        assert fb.read_pixel(32, 32)[0] == pytest.approx(255.0)
        # reset 后帧缓冲被清空
        gpu.reset()
        assert gpu.framebuffer().read_pixel(32, 32) == (0.0, 0.0, 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
