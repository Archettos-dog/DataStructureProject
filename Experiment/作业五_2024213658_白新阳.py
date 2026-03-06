import numpy as np
import matplotlib
matplotlib.use('Agg')   
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

#1.数据加载与预处理
def load_image(target_size=(128, 128)):
    """加载内置灰度图像并预处理至 (128,128) float32 [0,1]"""
    img = None
    try:
        from scipy.datasets import ascent
        img = ascent().astype(np.float32)
        print("✓ 使用 scipy.datasets.ascent()")
    except Exception:
        pass

    if img is None:
        try:
            from scipy.misc import ascent
            img = ascent().astype(np.float32)
            print("✓ 使用 scipy.misc.ascent()")
        except Exception:
            pass

    if img is None:
        from skimage.data import camera
        img = camera().astype(np.float32)
        print("✓ 使用 skimage.data.camera()")

    # Resize：用 PIL 降采样
    from PIL import Image as _Image
    pil = _Image.fromarray(img.astype(np.uint8))
    pil = pil.resize((target_size[1], target_size[0]), _Image.BILINEAR)
    img_resized = np.array(pil, dtype=np.float32)

    # 归一化至 [0, 1]
    img_norm = img_resized / 255.0
    print(f"  图像尺寸: {img_norm.shape}，值域: [{img_norm.min():.3f}, {img_norm.max():.3f}]")
    return img_norm

#2.2D卷积（纯 NumPy 实现）
def my_conv2d(img: np.ndarray, kernel: np.ndarray,
              stride: int = 1, padding: int = 0) -> np.ndarray:
    """
    2D 卷积前向传播（不调用任何深度学习框架）

    参数
    ----
    img     : (H, W)  输入灰度图
    kernel  : (K, K)  卷积核（假设为方形）
    stride  : 步长 S
    padding : 零填充宽度 P

    返回
    ----
    out     : (H_out, W_out) 卷积结果
    """
    H, W = img.shape
    K = kernel.shape[0]
    assert kernel.shape == (K, K), "卷积核必须为方形"
    S = stride
    P = padding

    # ── 零填充 ──────────────────────────────
    if P > 0:
        padded = np.pad(img, ((P, P), (P, P)), 'constant')
    else:
        padded = img
    H_pad, W_pad = padded.shape

    # ── 输出尺寸公式 ─────────────────────────
    # H_out = floor((H + 2P - K) / S) + 1
    H_out = int(np.floor((H + 2 * P - K) / S)) + 1
    W_out = int(np.floor((W + 2 * P - K) / S)) + 1

    # ── 断言验证 ─────────────────────────────
    assert H_out == (H_pad - K) // S + 1, "H_out 计算错误"
    assert W_out == (W_pad - K) // S + 1, "W_out 计算错误"

    # ── 卷积计算（双重循环 + NumPy 切片）───────
    out = np.zeros((H_out, W_out), dtype=np.float32)
    for i in range(H_out):
        for j in range(W_out):
            region = padded[i * S : i * S + K,
                            j * S : j * S + K]        # (K, K) 感受野
            out[i, j] = np.sum(region * kernel)        # 点乘求和

    return out

