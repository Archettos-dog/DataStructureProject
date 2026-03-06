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

#3.Max Pooling（纯 NumPy 实现）
def my_maxpool2d(img: np.ndarray,
                 kernel_size: int = 2,
                 stride: int = 2) -> np.ndarray:
    """
    2D 最大池化（无 padding，无权重）

    参数
    ----
    img         : (H, W) 输入图
    kernel_size : 池化窗口大小 K
    stride      : 步长 S

    返回
    ----
    out         : (H_out, W_out) 池化结果
    """
    H, W = img.shape
    K = kernel_size
    S = stride

    # 输出尺寸（与卷积公式相同，P=0）
    H_out = (H - K) // S + 1
    W_out = (W - K) // S + 1

    out = np.zeros((H_out, W_out), dtype=np.float32)
    for i in range(H_out):
        for j in range(W_out):
            region = img[i * S : i * S + K,
                         j * S : j * S + K]
            out[i, j] = np.max(region)                 # 取最大值

    return out

#4.主实验
def main():
    print("=" * 50)
    print("手写卷积算子实验")
    print("=" * 50)

    # 4-1. 加载图像
    print("\n[Step 1] 加载并预处理图像")
    img = load_image(target_size=(128, 128))

    # 4-2. 定义 Sobel 核
    print("\n[Step 2] 定义 Sobel 卷积核")
    sobel_x = np.array([[-1,  0,  1],
                         [-2,  0,  2],
                         [-1,  0,  1]], dtype=np.float32)   # 检测垂直边缘

    sobel_y = np.array([[-1, -2, -1],
                         [ 0,  0,  0],
                         [ 1,  2,  1]], dtype=np.float32)   # 检测水平边缘

    print(f"  Sobel X (垂直边缘检测):\n{sobel_x}")
    print(f"  Sobel Y (水平边缘检测):\n{sobel_y}")

    # 4-3. 调用 my_conv2d（padding=1 保持尺寸）
    print("\n[Step 3] 执行卷积（stride=1, padding=1）")
    H, W = img.shape
    P, S, K = 1, 1, 3
    H_out_expected = int(np.floor((H + 2*P - K) / S)) + 1
    W_out_expected = int(np.floor((W + 2*P - K) / S)) + 1
    print(f"  输入: ({H}, {W})  核: ({K},{K})  P={P}  S={S}")
    print(f"  预期输出尺寸: H_out=({H}+2×{P}-{K})/{S}+1={H_out_expected}, "
          f"W_out={W_out_expected}")

    feat_x = my_conv2d(img, sobel_x, stride=S, padding=P)
    feat_y = my_conv2d(img, sobel_y, stride=S, padding=P)
    print(f"  Sobel X 特征图尺寸: {feat_x.shape}  ✓")
    print(f"  Sobel Y 特征图尺寸: {feat_y.shape}  ✓")

    # 4-4. Max Pooling
    print("\n[Step 4] 执行 Max Pooling（kernel=2, stride=2）")
    pooled = my_maxpool2d(img, kernel_size=2, stride=2)
    print(f"  输入尺寸: {img.shape} → 池化后: {pooled.shape}  ✓")

    # 4-5. 可视化
    print("\n[Step 5] 绘制结果图")
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("Handwritten Conv2D Experiment Results", fontsize=14)

    axes[0].imshow(img, cmap='gray')
    axes[0].set_title(f"Original\n{img.shape}")
    axes[0].axis('off')

    axes[1].imshow(np.abs(feat_x), cmap='gray')
    axes[1].set_title(f"Sobel X (Vertical Edges)\n{feat_x.shape}")
    axes[1].axis('off')

    axes[2].imshow(np.abs(feat_y), cmap='gray')
    axes[2].set_title(f"Sobel Y (Horizontal Edges)\n{feat_y.shape}")
    axes[2].axis('off')

    axes[3].imshow(pooled, cmap='gray')
    axes[3].set_title(f"Max Pooling (2x2)\n{pooled.shape}")
    axes[3].axis('off')

    plt.tight_layout()
    out_path = "/mnt/user-data/outputs/conv2d_result.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"  结果已保存至 {out_path}")

    # 4-6. 维度计算演示（不同 P/S 组合）
    print("\n[Step 6] 维度计算公式演示")
    print(f"{'P':>3} {'S':>3} {'K':>3} | {'H_out':>6} {'W_out':>6}")
    print("-" * 28)
    for p in [0, 1, 2]:
        for s in [1, 2]:
            ho = int(np.floor((H + 2*p - K) / s)) + 1
            wo = int(np.floor((W + 2*p - K) / s)) + 1
            print(f"{p:>3} {s:>3} {K:>3} | {ho:>6} {wo:>6}")

    print("\n实验完成！")


if __name__ == "__main__":
    main()