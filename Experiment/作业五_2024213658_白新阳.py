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
