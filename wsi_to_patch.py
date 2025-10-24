# wsi_to_patch_openslide_parallel.py

import os
import numpy as np
import openslide
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from skimage.morphology import binary_dilation, binary_erosion, remove_small_holes, remove_small_objects, disk
from PIL import Image, ImageDraw
from concurrent.futures import ProcessPoolExecutor


# -------------------- 工具函数 --------------------

def color_filter(img_np):
    """蓝色区域保留，其他置白"""
    if len(img_np.shape) == 2 or img_np.shape[2] == 1:
        return img_np
    hsv_img = rgb2hsv(img_np / 255.0)
    hue = hsv_img[:, :, 0] * 360
    color_mask = (hue >= 230) & (hue <= 325)  # 蓝色范围
    out = img_np.copy()
    out[~color_mask] = 255
    return out


def tissue_mask(img_np):
    """组织掩膜生成 pipeline"""
    # 1. 蓝色过滤
    img_np = color_filter(img_np)
    # 2. 转灰度
    gray = np.mean(img_np, axis=2)
    # 3. Otsu 二值化
    thresh = threshold_otsu(gray)
    mask = gray < thresh
    # 4. 形态学操作
    mask = binary_erosion(mask, disk(0))
    mask = binary_dilation(mask, disk(1))
    mask = remove_small_holes(mask, area_threshold=100)
    mask = remove_small_objects(mask, min_size=1500)
    return mask


# -------------------- 单 tile 处理函数 --------------------

def process_tile(args):
    x, y, slide_path, tile_size, tissue_thresh, scored_path = args

    slide = openslide.OpenSlide(slide_path)   # 每个进程重新打开
    region = slide.read_region((x, y), 0, (tile_size, tile_size)).convert("RGB")
    region_np = np.array(region)

    mask = tissue_mask(region_np)

    if mask.mean() > tissue_thresh:
        save_path = os.path.join(scored_path, f"tile_{x}_{y}.png")
        region.save(save_path)

    slide.close()
    return (x, y, mask.mean() > tissue_thresh)


# -------------------- 主函数 --------------------

def histolab_func(input, processed_path, i, num_workers=8):
    """
    输入 WSI，输出处理结果 (多进程, level=0)
    """
    folder_name = os.path.basename(input)
    scored_path = os.path.join(processed_path, f"raw{i}_{folder_name}")
    os.makedirs(scored_path, exist_ok=True)

    slide = openslide.OpenSlide(input)
    w, h = slide.dimensions

    # Tile 参数
    tile_size = 860
    overlap = 150
    tissue_thresh = 0.38

    # 定位图初始化
    thumbnail = slide.get_thumbnail((w // 6, h // 6)).convert("RGB")
    locate_img = thumbnail.copy()
    draw = ImageDraw.Draw(locate_img)
    slide.close()

    # 构建任务列表
    tasks = [(x, y, input, tile_size, tissue_thresh, scored_path)
             for y in range(0, h, tile_size - overlap)
             for x in range(0, w, tile_size - overlap)]

    # 多进程执行
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_tile, tasks))

    # 在定位图绘制红框
    for (x, y, kept) in results:
        if kept:
            rect = [x // 6, y // 6, (x + tile_size) // 6, (y + tile_size) // 6]
            draw.rectangle(rect, outline="red", width=2)

    # 保存定位图
    output_image_path = os.path.join(processed_path, f"located_tiles{i}.png")
    locate_img.save(output_image_path)
    print(f"定位结果已保存至: {output_image_path}")
    print(f"Tile 提取完成，结果保存在: {scored_path}")
