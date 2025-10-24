# -*- coding: utf-8 -*-
"""
WSI 切片提取工具（多进程并行版）
功能：
  1. 蓝色组织区域过滤
  2. 组织掩膜生成（Otsu + 形态学）
  3. 多进程并行切片保存
  4. 生成定位缩略图（红框标注有效 tile）
作者：yrt shannon yxn
日期：2025-10
"""

import os
import numpy as np
import openslide
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from skimage.morphology import (
    binary_dilation, binary_erosion, remove_small_holes,
    remove_small_objects, disk
)
from PIL import Image, ImageDraw
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple


# ------------------------------------------------------------------
# 1. 颜色过滤：保留蓝色区域（苏木精染色），其余置白
# ------------------------------------------------------------------
def color_filter(img_np: np.ndarray) -> np.ndarray:
    """
    蓝色区域保留，其他置白

    :param img_np: RGB 图像，uint8
    :return:       过滤后 RGB 图像
    """
    # 灰度图直接返回
    if len(img_np.shape) == 2 or img_np.shape[2] == 1:
        return img_np

    # HSV 空间提取蓝色
    hsv_img = rgb2hsv(img_np / 255.0)
    hue = hsv_img[:, :, 0] * 360  # 0~360
    color_mask = (hue >= 230) & (hue <= 325)  # 蓝色范围
    out = img_np.copy()
    out[~color_mask] = 255  # 非蓝色置白
    return out


# ------------------------------------------------------------------
# 2. 组织掩膜生成：综合颜色过滤、Otsu、形态学后处理
# ------------------------------------------------------------------
def tissue_mask(img_np: np.ndarray) -> np.ndarray:
    """
    组织掩膜生成 pipeline

    :param img_np: RGB 图像
    :return:       二值掩膜，True 表示组织区域
    """
    # 1. 蓝色过滤
    img_np = color_filter(img_np)
    # 2. 转灰度
    gray = np.mean(img_np, axis=2)
    # 3. Otsu 自动阈值
    thresh = threshold_otsu(gray)
    mask = gray < thresh
    # 4. 形态学后处理
    mask = binary_erosion(mask, disk(0))      # 去除孤点
    mask = binary_dilation(mask, disk(1))     # 弥合小缝
    mask = remove_small_holes(mask, area_threshold=100)     # 填小洞
    mask = remove_small_objects(mask, min_size=1500)        # 去小对象
    return mask


# ------------------------------------------------------------------
# 3. 单 tile 处理函数（进程入口）
# ------------------------------------------------------------------
def process_tile(args: Tuple) -> Tuple[int, int, bool]:
    """
    单个 tile 的处理流程：读取 → 掩膜计算 → 保存（若合格）

    :param args: (x, y, slide_path, tile_size, tissue_thresh, scored_path)
    :return:     (x, y, 是否保留)
    """
    x, y, slide_path, tile_size, tissue_thresh, scored_path = args

    # 每个进程重新打开 slide，避免句柄冲突
    slide = openslide.OpenSlide(slide_path)
    region = slide.read_region((x, y), 0, (tile_size, tile_size)).convert("RGB")
    region_np = np.array(region)

    # 计算组织掩膜
    mask = tissue_mask(region_np)

    # 组织面积占比 > 阈值则保存
    kept = mask.mean() > tissue_thresh
    if kept:
        save_path = os.path.join(scored_path, f"tile_{x}_{y}.png")
        region.save(save_path)

    slide.close()
    return (x, y, kept)


# ------------------------------------------------------------------
# 4. 主函数：WSI → 切片 + 定位图
# ------------------------------------------------------------------
def histolab_func(input: str,
                  processed_path: str,
                  i: int,
                  num_workers: int = 8):
    """
    输入单张 WSI，输出切片结果（多进程并行，level=0）

    :param input:          WSI 文件路径
    :param processed_path: 结果保存根目录
    :param i:              当前 WSI 序号（用于命名）
    :param num_workers:    进程数
    """
    # 结果子目录：raw{i}_文件名
    folder_name = os.path.basename(input)
    scored_path = os.path.join(processed_path, f"raw{i}_{folder_name}")
    os.makedirs(scored_path, exist_ok=True)

    # 打开 WSI 获取尺寸
    slide = openslide.OpenSlide(input)
    w, h = slide.dimensions

    # Tile 参数
    tile_size = 860      # 正方形切片边长（像素）
    overlap = 150        # 相邻 tile 重叠
    tissue_thresh = 0.38  # 组织面积占比阈值

    # 生成缩略图（1/6 尺寸）用于定位图绘制
    thumbnail = slide.get_thumbnail((w // 6, h // 6)).convert("RGB")
    locate_img = thumbnail.copy()
    draw = ImageDraw.Draw(locate_img)
    slide.close()

    # 构建任务列表：从左到右、从上到下扫描
    tasks: List[Tuple] = [
        (x, y, input, tile_size, tissue_thresh, scored_path)
        for y in range(0, h, tile_size - overlap)
        for x in range(0, w, tile_size - overlap)
    ]

    # 多进程并行处理
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_tile, tasks))

    # 在缩略图上绘制红框（坐标按比例缩放）
    scale = 1 / 6
    for (x, y, kept) in results:
        if kept:
            rect = [
                x * scale, y * scale,
                (x + tile_size) * scale, (y + tile_size) * scale
            ]
            draw.rectangle(rect, outline="red", width=2)

    # 保存定位图
    output_image_path = os.path.join(processed_path, f"located_tiles{i}.png")
    locate_img.save(output_image_path)
    print(f"定位结果已保存至: {output_image_path}")
    print(f"Tile 提取完成，结果保存在: {scored_path}")

