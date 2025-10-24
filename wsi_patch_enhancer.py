# -*- coding: utf-8 -*-
"""
图像增强器：支持高斯模糊、中值滤波、CLAHE 直方图均衡、USM 锐化
作者：shannon yxn
日期：2025-10
"""

import os
import cv2
import numpy as np
from typing import List, Tuple


class ImageEnhancer:
    """
    对普通图像做批量增强：
    1. 高斯模糊
    2. L* 通道中值滤波（保色）
    3. L* 通道 CLAHE 直方图均衡
    4. USM 锐化（边缘增强）
    """

    def __init__(self, input_dir: str, output_dir: str):
        """
        初始化增强器

        :param input_dir: 原始图片目录
        :param output_dir: 增强结果保存目录
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ---------------------------- 公有入口 ----------------------------
    def run(self,
            pipeline: List[str],
            extensions: Tuple[str, ...] = ('.jpg', '.png')) -> None:
        """
        按指定顺序对目录内所有图片做增强

        :param pipeline: 增强步骤列表，可选值：
                         ['gaussian', 'median', 'hist_eq', 'usm_sharpen']
        :param extensions: 处理的文件后缀，默认 jpg/png
        :return: None，结果直接写入 output_dir
        """
        # 步骤名 -> 具体函数映射
        func_map = {
            'gaussian': lambda x: cv2.GaussianBlur(x, (5, 5), 0),
            'median': lambda x: self._median_lab(x),
            'hist_eq': lambda x: self._clahe_lab(x),
            'usm_sharpen': lambda x: self._usm(x)
        }

        # 遍历输入目录
        for fname in os.listdir(self.input_dir):
            if not fname.lower().endswith(extensions):
                continue

            img_path = os.path.join(self.input_dir, fname)
            img = cv2.imread(img_path)
            if img is None:          # 读图失败跳过
                continue

            # 按顺序执行增强
            for step in pipeline:
                img = func_map[step](img)

            # 保存结果
            save_path = os.path.join(self.output_dir, fname)
            cv2.imwrite(save_path, img)

    # ---------------------------- 私有实现 ----------------------------
    @staticmethod
    def _median_lab(img: np.ndarray) -> np.ndarray:
        """
        在 L* 通道做中值滤波，保持颜色信息

        :param img: BGR 图像
        :return: 处理后 BGR 图像
        """
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.medianBlur(lab[:, :, 0], 5)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _clahe_lab(img: np.ndarray) -> np.ndarray:
        """
        在 L* 通道做 CLAHE 直方图均衡，提升对比度

        :param img: BGR 图像
        :return: 处理后 BGR 图像
        """
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _usm(img: np.ndarray) -> np.ndarray:
        """
        USM 锐化：边缘区域增强细节

        :param img: BGR 图像
        :return: 锐化后 BGR 图像
        """
        blur = cv2.GaussianBlur(img, (0, 0), 3)                 # 低通
        detail = cv2.addWeighted(img, 1.5, blur, -0.5, 0)       # 细节层
        edges = cv2.Canny(img, 100, 200)                        # 边缘掩膜
        # 边缘处叠加细节
        return np.where(edges[..., None] != 0, img + 0.7 * detail, img)