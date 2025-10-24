# -*- coding: utf-8 -*-
"""
颜色归一化工具
支持两种算法：
  1. Reinhard：在 LAB 空间对齐均值/方差
  2. Macenko：基于 PCA 提取染色矩阵并标准化浓度
作者：shannon yxn
日期：2025-10
"""

import cv2
import numpy as np
from sklearn.decomposition import PCA


class StainNormalizer:
    """
    Reinhard / Macenko 颜色标准化
    支持两种调用方式：
      1. 单张图：norm_img = sn.normalize(img, method='reinhard')
      2. 批量图：sn.batch_normalize(src_dir, dst_dir, method='macenko')
    """

    def __init__(self):
        """
        初始化归一化器
        默认无参考图像，可在后续调用 set_reference() 设置
        """
        self.ref_img = None

    # -------------------- 单张接口 --------------------
    def normalize(self, img_bgr: np.ndarray, method: str = 'reinhard') -> np.ndarray:
        """
        对单张图片做颜色归一化

        :param img_bgr: 原始 BGR 图像
        :param method:  'reinhard' | 'macenko'
        :return:        归一化后的 BGR 图像
        """
        img = img_bgr[:, :, ::-1]  # BGR→RGB
        if method == 'reinhard':
            return self._reinhard(img)
        elif method == 'macenko':
            return self._macenko(img)
        else:
            raise ValueError('method must be reinhard or macenko')

    def set_reference(self, ref_path: str):
        """
        设置 Reinhard 参考图像

        :param ref_path: 参考图片路径
        """
        self.ref_img = cv2.imread(ref_path)[:, :, ::-1]  # BGR→RGB

    # -------------------- 批量接口 --------------------
    def batch_normalize(self,
                        src_dir: str,
                        dst_dir: str,
                        method: str = 'reinhard',
                        ext: Tuple[str, ...] = ('.png', '.jpg')):
        """
        对目录内所有图片批量归一化并保存

        :param src_dir: 输入目录
        :param dst_dir: 输出目录（自动创建）
        :param method:  算法选择
        :param ext:     处理的后缀
        """
        import os
        import glob

        os.makedirs(dst_dir, exist_ok=True)
        for p in glob.glob(os.path.join(src_dir, '*')):
            if not p.lower().endswith(ext):
                continue
            img = cv2.imread(p)
            if img is None:
                continue
            norm = self.normalize(img, method=method)
            # RGB→BGR 后保存
            cv2.imwrite(os.path.join(dst_dir, os.path.basename(p)), norm[:, :, ::-1])

    # -------------------- 私有算法实现 --------------------
    def _reinhard(self, src_rgb: np.ndarray) -> np.ndarray:
        """
        Reinhard 归一化（LAB 空间）

        步骤：
        1. RGB→LAB
        2. 计算 src 均值/方差
        3. 若有参考图则使用参考统计量，否则使用默认
        4. 标准化后转回 RGB

        :param src_rgb: 原始 RGB 图像
        :return:        归一化 RGB 图像
        """
        src_lab = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        src_mean, src_std = src_lab.mean((0, 1)), src_lab.std((0, 1))

        # 目标统计量
        if self.ref_img is None:
            tgt_mean, tgt_std = np.array([128, 128, 128]), np.array([50, 50, 50])
        else:
            ref_lab = cv2.cvtColor(self.ref_img, cv2.COLOR_RGB2LAB).astype(np.float32)
            tgt_mean, tgt_std = ref_lab.mean((0, 1)), ref_lab.std((0, 1))

        # 标准化
        norm = (src_lab - src_mean) * (tgt_std / (src_std + 1e-6)) + tgt_mean
        return cv2.cvtColor(np.clip(norm, 0, 255).astype('uint8'), cv2.COLOR_LAB2RGB)

    def _macenko(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Macenko 归一化（PCA 提取染色向量）

        步骤：
        1. 光学密度 OD = -log((I+1)/255)
        2. PCA 降维到 2 维，得到主要染色方向
        3. 根据角度百分位确定两种染色向量
        4. 最小二乘解浓度后按 99 分位归一化
        5. 重构并裁剪到有效范围

        :param img_rgb: 原始 RGB 图像
        :return:        归一化 RGB 图像
        """
        # 光学密度
        od = -np.log((img_rgb.astype(np.float32) + 1) / 255)
        flat = od.reshape(-1, 3)

        # PCA 提取主成分
        pca = PCA(n_components=2).fit(flat)
        phi = pca.components_  # 2×3
        angles = np.arctan2(phi[:, 1], phi[:, 0])  # 2D 角度

        # 百分位阈值
        alpha, beta = 0.01, 0.15
        min_ang = np.percentile(angles, alpha * 100)
        max_ang = np.percentile(angles, (1 - beta) * 100)

        # 两种染色向量（3D 扩展）
        stain1 = np.array([np.cos(min_ang), np.sin(min_ang), 0])
        stain2 = np.array([np.cos(max_ang), np.sin(max_ang), 0])
        stains = np.vstack([stain1, stain2]).T  # 3×2

        # 最小二乘解浓度
        conc = np.linalg.lstsq(stains, flat.T, rcond=None)[0]  # 2×N
        # 99 分位归一化
        conc_norm = conc / np.percentile(conc, 99, axis=1)[:, None]

        # 重构光学密度
        od_norm = (stains @ conc_norm).T
        recon = 255 * np.exp(-od_norm)
        return np.clip(recon, 0, 255).reshape(img_rgb.shape).astype('uint8')


