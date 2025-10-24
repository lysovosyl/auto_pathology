import cv2, numpy as np
from sklearn.decomposition import PCA

class StainNormalizer:
    """
    Reinhard / Macenko 颜色标准化
    支持两种调用方式：
      1. 单张图：norm_img = sn.normalize(img, method='reinhard')
      2. 批量图：sn.batch_normalize(src_dir, dst_dir, method='macenko')
    """
    def __init__(self):
        self.ref_img = None

    def set_reference(self, ref_path: str):
        self.ref_img = cv2.imread(ref_path)[:, :, ::-1]   # BGR→RGB

    def normalize(self, img_bgr: np.ndarray, method: str = 'reinhard') -> np.ndarray:
        img = img_bgr[:, :, ::-1]  # BGR→RGB
        if method == 'reinhard':
            return self._reinhard(img)
        elif method == 'macenko':
            return self._macenko(img)
        else:
            raise ValueError('method must be reinhard or macenko')

    # —— 批量接口 ——
    def batch_normalize(self, src_dir: str, dst_dir: str,
                        method: str = 'reinhard', ext=('.png', '.jpg')):
        import os, glob
        os.makedirs(dst_dir, exist_ok=True)
        for p in glob.glob(os.path.join(src_dir, '*')):
            if not p.lower().endswith(ext):
                continue
            img = cv2.imread(p)
            if img is None:
                continue
            norm = self.normalize(img, method=method)
            cv2.imwrite(os.path.join(dst_dir, os.path.basename(p)), norm[:, :, ::-1])

    # —— 私有算法实现 ——
    def _reinhard(self, src_rgb):
        src_lab = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        src_mean, src_std = src_lab.mean((0, 1)), src_lab.std((0, 1))
        if self.ref_img is None:
            tgt_mean, tgt_std = np.array([128, 128, 128]), np.array([50, 50, 50])
        else:
            ref_lab = cv2.cvtColor(self.ref_img, cv2.COLOR_RGB2LAB).astype(np.float32)
            tgt_mean, tgt_std = ref_lab.mean((0, 1)), ref_lab.std((0, 1))
        norm = (src_lab - src_mean) * (tgt_std / (src_std + 1e-6)) + tgt_mean
        return cv2.cvtColor(np.clip(norm, 0, 255).astype('uint8'), cv2.COLOR_LAB2RGB)

    def _macenko(self, img_rgb):
        od = -np.log((img_rgb.astype(np.float32) + 1) / 255)
        flat = od.reshape(-1, 3)
        pca = PCA(n_components=2).fit(flat)
        phi = pca.components_
        angles = np.arctan2(phi[:, 1], phi[:, 0])
        alpha, beta = 0.01, 0.15
        min_ang, max_ang = np.percentile(angles, alpha * 100), np.percentile(angles, (1 - beta) * 100)
        stain1 = np.array([np.cos(min_ang), np.sin(min_ang), 0])
        stain2 = np.array([np.cos(max_ang), np.sin(max_ang), 0])
        stains = np.vstack([stain1, stain2]).T
        conc = np.linalg.lstsq(stains, flat.T, rcond=None)[0]
        conc_norm = conc / np.percentile(conc, 99, axis=1)[:, None]
        od_norm = (stains @ conc_norm).T
        recon = 255 * np.exp(-od_norm)
        return recon.reshape(img_rgb.shape).astype('uint8')