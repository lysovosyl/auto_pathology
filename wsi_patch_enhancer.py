import os, cv2, numpy as np
from typing import List, Tuple

class ImageEnhancer:
    """
    对普通图像做增强（支持批量）
    """
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run(self, pipeline: List[str],
            extensions: Tuple[str, ...] = ('.jpg', '.png')):
        """
        pipeline: ['gaussian', 'median', 'hist_eq', 'usm_sharpen'] 按需填写
        """
        func_map = {
            'gaussian': lambda x: cv2.GaussianBlur(x, (5, 5), 0),
            'median': lambda x: self._median_lab(x),
            'hist_eq': lambda x: self._clahe_lab(x),
            'usm_sharpen': lambda x: self._usm(x)
        }

        for fname in os.listdir(self.input_dir):
            if not fname.lower().endswith(extensions):
                continue
            img = cv2.imread(os.path.join(self.input_dir, fname))
            if img is None:
                continue
            for step in pipeline:
                img = func_map[step](img)
            cv2.imwrite(os.path.join(self.output_dir, fname), img)

    # —— 以下私有方法可根据需要扩展 ——
    @staticmethod
    def _median_lab(img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.medianBlur(lab[:, :, 0], 5)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _clahe_lab(img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _usm(img):
        blur = cv2.GaussianBlur(img, (0, 0), 3)
        detail = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
        edges = cv2.Canny(img, 100, 200)
        return np.where(edges[..., None] != 0, img + 0.7 * detail, img)