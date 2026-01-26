import cv2
from typing import List, Dict
from models.base import BaseYoloModel
from models.asleep.v1.face import faceCheck_getEAR


class AsleepModelV1(BaseYoloModel):
    """封装基于 face.py 的疲劳检测，使其与其它模型接口一致（predict 返回 detections 列表）。"""

    def __init__(self, conf_threshold: float = 0.5):
        self.conf_threshold = conf_threshold

    def predict(self, image_path: str) -> List[Dict]:
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            return []

        # faceCheck_getEAR 接受图像列表并返回 (flag, bbox)
        try:
            res, face_bbox = faceCheck_getEAR([img])
        except Exception:
            res, face_bbox = 0, None

        h, w = img.shape[:2]

        if res == 1:
            if face_bbox:
                x1, y1, x2, y2 = face_bbox
            else:
                x1, y1, x2, y2 = 0, 0, w, h

            return [{
                "class_id": 0,
                "confidence": 0.99,
                "bbox": [float(x1), float(y1), float(x2), float(y2)]
            }]
        else:
            return []
