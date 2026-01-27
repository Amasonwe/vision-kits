import cv2
import os
from typing import List, Dict
from models.base import BaseYoloModel
from models.asleep.v1.face import faceCheck_getEAR


class AsleepModelV1(BaseYoloModel):
    """封装基于 face.py 的疲劳检测，使其与其它模型接口一致（predict 返回 detections 列表）。"""

    def __init__(self, conf_threshold: float = 0.5):
        self.conf_threshold = conf_threshold

    def predict(self, image_path: str) -> List[Dict]:
        # 读取图片
        # 如果传入的是视频文件（mp4/avi/mov 等），直接把路径传给 faceCheck_getEAR
        suffix = os.path.splitext(image_path)[1].lower()
        video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}

        if suffix in video_exts:
            try:
                res, face_bbox = faceCheck_getEAR(image_path)
            except Exception:
                res, face_bbox = 0, None
            # get video frame size for fallback bbox
            try:
                cap = cv2.VideoCapture(image_path)
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
            except Exception:
                w, h = 0, 0
        else:
            img = cv2.imread(image_path)
            if img is None:
                # 兼容旧错误信息：打开图片失败
                print(f"Warning: failed to open image: {image_path}")
                return []

            # faceCheck_getEAR 接受图像列表并返回 (flag, bbox)
            try:
                res, face_bbox = faceCheck_getEAR([img])
            except Exception:
                res, face_bbox = 0, None

        # determine image/video size
        if 'img' in locals() and img is not None:
            h, w = img.shape[:2]
        else:
            # for video branch we tried to set w,h above
            try:
                h = int(h)
                w = int(w)
            except Exception:
                h, w = 0, 0

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
