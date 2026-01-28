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
        # 前置校验：路径存在且为文件
        if not os.path.exists(image_path):
            print(f"Warning: path does not exist: {image_path}")
            return []
        if not os.path.isfile(image_path):
            print(f"Warning: not a file: {image_path}")
            return []

        suffix = os.path.splitext(image_path)[1].lower()
        video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.mpeg', '.mpg', '.webm'}
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}

        res = 0
        face_bbox = None
        w, h = 0, 0

        if suffix in video_exts:
            # 视频分支
            try:
                res, face_bbox = faceCheck_getEAR(image_path)
            except Exception as e:
                print(f"Warning: video process error: {e}")
                res, face_bbox = 0, None
            # 获取视频帧尺寸（用于 fallback bbox）
            try:
                cap = cv2.VideoCapture(image_path)
                if cap.isOpened():
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
            except Exception as e:
                print(f"Warning: get video size error: {e}")
                w, h = 0, 0
        elif suffix in image_exts:
            # 图片分支
            img = cv2.imread(image_path)
            if img is None:
                # 仅图片文件读取失败时才打印“打开图片失败”
                #print(f"打开图片失败: cannot identify image file '{image_path}'")
                return []
            h, w = img.shape[:2]
            try:
                res, face_bbox = faceCheck_getEAR([img])
            except Exception as e:
                print(f"Warning: image process error: {e}")
                res, face_bbox = 0, None
        else:
            print(f"Warning: unknown file type '{suffix}', try to process as video: {image_path}")
            try:
                res, face_bbox = faceCheck_getEAR(image_path)
                # 尝试获取尺寸
                cap = cv2.VideoCapture(image_path)
                if cap.isOpened():
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
            except Exception as e:
                print(f"Warning: unknown file process error: {e}")
                res, face_bbox = 0, None

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