from typing import List, Optional, Union
from ultralytics import YOLO
from models.base import BaseYoloModel


class BusModelV1(BaseYoloModel):
    def __init__(self,
                 weights_path: str = "weights/bus_v1.pt",
                 conf: float = 0.4,
                 iou: float = 0.45,
                 imgsz: int = 640,
                 classes: Optional[Union[List[int], List[str]]] = None,
                 device: Optional[str] = None):

        self.model = YOLO(weights_path)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        # 如果未指定 classes，默认只识别巴士（yolov8n coco 中巴士的 class_id=5）
        self.classes = classes if classes is not None else [5]
        if device:
            try:
                self.model.to(device)
            except Exception:
                pass

    def _resolve_class_ids(self) -> Optional[List[int]]:
        if not self.classes:
            return None
        if isinstance(self.classes[0], str):
            names = None
            if hasattr(self.model, 'names') and self.model.names:
                names = self.model.names
            else:
                try:
                    names = getattr(self.model.model, 'names', None)
                except Exception:
                    names = None
            if not names:
                return None
            name_to_id = {v: k for k, v in names.items()}
            ids = [name_to_id[c] for c in self.classes if c in name_to_id]
            return ids or None
        return list(self.classes)

    def predict(self, image_path: str):
        class_ids = self._resolve_class_ids()
        results = self.model.predict(source=image_path,
                                     conf=self.conf,
                                     iou=self.iou,
                                     imgsz=self.imgsz,
                                     classes=class_ids,
                                     max_det=300)
        result = results[0]
        return self.format_result(result.boxes)
