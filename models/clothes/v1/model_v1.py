from typing import List, Optional, Union
from ultralytics import YOLO
from models.base import BaseYoloModel


class clothesModelV1(BaseYoloModel):
    """专门识别汽车的 YOLOv8n 模型包装器。"""

    def __init__(self,
                 weights_path: str = "weights/clothes.pt",
                 conf: float = 0.4,
                 iou: float = 0.45,
                 imgsz: int = 640,
                 classes: Optional[Union[List[int], List[str]]] = None,
                 device: Optional[str] = None):
        """
        参数说明：
        - `weights_path`: 权重文件路径（实际上是 yolov8n，被改名为 car_v1.pt）
        - `conf`: 置信度阈值（默认 0.4 用于识别汽车）
        - `iou`: NMS IOU 阈值
        - `imgsz`: 输入图像大小（像素）
        - `classes`: 要识别的类别（None 时默认识别汽车 class_id=2）
        - `device`: 设备，例如 'cpu' 或 'cuda:0'
        """
        self.model = YOLO(weights_path)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        # 如果未指定 classes，默认只识别汽车（yolov8n coco 中汽车的 class_id=2）
        self.classes = classes if classes is not None else ["clothes"]
        if device:
            try:
                self.model.to(device)
            except Exception:
                pass

    def _resolve_class_ids(self) -> Optional[List[int]]:
        """如果 `self.classes` 为名称列表，转换为 id 列表。"""
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
        # 使用参数化的 predict 调用，专门识别指定的类别（默认为汽车）
        results = self.model.predict(source=image_path,
                                     conf=self.conf,
                                     iou=self.iou,
                                     imgsz=self.imgsz,
                                     classes=class_ids,
                                     max_det=300)
        result = results[0]
        return self.format_result(result.boxes)

