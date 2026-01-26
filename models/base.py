from abc import ABC, abstractmethod

class BaseYoloModel(ABC):

    @abstractmethod
    def predict(self, image_path: str):
        pass

    def format_result(self, boxes):
        results = []

        if boxes is None:
            return results

        for box in boxes:
            results.append({
                "class_id": int(box.cls),
                "confidence": float(box.conf),
                "bbox": [
                    float(box.xyxy[0][0]),
                    float(box.xyxy[0][1]),
                    float(box.xyxy[0][2]),
                    float(box.xyxy[0][3])
                ]
            })

        return results
