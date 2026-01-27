import os
import cv2
import numpy as np
import base64
import requests
import io
import contextlib
from ultralytics import YOLO
from models.base import BaseYoloModel

# --- 模型与路径配置 ---
# 使用仓库内的 weights 目录，便于部署和切换权重文件
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
WEIGHTS_DIR = os.path.join(ROOT_DIR, 'weights')
# 吊篮权重（weights/吊篮.pt），人员检测权重（weights/person_v1.pt）
def _find_weight(preferred_name_variants):
    """在 WEIGHTS_DIR 中查找首个匹配的权重文件名（大小写/部分匹配）。"""
    try:
        files = os.listdir(WEIGHTS_DIR)
    except Exception:
        return None

    for name in preferred_name_variants:
        for f in files:
            if f.lower() == name.lower():
                return os.path.join(WEIGHTS_DIR, f)

    # 尝试部分匹配
    for name in preferred_name_variants:
        for f in files:
            if name.lower() in f.lower():
                return os.path.join(WEIGHTS_DIR, f)

    # 未找到
    return None

# 首选文件名列表（覆盖本地非 ASCII 文件名问题）
basket_candidates = ['吊篮.pt', 'basket.pt', '吊篮.pt', 'best.pt']
person_candidates = ['person_v1.pt', 'person_v1.pt', 'person.pt']

basket_model_path = _find_weight(basket_candidates) or os.path.join(WEIGHTS_DIR, '吊篮.pt')
basis_person_model_path = _find_weight(person_candidates) or os.path.join(WEIGHTS_DIR, 'person_v1.pt')


# --- 多模态API配置 (仅用于无人吊篮复核) ---
MULTIMODAL_API_URL = "http://110.80.146.172:19082/v1/chat/completions"
MULTIMODAL_REVERIFY_PERSON_PROMPT = """请严格判断这张图像中是否包含任何真实的人类，特别注意吊篮内部区域。
只返回一个数字：包含真人返回 1，不包含返回 0。"""

# --- 设备与置信度配置 ---
DEVICE = 'cuda:2'
IOU = 0.25
basket_conf_threshold = 0.6
initial_person_conf_threshold = 0.25  # 只依靠YOLO，建议适当调高置信度

class ConstructionSiteAnalyzer:
    def __init__(self):
        self.basket_model = None
        self.basis_person_model = None
        self.person_count_total = 0

    def initialize_models(self):
        """初始化模型"""
        print("--- 正在加载模型 ---")
        try:
            # 抑制 Ultralytics 在加载模型时的控制台输出
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.basket_model = YOLO(basket_model_path)
                self.basis_person_model = YOLO(basis_person_model_path)
            print("--- 所有模型加载成功 ---")
            return True
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False

    def check_by_multimodal(self, image, prompt):
        """调用多模态API进行最后一次兜底复核"""
        try:
            _, buffer = cv2.imencode('.jpg', image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            payload = {
                "model": "Qwen25VL",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}]}
                ]
            }
            response = requests.post(MULTIMODAL_API_URL, json=payload, timeout=30)
            if response.status_code == 200:
                content = response.json().get('choices', [{}])[0].get('message', {}).get('content', '')
                return "1" in str(content)
            return False
        except:
            return False

    def detect_baskets(self, image):
        # 将 predict 的 stdout/stderr 重定向，避免打印推理进度行
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                return self.basket_model.predict(source=image, device=DEVICE, conf=basket_conf_threshold, verbose=False)
        except Exception:
            # 若预测异常，回退为直接调用（以便抛出更明确的错误）
            return self.basket_model.predict(source=image, device=DEVICE, conf=basket_conf_threshold, verbose=False)

    def process_basket_person(self, original_image, basket_coords):
        """
        使用YOLO在吊篮区域内识别人
        """
        x1, y1, x2, y2 = basket_coords
        basket_roi = original_image[y1:y2, x1:x2]
        
        # 在裁切后的区域进行检测，通常比全图检测小目标更准
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                results = self.basis_person_model.predict(
                    source=basket_roi,
                    iou=IOU,
                    classes=[0],
                    device=DEVICE,
                    conf=initial_person_conf_threshold,
                    verbose=False
                )
        except Exception:
            results = self.basis_person_model.predict(
                source=basket_roi,
                iou=IOU,
                classes=[0],
                device=DEVICE,
                conf=initial_person_conf_threshold,
                verbose=False
            )

        detected_persons = []
        for box in results[0].boxes:
            px1, py1, px2, py2 = map(int, box.xyxy[0])
            # 将坐标映射回原图
            abs_px1, abs_py1 = x1 + px1, y1 + py1
            abs_px2, abs_py2 = x1 + px2, y1 + py2
            detected_persons.append((abs_px1, abs_py1, abs_px2, abs_py2))
            
            # 绘制检测框
            cv2.rectangle(original_image, (abs_px1, abs_py1), (abs_px2, abs_py2), (0, 255, 0), 2)
            cv2.putText(original_image, "Person", (abs_px1, abs_py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return len(detected_persons)

    def run_pipeline(self, source_image_folder: str):
        """批处理接口（离线使用）：接受图片文件夹路径，返回每张图的违规检测汇总。

        注意：该方法不再保存任何图像到磁盘；若需要保存，请在调用端处理返回的结果并进行保存。
        """
        print("--- 启动工地吊篮分析管线（接口化，无磁盘写入）---")

        if not self.initialize_models():
            return []

        image_files = [f for f in os.listdir(source_image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        summary = []

        for filename in image_files:
            source_path = os.path.join(source_image_folder, filename)
            original_image = cv2.imread(source_path)
            if original_image is None:
                print(f"无法读取图片: {filename}")
                continue

            print(f"分析图片: {filename}")
            basket_results = self.detect_baskets(original_image)

            image_violations = []

            if len(basket_results[0].boxes) == 0:
                print(f"   > 未检测到吊篮: {filename}")
            else:
                for idx, basket_box in enumerate(basket_results[0].boxes, start=1):
                    bx1, by1, bx2, by2 = map(int, basket_box.xyxy[0])
                    count = self.process_basket_person(original_image, (bx1, by1, bx2, by2))

                    if count == 0:
                        basket_roi = original_image[by1:by2, bx1:bx2]
                        if self.check_by_multimodal(basket_roi, MULTIMODAL_REVERIFY_PERSON_PROMPT):
                            print(f"   > 吊篮{idx}: YOLO未发现但多模态复核有人")
                            count = 1

                    if count != 2:
                        image_violations.append({
                            "basket_idx": idx,
                            "count": count,
                            "bbox": [float(bx1), float(by1), float(bx2), float(by2)]
                        })

            summary.append({"filename": filename, "violations": image_violations})

        print("--- 批量分析完成 ---")
        return summary


class BasketModelV1(BaseYoloModel):
    """兼容其他模型的包装类：提供 `predict(image_path)` 接口，仅返回吊篮检测框列表。

    说明：如果需要运行完整的工地分析流水线，可直接使用 `ConstructionSiteAnalyzer`。
    """
    def __init__(self, weights_path: str = None, conf: float = None, iou: float = None, imgsz: int = 640, device: str = None):
        self.weights_path = weights_path or basket_model_path
        self.conf = conf if conf is not None else basket_conf_threshold
        self.iou = iou if iou is not None else IOU
        self.imgsz = imgsz
        self.device = device if device is not None else DEVICE

        # 延迟加载模型，避免导入时占用显存
        self.model = None

    def _ensure_model(self):
        if self.model is None:
            try:
                self.model = YOLO(self.weights_path)
                if self.device:
                    try:
                        self.model.to(self.device)
                    except Exception:
                        pass
            except Exception as e:
                print(f"加载吊篮模型失败: {e}")
                raise

    def _ensure_person_model(self):
        if not hasattr(self, 'person_model') or self.person_model is None:
            try:
                self.person_model = YOLO(basis_person_model_path)
                if self.device:
                    try:
                        self.person_model.to(self.device)
                    except Exception:
                        pass
            except Exception as e:
                print(f"加载人员模型失败: {e}")
                self.person_model = None

    def check_by_multimodal(self, image, prompt: str, timeout: int = 30) -> bool:
        """简单的多模态判断器（复用 ConstructionSiteAnalyzer 的逻辑）"""
        try:
            _, buffer = cv2.imencode('.jpg', image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            payload = {
                "model": "Qwen25VL",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}]}
                ]
            }
            response = requests.post(MULTIMODAL_API_URL, json=payload, timeout=timeout)
            if response.status_code == 200:
                content = response.json().get('choices', [{}])[0].get('message', {}).get('content', '')
                return "1" in str(content)
            return False
        except Exception:
            return False

    def predict(self, image_path: str):
        """按照统一接口：输入图片路径，返回标准 detections 列表。"""
        # 确保加载模型
        self._ensure_model()
        self._ensure_person_model()

        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图片: {image_path}")
            return []

        detections = []

        try:
            # 1) 检测吊篮
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    results = self.model.predict(source=img, conf=self.conf, iou=self.iou, imgsz=self.imgsz, verbose=False)
            except Exception:
                results = self.model.predict(source=img, conf=self.conf, iou=self.iou, imgsz=self.imgsz, verbose=False)
            boxes = results[0].boxes

            for box in boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]

                # 在吊篮区域内检测人员
                crop = img[y1:y2, x1:x2]
                person_count = 0
                if hasattr(self, 'person_model') and self.person_model is not None and crop.size != 0:
                    try:
                        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                            pres = self.person_model.predict(source=crop, iou=self.iou, classes=[0], device=self.device, conf=initial_person_conf_threshold, verbose=False)
                    except Exception:
                        pres = self.person_model.predict(source=crop, iou=self.iou, classes=[0], device=self.device, conf=initial_person_conf_threshold, verbose=False)
                    person_count = len(pres[0].boxes)

                # 如果 YOLO 未检测到人，则用多模态复核
                if person_count == 0:
                    if crop.size != 0 and self.check_by_multimodal(crop, MULTIMODAL_REVERIFY_PERSON_PROMPT):
                        person_count = 1

                # 违规判断：人数不等于2 -> 标记为 violation（class_id=1）
                if person_count != 2:
                    detections.append({
                        "class_id": 1,
                        "confidence": 1.0,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)]
                    })

            return detections
        except Exception as e:
            print(f"BasketModelV1.predict 异常: {e}")
            return []

if __name__ == '__main__':
    analyzer = ConstructionSiteAnalyzer()
    analyzer.run_pipeline()