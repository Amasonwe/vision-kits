import os
import uuid
import math
import json
from typing import List, Dict
from PIL import Image, ImageDraw, ImageFont

TMP_DIR = "tmp"

os.makedirs(TMP_DIR, exist_ok=True)

def save_image(file_bytes: bytes) -> str:
    filename = f"{uuid.uuid4().hex}.jpg"
    path = os.path.join(TMP_DIR, filename)
    with open(path, "wb") as f:
        f.write(file_bytes)
    return path


def get_bounding_boxes(detections: List[Dict]) -> List[Dict]:
    """返回标准的边界框列表，每项包含 class_id, confidence, bbox.

    参数 detections: 模型返回的检测列表（每项为 dict，包含 "bbox" 等字段）
    """
    if not detections:
        return []
    boxes = []
    for d in detections:
        boxes.append({
            "class_id": int(d.get("class_id", -1)),
            "confidence": float(d.get("confidence", 0.0)),
            "bbox": [float(x) for x in d.get("bbox", [])]
        })
    return boxes


def annotate_image(image_path: str, detections: List[Dict], out_dir: str = "static/annotated") -> str:
    """在图片上绘制检测到的框并保存到 `out_dir`，返回相对于项目根的静态路径（可通过 /static/ 访问）。

    detections: 每项包含 "bbox", "class_id", "confidence"
    """
    os.makedirs(out_dir, exist_ok=True)
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        # 打印异常信息便于排查，同时返回空字符串
        print(f"打开图片失败: {e}")
        return ""

    draw = ImageDraw.Draw(img)
    # 优化字体加载逻辑，避免因字体缺失报错
    try:
        font = ImageFont.load_default(size=16)  # 指定默认字体大小
    except Exception:
        font = None  # 极端情况设为None，Pillow会用内置默认字体

    for d in detections:
        bbox = d.get("bbox", [])
        if not bbox or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = [math.floor(float(v)) for v in bbox]
        label = f"id:{d.get('class_id', '')} {d.get('confidence', 0):.2f}"
        
        # 绘制检测矩形框（原逻辑保留）
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        
        # 修复核心：替换 textsize 为 textbbox（适配Pillow 9.1.0+）
        if font:
            # textbbox返回 (left, top, right, bottom)，计算文字宽高
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            # 无字体时使用固定尺寸，避免报错
            text_width = len(label) * 8
            text_height = 16
        
        # 绘制标签背景（避免超出图片顶部）
        text_bg_y1 = max(0, y1 - text_height - 2)
        text_bg = [x1, text_bg_y1, x1 + text_width + 4, y1]
        draw.rectangle(text_bg, fill=(255, 0, 0))
        
        # 绘制标签文字（处理font为None的情况）
        draw.text((x1 + 2, text_bg_y1 + 1), label, fill=(255, 255, 255), font=font)

    # 保存标注后的图片
    out_name = f"annotated_{uuid.uuid4().hex}.jpg"
    out_path = os.path.join(out_dir, out_name)
    try:
        img.save(out_path)
    except Exception as e:
        print(f"保存标注图片失败: {e}")
        return ""

    # 返回可以通过 /static/ 访问的相对路径
    return f"/static/annotated/{out_name}"