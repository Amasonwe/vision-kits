from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from utils.image import save_image, get_bounding_boxes, annotate_image
from utils.db import save_detection, get_detection
from router import get_model, MODEL_REGISTRY
from response import build_response
import logging
try:
    # 抑制 ultralytics 控制台推理进度/速度输出
    logging.getLogger('ultralytics').setLevel(logging.ERROR)
    logging.getLogger('ultralytics.yolo').setLevel(logging.ERROR)
    # also try to set internal LOGGER if available
    try:
        from ultralytics.yolo.utils import LOGGER as _UL_LOGGER
        _UL_LOGGER.setLevel('ERROR')
    except Exception:
        pass
except Exception:
    pass

app = FastAPI(title="视觉算法合集",
    version="1.0.0",
    docs_url=None,
    redoc_url=None)

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/docs")
async def custom_docs():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>(FFCS)智慧城市事业部-视觉算法合集</title>
        <link rel="stylesheet" href="/static/swagger-ui.css">
    </head>
    <body>
        <div id="swagger-ui"></div>
        <script src="/static/swagger-ui-bundle.js"></script>
        <script>
            window.onload = function() {
                SwaggerUIBundle({
                    url: '/openapi.json',
                    dom_id: '#swagger-ui',
                    presets: [
                        SwaggerUIBundle.presets.apis,
                        SwaggerUIBundle.presets.standalone
                    ],
                    layout: "BaseLayout"  // 改为 BaseLayout
                });
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.post("/detect", summary="识别图片中的目标对象")
async def detect(
    category: str = Form(...),
    version: str = Form(...),
    file: UploadFile = File(...)
):
        # 新增调试打印
    print(f"📥 接口传入参数：category={category}, version={version}")
    print(f"📋 当前 MODEL_REGISTRY 类别：{list(MODEL_REGISTRY.keys())}")
    if category in MODEL_REGISTRY:
        print(f"📋 {category} 支持的版本：{list(MODEL_REGISTRY[category].keys())}")
    
    """
    识别图片中的目标对象

    Args:
        category: 算法类别
        version: 版本号
        file: 待识别的图片
    """
    image_bytes = await file.read()
    # preserve original filename extension so videos keep their extension
    image_path = save_image(image_bytes, original_filename=file.filename)

    model = get_model(category, version)
    detections = model.predict(image_path)

    # 返回标准 bbox 列表
    boxes = get_bounding_boxes(detections)

    # 生成可通过 /static/ 访问的标注图片
    annotated_url = annotate_image(image_path, boxes)

    # 保存到数据库，返回记录 id
    try:
        record_id = save_detection(category, version, image_path, annotated_url, detections)
    except Exception:
        record_id = None

    return build_response(category, version, detections, annotated_image=annotated_url, record_id=record_id)


@app.get("/detections/{record_id}", summary="查询单条检测记录")
async def get_detection_record(record_id: int):
    rec = get_detection(record_id)
    if not rec:
        return {"error": "record not found"}
    return rec

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=18001)
    # uvicorn.run("renewal.main:app", host="0.0.0.0", port=8002, reload=True)