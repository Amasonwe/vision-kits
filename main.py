from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from utils.image import save_image, get_bounding_boxes, annotate_image
from utils.db import save_detection, get_detection
from router import get_model
from response import build_response

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
    """
    识别图片中的目标对象

    Args:
        category: 算法类别
        version: 版本号
        file: 待识别的图片
    """
    image_bytes = await file.read()
    image_path = save_image(image_bytes)

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