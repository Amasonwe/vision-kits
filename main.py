from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from utils.image import save_image
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

    return build_response(category, version, detections)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=18001)
    # uvicorn.run("renewal.main:app", host="0.0.0.0", port=8002, reload=True)