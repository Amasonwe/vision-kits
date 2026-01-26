import os
import uuid

TMP_DIR = "tmp"

os.makedirs(TMP_DIR, exist_ok=True)

def save_image(file_bytes: bytes) -> str:
    filename = f"{uuid.uuid4().hex}.jpg"
    path = os.path.join(TMP_DIR, filename)
    with open(path, "wb") as f:
        f.write(file_bytes)
    return path
