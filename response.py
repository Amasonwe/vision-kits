def build_response(category: str, version: str, detections: list):
    return {
        "category": category,
        "model_version": version,
        "total": len(detections),
        "detections": detections
    }
