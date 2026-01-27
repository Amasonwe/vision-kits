def build_response(category: str, version: str, detections: list, annotated_image: str = None, record_id: int = None, warnings: list = None):
    resp = {
        "category": category,
        "model_version": version,
        "total": len(detections),
        "detections": detections
    }
    if annotated_image:
        resp["annotated_image"] = annotated_image
    if record_id is not None:
        resp["record_id"] = record_id
    if warnings:
        resp["warnings"] = warnings
    return resp
