from models.person.v1.model_v1 import PersonModelV1
from models.car.v1.model_v1 import CarModelV1
from models.ebike.v1.model_v1 import EBikeModelV1
from models.bus.v1.model_v1 import BusModelV1

MODEL_REGISTRY = {
    "person": {
        "v1": PersonModelV1
    },
    "car": {
        "v1": CarModelV1
    },
    "ebike": {
        "v1": EBikeModelV1
    },
    "bus": {
        "v1": BusModelV1
    }
}

_model_cache = {}   

def get_model(category: str, version: str):
    key = f"{category}_{version}"

    if key in _model_cache:
        return _model_cache[key]

    try:
        model_class = MODEL_REGISTRY[category][version]
    except KeyError:
        raise ValueError("模型类别或版本不存在")

    model = model_class()
    _model_cache[key] = model
    return model
