from models.person.v1.model_v1 import PersonModelV1
from models.car.v1.model_v1 import CarModelV1
from models.ebike.v1.model_v1 import EBikeModelV1
from models.bus.v1.model_v1 import BusModelV1
from models.calling.v1.model_v1 import phoneModelV1
from models.clothes.v1.model_v1 import clothesModelV1
from models.helmet.v1.model_v1 import helmetModelV1
from models.smoking.v1.model_v1 import smokingModelV1
from models.smoke.v1.model_v1 import smokeModelV1
from models.fire.v1.model_v1 import fireModelV1
from models.asleep.v1.model_v1 import AsleepModelV1

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
    },
    "calling": {
        "v1": phoneModelV1
    },
    "clothes": {
        "v1": clothesModelV1
    },
    "helmet": {
        "v1": helmetModelV1
    },
    "asleep": {
        "v1": AsleepModelV1
    },
    "smoking": {
        "v1": smokingModelV1
    },
    "fire": {
        "v1": fireModelV1
    },
    "smoke": {
        "v1": smokeModelV1
    },
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
