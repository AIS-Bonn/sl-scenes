import stillleben as sl
from pathlib import Path


IBL_BASE_PATH = Path(".") / "external_data" / "light_maps"
ALL_LIGHTMAPS = {
    "default": IBL_BASE_PATH / "Chiricahua_Plaza" / "Chiricahua_Plaza.ibl",
}


def get_default_light_map():
    return sl.LightMap(ALL_LIGHTMAPS["default"])
