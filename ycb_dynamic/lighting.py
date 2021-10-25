import stillleben as sl
from pathlib import Path
import ycb_dynamic.CONSTANTS as CONSTANTS





def get_default_light_map():
    return sl.LightMap(CONSTANTS.ALL_LIGHTMAPS["default"])
