import random
import stillleben as sl
import sl_cutscenes.constants as CONSTANTS


def get_lightmap(map_name="random"):
    """
    Fetches the right lightmap given command line argument.
    """
    assert map_name in ["default", "random"] + list(CONSTANTS.ALL_LIGHTMAPS.keys()), f"Unknown lightmap {map_name}..."

    if map_name == "random":
        map_name = random.choice(list(CONSTANTS.ALL_LIGHTMAPS.keys()))
    elif map_name == "default":
        map_name = "Subway_Lights"
    lightmap = sl.LightMap(CONSTANTS.ALL_LIGHTMAPS[map_name])

    return lightmap
