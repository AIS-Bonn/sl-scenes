"""
Constant values such as:
    - Paths and directories
    - Objects meshes and proprerties.
    - Collections of objects for each scene

Everything non-code that can eventually be changed.
"""

from pathlib import Path
import torch
from ycb_dynamic.OBJECT_INFO import OBJECT_INFO


# Paths and Directories
MESH_BASE_DIR = Path(".") / "external_data" / "object_models"

# Lighting
IBL_BASE_PATH = Path(".") / "external_data" / "light_maps"
ALL_LIGHTMAPS = {
    "default": IBL_BASE_PATH / "Chiricahua_Plaza" / "Chiricahua_Plaza.ibl",  # DEPRECATED
    "Alexs_Apartment": IBL_BASE_PATH / "Alexs_Apartment" / "Alexs_Apartment.ibl",
    "Barcelona_Rooftops": IBL_BASE_PATH / "Barcelona_Rooftops" / "Barcelona_Rooftops.ibl",
    "Chiricahua_Plaza": IBL_BASE_PATH / "Chiricahua_Plaza" / "Chiricahua_Plaza.ibl",
    "Circus_Backstage": IBL_BASE_PATH / "Circus_Backstage" / "Circus_Backstage.ibl",
    "Helipad_GoldenHour": IBL_BASE_PATH / "Helipad_GoldenHour" / "Helipad_GoldenHour.ibl",
    # "Old_Industrial_Hall": IBL_BASE_PATH / "Old_Industrial_Hall" / "Old_Industrial_Hall.ibl",  # Path issue?
    "Popcorn_Lobby": IBL_BASE_PATH / "Popcorn_Lobby" / "Popcorn_Lobby.ibl",
    "Wooden_Door": IBL_BASE_PATH / "Wooden_Door" / "Wooden_Door.ibl",
}

#########################
# Scene global configurations
#########################

# TABLE
TABLE_POSE = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.0], [0, 0, 0, 1]])

# BOWL
WOODEN_BOWL_POSE = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.], [0, 0, 0, 1]])
BOWL_FRUIT_INIT_POS = torch.tensor([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0.], [0, 0, 0, 1]])
RED_BOWL_POSE = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.], [0, 0, 0, 1]])

# BOWLING
WOOD_BLOCK_POSES = [
    torch.tensor([[1, 0, 0, 0.400], [0, 1, 0, -0.20], [0,  0, 1, 0.0], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.400], [0, 1, 0,     0], [0,  0, 1, 0.0], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.400], [0, 1, 0,  0.20], [0,  0, 1, 0.0], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.398], [0, 0, 1, -0.11], [0, -1, 0, 0.15], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.398], [0, 0, 1,  0.11], [0, -1, 0, 0.15], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.396], [0, 1, 0,  -0.1], [0,  0, 1, 0.3], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.396], [0, 1, 0,   0.1], [0,  0, 1, 0.3], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.395], [0, 0, 1,     0], [0, -1, 0, 0.45], [0, 0, 0, 1]]),
]

# BILLIARDS
BILLIARDS_TRIANLGE_POSES = [
    torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.0], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.10], [0, 1, 0, -0.06], [0, 0, 1, 0.0], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.10], [0, 1, 0, 0.06], [0, 0, 1, 0.0], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.20], [0, 1, 0, 0], [0, 0, 1, 0.0], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.20], [0, 1, 0, 0.12], [0, 0, 1, 0.0], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.20], [0, 1, 0, -0.12], [0, 0, 1, 0.0], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.30], [0, 1, 0, -0.06], [0, 0, 1, 0.0], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.30], [0, 1, 0, 0.06], [0, 0, 1, 0.0], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.30], [0, 1, 0, -0.18], [0, 0, 1, 0.0], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.30], [0, 1, 0, 0.18], [0, 0, 1, 0.0], [0, 0, 0, 1]]),
]

# STACK PYRAMID (4-1-1)
STACK_PYRAMID_POSES = [
    torch.tensor([[1, 0, 0, -0.05], [0, 1, 0, -0.05], [0, 0, 1, 0.0], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, -0.05], [0, 1, 0, 0.05], [0, 0, 1, 0.0], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.05], [0, 1, 0, -0.05], [0, 0, 1, 0.0], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.05], [0, 1, 0, 0.05], [0, 0, 1, 0.0], [0, 0, 0, 1]]),
    #
    torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.15], [0, 0, 0, 1]]),
    #
    torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.30], [0, 0, 0, 1]]),
]


#########################
# Pre-defined object sets
#########################

FLAT_OBJS = [str(i).zfill(3) for i in range(0, 11)]
FRUIT_OBJS = [str(i).zfill(3) for i in range(11, 19)]
CLEANING_OBJS = [str(i).zfill(3) for i in range(19, 22)]
KITCHEN_OBJS = [str(i).zfill(3) for i in range(22, 35)]
DICE_OBJS = [str(i).zfill(3) for i in [5, 7, 8, 9]]
YCBV_OBJS = [str(i).zfill(3) for i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 19, 21, 24, 25, 35, 36, 37, 40, 51, 52, 61]]

YCB_OBJECTS = [obj for obj in OBJECT_INFO if obj.name[0].isdigit()]
OTHER_OBJECTS = [obj for obj in OBJECT_INFO if not obj.name[0].isdigit()]
TABLE = [obj for obj in OBJECT_INFO if obj.name == "art_deco_table"]
# TABLE = [obj for obj in OBJECT_INFO if obj.name == "wooden_table"]
# TABLE = [obj for obj in OBJECT_INFO if obj.name == "folding_table"]
# TABLE = [obj for obj in OBJECT_INFO if obj.name == "pool_table"]
TABLES = [obj for obj in OBJECT_INFO if obj.name.endswith("_table")]
CHAIRS = [obj for obj in OBJECT_INFO if obj.name.endswith("_chair")]
CUPBOARDS = [obj for obj in OBJECT_INFO if obj.name.endswith("_cupboard")]
BOWLS = [obj for obj in OBJECT_INFO if obj.name.endswith("_bowl")]
BOWLING_BALL = [obj for obj in OBJECT_INFO if obj.name == "bowling_ball"]
WOOD_BLOCK = [obj for obj in OBJECT_INFO if obj.name == "036_wood_block"]
YCBV_OBJECTS = [  # Only considering the YCB-Video subset of objects
    obj for obj in OBJECT_INFO if obj.name.split("_")[0] in YCBV_OBJS
]
STACK_OBJECTS = [  # Only considering a subset of 'stackable' objects
    obj for obj in OBJECT_INFO if obj.name.split("_")[0] in FLAT_OBJS
]
BILLIARDS_OBJECTS = [  # Considering objects that do not roll, e.g. only objects with flat surfaces
    obj for obj in OBJECT_INFO if obj.name.split("_")[0] in FLAT_OBJS
]
DICE_OBJECTS = [  # Considering objects that do roll, e.g. small and regular shapes
    obj for obj in OBJECT_INFO if obj.name.split("_")[0] in DICE_OBJS
]
FRUIT_OBJECTS = [  # Considering fruits
    obj for obj in OBJECT_INFO if obj.name.split("_")[0] in FRUIT_OBJS
]
BOWL = [obj for obj in OBJECT_INFO if obj.name == "red_bowl"]
WOODEN_BOWL = [obj for obj in OBJECT_INFO if obj.name == "wooden_bowl"]
