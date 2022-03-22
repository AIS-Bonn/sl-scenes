"""
Constant values such as:
    - Paths and directories
    - Objects meshes and proprerties.
    - Collections of objects for each scene

Everything non-code that can eventually be changed.
"""
import os
from pathlib import Path

import random
import torch
from sl_cutscenes.OBJECT_INFO import OBJECT_INFO

PI = torch.acos(torch.tensor(-1))

# Paths and Directories
PKG_ROOT_PATH = Path(__file__).parent.parent
PKG_SRC_PATH = Path(__file__).parent
EXT_DATA_PATH = PKG_SRC_PATH / "assets" / "external_data"
MESH_BASE_DIR = EXT_DATA_PATH / "object_models"
TEXT_BASE_DIR = EXT_DATA_PATH / "textures"
IBL_BASE_PATH = EXT_DATA_PATH / "light_maps"
ALL_LIGHTMAPS = {
    "Alexs_Apartment": IBL_BASE_PATH / "Alexs_Apartment" / "Alexs_Apartment.ibl",
    "Circus_Backstage": IBL_BASE_PATH / "Circus_Backstage" / "Circus_Backstage.ibl",
    "Milkyway": IBL_BASE_PATH / "Milkyway" / "Milkyway.ibl",
    "Popcorn_Lobby": IBL_BASE_PATH / "Popcorn_Lobby" / "Popcorn_Lobby.ibl",
    "Siggraph2007_UpperFloor": IBL_BASE_PATH / "Siggraph2007_UpperFloor" / "Siggraph2007_UpperFloor.ibl",
    "Subway_Lights": IBL_BASE_PATH / "Subway_Lights" / "Subway_Lights.ibl",
    "Theatre_Seating": IBL_BASE_PATH / "Theatre_Seating" / "Theatre_Seating.ibl",
    "Ueno-Shrine": IBL_BASE_PATH / "Ueno-Shrine" / "Ueno-Shrine.ibl",
}

#########################
# Scene global configurations
#########################

# ROBOTS
END_EFFECTOR_POSE = torch.tensor([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0.0], [0, 0, 0, 1]])

# TABLE
TABLE_POSE = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.0], [0, 0, 0, 1]])

# BOWL
WOODEN_BOWL_POSE = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.], [0, 0, 0, 1]])
BOWL_FRUIT_INIT_POS = torch.tensor([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0.], [0, 0, 0, 1]])
RED_BOWL_POSE = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.], [0, 0, 0, 1]])

# BALL BOX
BALL_BOX_POSE = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.], [0, 0, 0, 1]])
BALL_BOX_BALL_INIT_POSE = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.], [0, 0, 0, 1]])

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

# BILLARDS
BILLARDS_TRIANGLE_POSES = [
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

# Robots etc.
SUCTION_GRIPPER = [obj for obj in OBJECT_INFO if obj.name == "suction_gripper"]

# YCB Objects
FLAT_OBJS = [str(i).zfill(3) for i in range(0, 11)]
FRUIT_OBJS = [str(i).zfill(3) for i in range(11, 19)]
CLEANING_OBJS = [str(i).zfill(3) for i in range(19, 22)]
KITCHEN_OBJS = [str(i).zfill(3) for i in range(22, 35)]
DICE_OBJS = [str(i).zfill(3) for i in [5, 7, 8, 9]]
YCB_SMALL_BALL_OBJS = [str(i).zfill(3) for i in [55, 56, 57, 58]]
YCBV_OBJS = [str(i).zfill(3) for i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 19, 21, 24, 25, 35, 36, 37, 40, 51, 52, 61]]

# Meta-Subsets
YCB_OBJECTS = [obj for obj in OBJECT_INFO if obj.name[0].isdigit()]
OTHER_OBJECTS = [obj for obj in OBJECT_INFO if not obj.name[0].isdigit()]
BOWLING_BALL = [obj for obj in OBJECT_INFO if obj.name == "bowling_ball"]
WOOD_BLOCK = [obj for obj in OBJECT_INFO if obj.name == "036_wood_block"]
WOODEN_BOX = [obj for obj in OBJECT_INFO if obj.name == "wooden_box"]
CAMERA_OBJ = [obj for obj in OBJECT_INFO if obj.name == "camera_object"]
DUMMY_CAMERA_OBJ = [obj for obj in OBJECT_INFO if obj.name == "dummy_camera_object"]

# Decoration Objects
CHAIRS = [obj for obj in OBJECT_INFO if obj.name.endswith("_chair")]
CUPBOARDS = [obj for obj in OBJECT_INFO if obj.name.endswith("_cupboard")]
TABLES = [obj for obj in OBJECT_INFO if obj.name.endswith("_table")]
NO_POOL_TABLES = [obj for obj in OBJECT_INFO if obj.name.endswith("_table") and not obj.name.endswith("pool_table")]
BOWLS = [obj for obj in OBJECT_INFO if obj.name.endswith("_bowl") and obj.name != "024_bowl"]
BALL_BOXES = [obj for obj in OBJECT_INFO if obj.name in ["laundry_basket"]]  #, "wooden_box"]]
FURNITURES = [obj for obj in OBJECT_INFO if "furniture/" in obj.mesh_fp and
                                            "table" not in obj.name and
                                            "chair" not in obj.name]
# FURNITURES = [obj for obj in OBJECT_INFO if "furniture/" in obj.mesh_fp and obj.name == "kitchen_wood_m"]

# Surfaces and Rooms
FLOORS = [obj for obj in OBJECT_INFO if obj.name.endswith("_floor")]
# FLOORS = [obj for obj in OBJECT_INFO if obj.name.endswith("_floor") and "carpet" in obj.name]
FLOOR_NAMES = [os.path.basename(obj.mesh_fp) for obj in OBJECT_INFO if obj.name.endswith("_floor")]
WALLS = [obj for obj in OBJECT_INFO if obj.name.endswith("_wall")]
# WALLS = [obj for obj in OBJECT_INFO if obj.name.endswith("_wall") and "black_tiling" in obj.name]
ROOMS = [obj for obj in OBJECT_INFO if "complete_rooms" in obj.mesh_fp]

# for accessing
TABLE = [random.choice(TABLES)]
NO_POOL_TABLE = [random.choice(NO_POOL_TABLES)]
BOWL = [random.choice(BOWLS)]
BALL_BOX = [random.choice(BALL_BOXES)]
ROOM = [random.choice(ROOMS)]
FLOOR = [random.choice(FLOORS)]
WALL = [random.choice(WALLS)]
FURNITURE = [random.choice(FURNITURES)]

# other collections
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
YCB_SMALL_BALLS = [  # Small YCB balls
    obj for obj in OBJECT_INFO if obj.name.split("_")[0] in YCB_SMALL_BALL_OBJS
]
