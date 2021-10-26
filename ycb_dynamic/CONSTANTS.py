"""
Constant values such as:
    - Paths and directories
    - Objects meshes and proprerties.
    - Collections of objects for each scene

Everything non-code that can eventually be changed.
"""

from pathlib import Path
from collections import namedtuple
import torch


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


# Scene global configureations
CAM_POS = torch.Tensor([0.5, 2.4, 1.5])
CAM_LOOKAT = torch.Tensor([0, 0, 1.2])

# TABLE
TABLE_POSE = torch.tensor([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0.60], [0, 0, 0, 1]])
DROP_LIMITS = {
    "x_min": -0.6,
    "x_max": 0.6,
    "y_min": -0.4,
    "y_max": 0.4,
    "z_min": 1.5,
    "z_max": 1.8,
}

# BOWL
BOWL_POSE = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

# BOWLING
BOWLING_CAM_POS = torch.Tensor([-0.5, 2.4, 1.5])
BOWLING_INITIAL_VELOCITY = torch.tensor([2.5, 0, 0])
WOOD_BLOCK_POSES = [
    torch.tensor([[1, 0, 0, 0.400], [0, 1, 0, -0.20], [0,  0, 1, 1.23], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.400], [0, 1, 0,     0], [0,  0, 1, 1.23], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.400], [0, 1, 0,  0.20], [0,  0, 1, 1.23], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.398], [0, 0, 1, -0.11], [0, -1, 0, 1.38], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.398], [0, 0, 1,  0.11], [0, -1, 0, 1.38], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.396], [0, 1, 0,  -0.1], [0,  0, 1, 1.53], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.396], [0, 1, 0,   0.1], [0,  0, 1, 1.53], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.395], [0, 0, 1,     0], [0, -1, 0, 1.68], [0, 0, 0, 1]]),
    #torch.tensor([[1, 0, 0, 0.395], [0, 1, 0,     0], [0,  0, 1, 1.83], [0, 0, 0, 1]]),
]

# BILLIARDS
BILLIARDS_TRIANLGE_POSES = [
    torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1.23], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.10], [0, 1, 0, -0.06], [0, 0, 1, 1.23], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.10], [0, 1, 0, 0.06], [0, 0, 1, 1.23], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.20], [0, 1, 0, 0], [0, 0, 1, 1.23], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.20], [0, 1, 0, 0.12], [0, 0, 1, 1.23], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.20], [0, 1, 0, -0.12], [0, 0, 1, 1.23], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.30], [0, 1, 0, -0.06], [0, 0, 1, 1.23], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.30], [0, 1, 0, 0.06], [0, 0, 1, 1.23], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.30], [0, 1, 0, -0.18], [0, 0, 1, 1.23], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.30], [0, 1, 0, 0.18], [0, 0, 1, 1.23], [0, 0, 0, 1]]),
]


# STACK PYRAMID (4-1-1)
STACK_PYRAMID_POSES = [
    torch.tensor([[1, 0, 0, -0.05], [0, 1, 0, -0.05], [0, 0, 1, 1.23], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, -0.05], [0, 1, 0, 0.05], [0, 0, 1, 1.23], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.05], [0, 1, 0, -0.05], [0, 0, 1, 1.23], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.05], [0, 1, 0, 0.05], [0, 0, 1, 1.23], [0, 0, 0, 1]]),
    #
    torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1.4], [0, 0, 0, 1]]),
    #
    torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1.55], [0, 0, 0, 1]]),
]

# BOWL
RED_BOWL_POSE = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1.28], [0, 0, 0, 1]])
WOODEN_BOWL_POSE = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1.1], [0, 0, 0, 1]])
BOWL_CAM_POS = torch.Tensor([-0.3, 1.2, 2.1])
BOWL_CAM_LOOKAT = torch.Tensor([0, 0, 1.25])
BOWL_FRUIT_INIT_POS = torch.tensor([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 1.4], [0, 0, 0, 1]])


# Object Information
FLAG_CONCAVE = 1 << 0
ObjectInfo = namedtuple(
    "ObjectInfo",
    ["name", "mesh_fp", "weight", "flags", "metallic", "roughness", "scale"],
)

# source: http://www.ycbbenchmarks.com/wp-content/uploads/2015/09/object-list-Sheet1.pdf
# BOP models use millimeter units for some strange reason. sl is completely metric, so scale accordingly.
OBJECT_INFO = [
    ObjectInfo(
        "002_master_chef_can",
        "ycbv_models/models_fine/obj_000001.ply",
        0.414,
        0,
        0.6,
        0.2,
        0.001,
    ),
    ObjectInfo(
        "003_cracker_box",
        "ycbv_models/models_fine/obj_000002.ply",
        0.411,
        0,
        0.1,
        0.5,
        0.001,
    ),
    ObjectInfo(
        "004_sugar_box",
        "ycbv_models/models_fine/obj_000003.ply",
        0.514,
        0,
        0.1,
        0.5,
        0.001,
    ),
    ObjectInfo(
        "005_tomato_soup_can",
        "ycbv_models/models_fine/obj_000004.ply",
        0.349,
        0,
        0.1,
        0.5,
        0.001,
    ),
    ObjectInfo(
        "006_mustard_bottle",
        "ycbv_models/models_fine/obj_000005.ply",
        0.603,
        0,
        0.3,
        0.5,
        0.001,
    ),
    ObjectInfo(
        "007_tuna_fish_can",
        "ycbv_models/models_fine/obj_000006.ply",
        0.171,
        0,
        0.6,
        0.2,
        0.001,
    ),
    ObjectInfo(
        "008_pudding_box",
        "ycbv_models/models_fine/obj_000007.ply",
        0.187,
        0,
        0.1,
        0.5,
        0.001,
    ),
    ObjectInfo(
        "009_gelatin_box",
        "ycbv_models/models_fine/obj_000008.ply",
        0.097,
        0,
        0.1,
        0.5,
        0.001,
    ),
    ObjectInfo(
        "010_potted_meat_can",
        "ycbv_models/models_fine/obj_000009.ply",
        0.370,
        0,
        0.6,
        0.3,
        0.001,
    ),
    ObjectInfo(
        "011_banana",
        "ycbv_models/models_fine/obj_000010.ply",
        0.066,
        0,
        0.3,
        0.3,
        0.001,
    ),
    ObjectInfo(
        "012_strawberry",
        "ycb_models/012_strawberry/google_512k/textured.obj",
        0.066,
        0,
        0.3,
        0.3,
        1.000,
    ),
    ObjectInfo(
        "013_apple",
        "ycb_models/013_apple/google_512k/textured.obj",
        0.180,
        0,
        0.3,
        0.3,
        1.000
    ),
    ObjectInfo(
        "014_lemon",
        "ycb_models/014_lemon/google_512k/textured.obj",
        0.100,
        0,
        0.3,
        0.4,
        1.000
    ),
    ObjectInfo(
        "015_peach",
        "ycb_models/015_peach/google_512k/textured.obj",
        0.150,
        0,
        0.0,
        0.4,
        1.000
    ),
    ObjectInfo(
        "016_pear",
        "ycb_models/016_pear/google_512k/textured.obj",
        0.170,
        0,
        0.1,
        0.3,
        1.000
    ),
    ObjectInfo(
        "017_orange",
        "ycb_models/017_orange/google_512k/textured.obj",
        0.150,
        0,
        0.3,
        0.4,
        1.000
    ),
    ObjectInfo(
        "018_plum",
        "ycb_models/018_plum/google_512k/textured.obj",
        0.100,
        0,
        0.3,
        0.3,
        1.000
    ),
    ObjectInfo(
        "019_pitcher_base",
        "ycbv_models/models_fine/obj_000011.ply",
        0.178 + 0.066,
        0,
        0.1,
        0.5,
        0.001,
    ),
    ObjectInfo(
        "021_bleach_cleanser",
        "ycbv_models/models_fine/obj_000012.ply",
        1.131,
        0,
        0.1,
        0.5,
        0.001,
    ),
    ObjectInfo(
        "022_windex_bottle",
        "ycb_models/022_windex_bottle/google_512k/textured.obj",
        1.022,
        FLAG_CONCAVE,
        0.1,
        0.5,
        1.0,
    ),
    ObjectInfo(
        "024_bowl",
        "ycbv_models/models_fine/obj_000013.ply",
        0.147,
        FLAG_CONCAVE,
        0.6,
        0.3,
        0.001,
    ),
    ObjectInfo(
        "025_mug",
        "ycbv_models/models_fine/obj_000014.ply",
        0.118,
        FLAG_CONCAVE,
        0.6,
        0.3,
        0.001,
    ),
    ObjectInfo(
        "026_sponge",
        "ycb_models/026_sponge/google_512k/textured.obj",
        0.006,
        0,
        0.0,
        1.0,
        1.0,
    ),
    ObjectInfo(
        "028_skillet_lid",
        "ycb_models/028_skillet_lid/google_512k/textured.obj",
        0.652,
        FLAG_CONCAVE,
        0.8,
        0.1,
        1.0,
    ),
    ObjectInfo(
        "029_plate",
        "ycb_models/029_plate/google_512k/textured.obj",
        0.279,
        FLAG_CONCAVE,
        0.6,
        0.3,
        1.0,
    ),
    ObjectInfo(
        "030_fork",
        "ycb_models/030_fork/google_64k/textured.obj",
        0.034,
        FLAG_CONCAVE,
        0.6,
        0.3,
        1.0,
    ),
    ObjectInfo(
        "031_spoon",
        "ycb_models/031_spoon/google_64k/textured.obj",
        0.030,
        FLAG_CONCAVE,
        0.6,
        0.3,
        1.0,
    ),
    ObjectInfo(
        "032_knife",
        "ycb_models/032_knife/google_64k/textured.obj",
        0.031,
        FLAG_CONCAVE,
        0.6,
        0.3,
        1.0,
    ),
    ObjectInfo(
        "033_spatula",
        "ycb_models/033_spatula/google_512k/textured.obj",
        0.052,
        FLAG_CONCAVE,
        0.3,
        0.5,
        1.0,
    ),
    ObjectInfo(
        "035_power_drill",
        "ycbv_models/models_fine/obj_000015.ply",
        0.895,
        FLAG_CONCAVE,
        0.1,
        0.6,
        0.001,
    ),
    # ObjectInfo('036_wood_block',        'ycbv_models/models_fine/obj_000016_rotated.ply', 0.729, 0,             0.3, 0.5, 0.001),  # edited
    ObjectInfo(
        "036_wood_block",
        "ycbv_models/models_fine/obj_000016_rotated.ply",
        0.729,
        0,
        0.3,
        0.5,
        0.001,
    ),  # edited
    ObjectInfo(
        "037_scissors",
        "ycbv_models/models_fine/obj_000017.ply",
        0.082,
        0,
        0.1,
        0.5,
        0.001,
    ),
    ObjectInfo(
        "038_padlock",
        "ycb_models/038_padlock/google_512k/textured.obj",
        0.208,
        FLAG_CONCAVE,
        0.9,
        0.3,
        1.0,
    ),
    ObjectInfo(
        "040_large_marker",
        "ycbv_models/models_fine/obj_000018.ply",
        0.016,
        0,
        0.1,
        0.5,
        0.001,
    ),
    ObjectInfo(
        "042_adjustable_wrench",
        "ycb_models/042_adjustable_wrench/google_512k/textured.obj",
        0.252,
        FLAG_CONCAVE,
        0.9,
        0.3,
        1.0,
    ),
    ObjectInfo(
        "051_large_clamp",
        "ycbv_models/models_fine/obj_000019.ply",
        0.125,
        0,
        0.1,
        0.5,
        0.001,
    ),
    ObjectInfo(
        "052_extra_large_clamp",
        "ycbv_models/models_fine/obj_000020.ply",
        0.202,
        0,
        0.1,
        0.5,
        0.001,
    ),
    ObjectInfo(
        "061_foam_brick",
        "ycbv_models/models_fine/obj_000021.ply",
        0.028,
        0,
        0.1,
        0.7,
        0.001,
    ),
    ObjectInfo(
        "062_dice",
        "ycb_models/062_dice/google_64k/textured.obj",
        0.006,
        0,
        0.3,
        0.5,
        1.0
    ),
    ObjectInfo(
        "table",
        "other_models/table/table.obj",
        30.000,
        0,
        0.3,
        0.5,
        0.010
    ),
    ObjectInfo(
        "red_bowl",
        "other_models/red_bowl/red_bowl.obj",
        7.000,
        FLAG_CONCAVE,
        0.3,
        0.1,
        0.1,
    ),  # https://sketchfab.com/3d-models/bowl-13339ceb5ddc44aaa8b69b7cee51d9c3
    ObjectInfo(
        "bowling_ball",
        "other_models/bowling_ball/ball_centered.obj",
        7.000,
        0,
        0.3,
        0.1,
        0.010,
    ),  # https://free3d.com/3d-model/-bowling-ball-v1--953922.html
    ObjectInfo(
        "beach_ball",
        "other_models/beach_ball/beach_ball_centered.obj",
        0.100,
        0,
        0.3,
        0.1,
        0.010,
    ),  # https://free3d.com/3d-model/beach-ball-v2--259926.html
    ObjectInfo(
        "wooden_bowl",
        "other_models/wooden_bowl/wooden_bowl.blend",
        5.000,
        FLAG_CONCAVE,
        0.3,
        0.3,
        0.180,
    ),  # "Wooden Bowl" (https://skfb.ly/6XMIP) by MIKErowaveoven is licensed under Creative Commons Attribution (http://creativecommons.org/licenses/by/4.0/).
]


# pre-defined object sets
FLAT_OBJS = [str(i).zfill(3) for i in range(0, 11)]
FRUIT_OBJS = [str(i).zfill(3) for i in range(11, 19)]
CLEANING_OBJS = [str(i).zfill(3) for i in range(19, 22)]
KITCHEN_OBJS = [str(i).zfill(3) for i in range(22, 35)]
DICE_OBJS = [str(i).zfill(3) for i in [5, 7, 8, 9]]

YCBV_OBJECTS = [obj for obj in OBJECT_INFO if obj.name[0].isdigit()]
BOWL = [obj for obj in OBJECT_INFO if obj.name == "red_bowl"]
TABLE = [obj for obj in OBJECT_INFO if obj.name == "table"]
BOWLING_BALL = [obj for obj in OBJECT_INFO if obj.name == "bowling_ball"]
BEACH_BALL = [obj for obj in OBJECT_INFO if obj.name == "beach_ball"]
WOOD_BLOCK = [obj for obj in OBJECT_INFO if obj.name == "036_wood_block"]
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
WOODEN_BOWL = [obj for obj in OBJECT_INFO if obj.name == "wooden_bowl"]
