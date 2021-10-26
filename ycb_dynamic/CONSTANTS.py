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

#########################
# Scene global configurations
#########################
# TODO: Cameras are scene-dependent and variable --> move to CONFIG
CAM_POS = torch.Tensor([0.5, 2.4, 1.5])
CAM_LOOKAT = torch.Tensor([0, 0, 1.2])
BOWL_CAM_POS = torch.Tensor([-0.3, 1.2, 2.1])
BOWL_CAM_LOOKAT = torch.Tensor([0, 0, 1.25])
BOWLING_CAM_POS = torch.Tensor([-0.5, 2.4, 1.5])

# TABLE
TABLE_POSE = torch.tensor([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0.6], [0, 0, 0, 1]])

# BOWL
WOODEN_BOWL_POSE = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.5], [0, 0, 0, 1]])
BOWL_FRUIT_INIT_POS = torch.tensor([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0.8], [0, 0, 0, 1]])
RED_BOWL_POSE = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.68], [0, 0, 0, 1]])

# BOWLING
WOOD_BLOCK_POSES = [
    torch.tensor([[1, 0, 0, 0.400], [0, 1, 0, -0.20], [0,  0, 1, 0.64], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.400], [0, 1, 0,     0], [0,  0, 1, 0.64], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.400], [0, 1, 0,  0.20], [0,  0, 1, 0.64], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.398], [0, 0, 1, -0.11], [0, -1, 0, 0.79], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.398], [0, 0, 1,  0.11], [0, -1, 0, 0.79], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.396], [0, 1, 0,  -0.1], [0,  0, 1, 0.94], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.396], [0, 1, 0,   0.1], [0,  0, 1, 0.94], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.395], [0, 0, 1,     0], [0, -1, 0, 1.09], [0, 0, 0, 1]]),
    #torch.tensor([[1, 0, 0, 0.395], [0, 1, 0,     0], [0,  0, 1, 1.84], [0, 0, 0, 1]]),
]
# BILLIARDS
BILLIARDS_TRIANLGE_POSES = [
    torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.64], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.10], [0, 1, 0, -0.06], [0, 0, 1, 0.64], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.10], [0, 1, 0, 0.06], [0, 0, 1, 0.64], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.20], [0, 1, 0, 0], [0, 0, 1, 0.64], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.20], [0, 1, 0, 0.12], [0, 0, 1, 0.64], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.20], [0, 1, 0, -0.12], [0, 0, 1, 0.64], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.30], [0, 1, 0, -0.06], [0, 0, 1, 0.64], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.30], [0, 1, 0, 0.06], [0, 0, 1, 0.64], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.30], [0, 1, 0, -0.18], [0, 0, 1, 0.64], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.30], [0, 1, 0, 0.18], [0, 0, 1, 0.64], [0, 0, 0, 1]]),
]

# STACK PYRAMID (4-1-1)
STACK_PYRAMID_POSES = [
    torch.tensor([[1, 0, 0, -0.05], [0, 1, 0, -0.05], [0, 0, 1, 0.63], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, -0.05], [0, 1, 0, 0.05], [0, 0, 1, 0.63], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.05], [0, 1, 0, -0.05], [0, 0, 1, 0.63], [0, 0, 0, 1]]),
    torch.tensor([[1, 0, 0, 0.05], [0, 1, 0, 0.05], [0, 0, 1, 0.63], [0, 0, 0, 1]]),
    #
    torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.8], [0, 0, 0, 1]]),
    #
    torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.95], [0, 0, 0, 1]]),
]


#########################
# Object Information
#########################
FLAG_CONCAVE = 1 << 0
ObjectInfo = namedtuple(
    "ObjectInfo",
    ["name", "mesh_fp", "weight", "flags", "metallic", "roughness", "scale"],
)

"""
YCB object weight sources: http://www.ycbbenchmarks.com/wp-content/uploads/2015/09/object-list-Sheet1.pdf
A few notes on the 'scale' parameter: stillleben is completely metric, so non-metric meshes need to be scaled:
 - YCB-Video (BOP version) in millimeters -> scale = 0.001
 - YCB Objects (the originals) in meters -> scale = 1.0
 - Other objects: many are in cm, some are whatever, some we'd like to be bigger/smaller... -> varying scales
"""
# source:
# BOP-Compatible YCB-Video models use millimeter units for some strange reason.
# -> sl is completely metric, so scale accordingly.
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
        "029_plate",
        "ycb_models/029_plate/google_512k/textured.obj",
        0.279,
        FLAG_CONCAVE,
        0.6,
        0.3,
        1.0,
    ),  # BAD QUALITY
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
    ObjectInfo(
        "036_wood_block",
        "ycbv_models/models_fine/obj_000016_rotated.ply",  # "ycbv_models/models_fine/obj_000016.ply"
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
        "043_phillips_screwdriver",
        "ycb_models/043_phillips_screwdriver/google_512k/textured.obj",
        0.097,
        0,  # concave?
        0.1,
        0.5,
        1.0,
    ),
    ObjectInfo(
        "044_flat_screwdriver",
        "ycb_models/044_flat_screwdriver/google_512k/textured.obj",
        0.098,
        0,  # concave?
        0.1,
        0.5,
        1.0,
    ),
    ObjectInfo(
        "048_hammer",
        "ycb_models/048_hammer/google_512k/textured.obj",
        0.665,
        FLAG_CONCAVE,  # concave?
        0.1,
        0.5,
        1.0,
    ),
    ObjectInfo(
        "050_medium_clamp",
        "ycb_models/050_medium_clamp/google_512k/textured.obj",
        0.059,
        FLAG_CONCAVE,  # concave?
        0.1,
        0.5,
        1.0,
    ),
    ObjectInfo(
        "051_large_clamp",
        "ycbv_models/models_fine/obj_000019.ply",
        0.125,
        FLAG_CONCAVE,  # concave?
        0.1,
        0.5,
        0.001,
    ),
    ObjectInfo(
        "052_extra_large_clamp",
        "ycbv_models/models_fine/obj_000020.ply",
        0.202,
        FLAG_CONCAVE,  # concave?
        0.1,
        0.5,
        0.001,
    ),
    ObjectInfo(
        "053_mini_soccer_ball",
        "ycb_models/053_mini_soccer_ball/google_512k/textured.obj",
        0.123,
        0,
        0.1,
        0.3,
        1.0,
    ),
    ObjectInfo(
        "054_softball",
        "ycb_models/054_softball/google_512k/textured.obj",
        0.191,
        0,
        0.0,
        0.7,
        1.0,
    ),
    ObjectInfo(
        "055_baseball",
        "ycb_models/055_baseball/google_512k/textured.obj",
        0.138,
        0,
        0.1,
        0.5,
        1.0,
    ),
    ObjectInfo(
        "056_tennis_ball",
        "ycb_models/056_tennis_ball/google_512k/textured.obj",
        0.058,
        0,
        0.0,
        0.9,
        1.0,
    ),
    ObjectInfo(
        "057_racquetball",
        "ycb_models/057_racquetball/google_512k/textured.obj",
        0.041,
        0,
        0.1,
        0.6,
        1.0,
    ),
    ObjectInfo(
        "058_golf_ball",
        "ycb_models/058_golf_ball/google_512k/textured.obj",
        0.046,
        0,
        0.3,
        0.5,
        1.0,
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
        "065-a_cups",
        "ycb_models/065-a_cups/google_512k/textured.obj",
        0.013,
        FLAG_CONCAVE,
        0.1,
        0.5,
        1.0,
    ),  # BAD QUALITY (interior)
    ObjectInfo(
        "065-b_cups",
        "ycb_models/065-b_cups/google_512k/textured.obj",
        0.014,
        FLAG_CONCAVE,
        0.1,
        0.5,
        1.0,
    ),  # BAD QUALITY (interior)
    ObjectInfo(
        "065-c_cups",
        "ycb_models/065-c_cups/google_512k/textured.obj",
        0.017,
        FLAG_CONCAVE,
        0.1,
        0.5,
        1.0,
    ),  # BAD QUALITY (interior)
    ObjectInfo(
        "065-d_cups",
        "ycb_models/065-d_cups/google_512k/textured.obj",
        0.019,
        FLAG_CONCAVE,
        0.1,
        0.5,
        1.0,
    ),  # BAD QUALITY (interior)
    ObjectInfo(
        "065-e_cups",
        "ycb_models/065-e_cups/google_512k/textured.obj",
        0.021,
        FLAG_CONCAVE,
        0.1,
        0.5,
        1.0,
    ),  # BAD QUALITY (interior)
    ObjectInfo(
        "065-f_cups",
        "ycb_models/065-f_cups/google_512k/textured.obj",
        0.026,
        FLAG_CONCAVE,
        0.1,
        0.5,
        1.0,
    ),  # BAD QUALITY (interior)
    ObjectInfo(
        "065-g_cups",
        "ycb_models/065-g_cups/google_512k/textured.obj",
        0.028,
        FLAG_CONCAVE,
        0.1,
        0.5,
        1.0,
    ),  # BAD QUALITY (interior)
    ObjectInfo(
        "065-h_cups",
        "ycb_models/065-h_cups/google_512k/textured.obj",
        0.031,
        FLAG_CONCAVE,
        0.1,
        0.5,
        1.0,
    ),  # BAD QUALITY (interior)
    ObjectInfo(
        "065-i_cups",
        "ycb_models/065-i_cups/google_512k/textured.obj",
        0.035,
        FLAG_CONCAVE,
        0.1,
        0.5,
        1.0,
    ),  # BAD QUALITY (interior)
    ObjectInfo(
        "065-j_cups",
        "ycb_models/065-j_cups/google_512k/textured.obj",
        0.038,
        FLAG_CONCAVE,
        0.1,
        0.5,
        1.0,
    ),  # BAD QUALITY (interior)
    ObjectInfo(
        "070-b_colored_wood_blocks",
        "ycb_models/070-b_colored_wood_blocks/google_64k/textured.obj",
        0.011,
        0,
        0.3,
        0.5,
        1.0,
    ),
    ObjectInfo(
        "071_nine_hole_peg_test",
        "ycb_models/071_nine_hole_peg_test/google_512k/textured.obj",
        1.435,
        0,
        0.4,
        0.5,
        1.0,
    ),
    ObjectInfo(
        "072-a_toy_airplane",
        "ycb_models/072-a_toy_airplane/google_512k/textured.obj",
        0.570,
        FLAG_CONCAVE,
        0.3,
        0.5,
        1.0,
    ),
    ObjectInfo(
        "077_rubiks_cube",
        "ycb_models/077_rubiks_cube/google_512k/textured.obj",
        0.094,
        0,
        0.1,
        0.4,
        1.0,
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
        "pool_table",
        "other_models/pool_table/source/pool_table.fbx",
        30.000,
        0,
        0.3,
        0.5,
        0.010
    ),
    ObjectInfo(
        "red_bowl",
        "other_models/red_bowl/red_bowl.blend",
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


#########################
# Pre-defined object sets
#########################
FLAT_OBJS = [str(i).zfill(3) for i in range(0, 11)]
FRUIT_OBJS = [str(i).zfill(3) for i in range(11, 19)]
CLEANING_OBJS = [str(i).zfill(3) for i in range(19, 22)]
KITCHEN_OBJS = [str(i).zfill(3) for i in range(22, 35)]
DICE_OBJS = [str(i).zfill(3) for i in [5, 7, 8, 9]]
YCBV_OBJS = [str(i).zfill(3) for i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 19, 21, 24, 25, 35, 36, 37, 40, 51, 52, 61]]

YCBV_OBJECTS = [obj for obj in OBJECT_INFO if obj.name[0].isdigit()][:30]
TABLE = [obj for obj in OBJECT_INFO if obj.name == "table"]
# TABLE = [obj for obj in OBJECT_INFO if obj.name == "pool_table"]
BOWLING_BALL = [obj for obj in OBJECT_INFO if obj.name == "bowling_ball"]
BEACH_BALL = [obj for obj in OBJECT_INFO if obj.name == "beach_ball"]
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
# BOWL = [obj for obj in OBJECT_INFO if obj.name in ["red_bowl", "wooden_bowl"]]
BOWL = [obj for obj in OBJECT_INFO if obj.name == "red_bowl"]
WOODEN_BOWL = [obj for obj in OBJECT_INFO if obj.name == "wooden_bowl"]

#
