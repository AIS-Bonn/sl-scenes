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
    "z_min": 1.2,
    "z_max": 1.5,
}

# BOWLING
BOWLING_CAM_POS = torch.Tensor([-0.5, 2.4, 1.5])
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
BOWLING_INITIAL_VELOCITY = torch.tensor([2.5, 0, 0])

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
        "040_large_marker",
        "ycbv_models/models_fine/obj_000018.ply",
        0.016,
        0,
        0.1,
        0.5,
        0.001,
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
        "table",
        "other_models/table/table.obj",
        30.000,
        0,
        0.3,
        0.5,
        0.010
    ),
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
    # ObjectInfo('062_dice',              'other_models/bowling_ball/ball.obj',            10.000, 0,             0.3, 0.1, 1.000)
]


# pre-defined object sets (TODO use mapping/frozendict?)
YCBV_OBJECTS = [obj for obj in OBJECT_INFO if obj.name[0].isdigit()]
TABLE = [obj for obj in OBJECT_INFO if obj.name == "table"]
BOWLING_BALL = [obj for obj in OBJECT_INFO if obj.name == "bowling_ball"]
BEACH_BALL = [obj for obj in OBJECT_INFO if obj.name == "beach_ball"]
WOOD_BLOCK = [obj for obj in OBJECT_INFO if obj.name == "036_wood_block"]
BILLIARDS_OBJECTS = [
    obj for obj in OBJECT_INFO if obj.name.split("_")[0] in ["002", "003", "007", "008"]
]


#
