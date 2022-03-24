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
from sl_cutscenes.object_info import OBJECT_INFO

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

"""
Scenario parameter defaults:
 - camera parameters:
   - lookat: point of camera focus and starting point for cam_pos calculation.
   - elevation angle (in degrees for better readability): val==0 means same height as lookat, val>0 means above lookat
   - orientation angle (in degrees): val==0 means cam sits on XZ-plane,
    increasing val shifts the pos clockwise when looking from above
   - distance: length of displacement vector from lookat
"""
SCENARIO_DEFAULTS = {
    "room": {
        "prob_assembled": 0.0
    },
    "decorator": {
        "decorations": CHAIRS,
        "min_objs": 2,
        "max_objs": 5,
        "bounds": {
            "min_x": -3,  # limits of the occupancy matrix. Define grid to place objects
            "max_x": 3,
            "min_y": -3,
            "max_y": 3,
            "res": 0.12,  # resolution of the matrix (in meters)
            "dist": 0.0   # minimum distance between objects
        }
    },
    "camera_movement": {
        "delta_elev": {
            "min": [-5, -10, -30],
            "max": [10, 20, 30]
        },
        "delta_ori": {
            "min": [-20, -40, -80],
            "max": [20, 40, 80]
        },
        "delta_dist": {
            "min": [-0.2, -0.5, -1.0],
            "max": [0.2, 0.5, 1.0]
        },
        "t_duration": {
            "min": [2, 1, 0.5],
            "max": [3, 3, 3]
        },
        "t_start": {
            "min": [0, 0, 0],
            "max": [1, 1, 1]
        }
    },
    "scenes": {
        # BALL BOX SCENE
        "ball_box": {
            "camera": {
                "elevation_angle_min": 55,
                "elevation_angle_max": 55,
                "orientation_angle_min": 0,
                "orientation_angle_max": 360,
                "orientation_angle_default": 24,
                "distance_min": 1.2,
                "distance_max": 1.2,
                "base_lookat": torch.Tensor([0, 0, 0.1])
            },
            "pos": {
                "x_min": 0.,
                "x_max": 0,
                "y_min": 0.,
                "y_max": 0.,
                "z_min": 0.1,
                "z_max": 0.3,
            },
            "velocity": {},
            "other": {
                "min_objs": 1,
                "max_objs": 7
            },
        },
        # BILLARDS SCENE
        "billards": {
            "camera": {
                "elevation_angle_min": 15,
                "elevation_angle_max": 15,
                "orientation_angle_min": 0,
                "orientation_angle_max": 360,
                "orientation_angle_default": 114,
                "distance_min": 2.00,
                "distance_max": 2.00,
                "base_lookat": torch.Tensor([0, 0, 0.1])
            },
            "pos": {  # bowling ball position
                "x_min": -0.9,
                "x_max": -0.9,
                "y_min": 0.,
                "y_max": 0.,
                "z_min": 0.0,
                "z_max": 0.2,
            },
            "velocity": {  # bowling ball velocity
                "lin_velocity": torch.tensor([2.5, 0, 0]),
                "lin_noise_mean": 0.,
                "lin_noise_std": 0.1,
            },
            "other": {},
        },
        # BOWL SCENE
        "bowl": {
            "camera": {
                "elevation_angle_min": 40,
                "elevation_angle_max": 40,
                "orientation_angle_min": 0,
                "orientation_angle_max": 360,
                "orientation_angle_default": 72,
                "distance_min": 1.5,
                "distance_max": 1.5,
                "base_lookat": torch.Tensor([0, 0, 0.2])
            },
            "pos": {
                "x_min": 0.,
                "x_max": 0,
                "y_min": 0.,
                "y_max": 0.,
                "z_min": 0.1,
                "z_max": 0.3,
            },
            "velocity": {},
            "other": {
                "min_objs": 2,
                "max_objs": 7
            },
        },
        # BOWLING SCENE
        "bowling": {
            "camera": {
                "elevation_angle_min": 10,
                "elevation_angle_max": 10,
                "orientation_angle_min": 0,
                "orientation_angle_max": 360,
                "orientation_angle_default": 114,
                "distance_min": 2.00,
                "distance_max": 2.00,
                "base_lookat": torch.Tensor([0, 0, 0.2])
            },
            "pos": {  # bowling ball position
                "x_min": -0.9,
                "x_max": -0.9,
                "y_min": 0.,
                "y_max": 0.,
                "z_min": 0.0,
                "z_max": 0.2,
            },
            "velocity": {  # bowling ball velocity
                "lin_velocity": torch.tensor([2.5, 0, 0]),
                "lin_noise_mean": 0.,
                "lin_noise_std": 0.1,
            },
            "other": {},
        },
        # DICE ROLL SCENE
        "dice_roll": {
            "camera": {
                "elevation_angle_min": 15,
                "elevation_angle_max": 15,
                "orientation_angle_min": 0,
                "orientation_angle_max": 360,
                "orientation_angle_default": 60,
                "distance_min": 3.00,
                "distance_max": 3.00,
                "base_lookat": torch.Tensor([0, 0, 0.5])
            },
            "pos": {
                "x_min": -1.2,
                "x_max": -1.2,
                "y_min": -0.4,
                "y_max": 0.4,
                "z_min": 0.2,
                "z_max": 0.6,
            },
            "velocity": {
                "lin_velocity": torch.Tensor([2, 0, 0]),
                "lin_noise_mean": 0,
                "lin_noise_std": 0.3,
                "ang_velocity": torch.Tensor([0, 30, 0]),
                "ang_noise_mean": 0,
                "ang_noise_std": 1,
            },
            "other": {
                "min_objs": 2,
                "max_objs": 7
            },
        },
        # STACK SCENE
        "stack": {
            "camera": {
                "elevation_angle_min": 15,
                "elevation_angle_max": 15,
                "orientation_angle_min": 0,
                "orientation_angle_max": 360,
                "orientation_angle_default": 90,
                "distance_min": 2.00,
                "distance_max": 2.00,
                "base_lookat": torch.Tensor([0, 0, 0.2])
            },
            "pos": {
                "x_min": -1,
                "x_max": 1,
                "y_min": -0.5,
                "y_max": 0.5,
                "z_min": 0,
                "z_max": 0,
            },
            "velocity": {},
            "other": {
                "stacks_min": 1,
                "stacks_max": 3,
            }
        },
        # TABLETOP SCENE
        "tabletop": {
            "camera": {
                "elevation_angle_min": 15,
                "elevation_angle_max": 15,
                "orientation_angle_min": 0,
                "orientation_angle_max": 360,
                "orientation_angle_default": 76,
                "distance_min": 2.00,
                "distance_max": 2.00,
                "base_lookat": torch.Tensor([0, 0, .2])
            },
            "pos": {
                "x_min": -0.8,
                "x_max": 0.8,
                "y_min": -0.4,
                "y_max": 0.4,
                "z_min": 0.1,
                "z_max": 0.3,
            },
            "velocity": {},
            "other": {}
        },
        # THROW SCENE
        "throw": {
            "camera": {
                "elevation_angle_min": 15,
                "elevation_angle_max": 15,
                "orientation_angle_min": 0,
                "orientation_angle_max": 360,
                "orientation_angle_default": 76,
                "distance_min": 2.00,
                "distance_max": 2.00,
                "base_lookat": torch.Tensor([0, 0, .2])
            },
            "pos": {
                "x_min": -0.6,
                "x_max": 0.6,
                "y_min": -1,
                "y_max": -1,
                "z_min": 0.1,
                "z_max": 0.4,
            },
            "velocity": {
                "lin_velocity": torch.Tensor([0, 2, 1.5]),
                "lin_noise_mean": 0,
                "lin_noise_std": 0.3,
                "ang_velocity": torch.Tensor([0, 0, 0]),
                "ang_noise_mean": 0,
                "ang_noise_std": 0.1
            },
            "other": {
                "min_objs": 2,
                "max_objs": 7
            },
        },
        # TIDY SCENE
        "tidy": {
            "camera": {
                "elevation_angle_min": 10,
                "elevation_angle_max": 10,
                "orientation_angle_min": 0,
                "orientation_angle_max": 0,
                "orientation_angle_default": 0,
                "distance_min": 1.00,
                "distance_max": 1.00,
                "base_lookat": torch.Tensor([0.9, 0, .2])
            },
            "pos": {
                "x_min": 0.7,
                "x_max": 0.8,
                "y_min": -0.3,
                "y_max": 0.3,
                "z_min": 0.02,
                "z_max": 0.05,
            },
            "endeffector_pos": {
                "x": 1.0,
                "y_1": -0.45,
                "y_2": 0.45,
                "z_min": 1.1,
                "z_max": 1.1,
            },
            "waypoint_pos": {
                "x_min": 0.55,
                "x_max": 0.65,
                "z_min": -0.14,
                "z_max": -0.14,
            },
            "velocity": {},
            "other": {}
        },
        # ROBOPUSHING SCENE
        "robopushing": {
            "camera": {
                "elevation_angle_min": 15,
                "elevation_angle_max": 15,
                "orientation_angle_min": 0,
                "orientation_angle_max": 360,
                "orientation_angle_default": 76,
                "distance_min": 2.00,
                "distance_max": 2.00,
                "base_lookat": torch.Tensor([0, 0, .2])
            },
            "pos": {
                "x_min": -0.8,
                "x_max": 0.8,
                "y_min": -0.4,
                "y_max": 0.4,
                "z_min": 0.1,
                "z_max": 0.3,
            },
            "velocity": {},
            "other": {}
        },
    }
}
