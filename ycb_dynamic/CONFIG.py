"""
Configurations and variables for each scene.
"""
import torch
import ycb_dynamic.CONSTANTS as CONSTANTS

"""
Explanations:
 - camera parameters:
   - lookat: point of camera focus and starting point for cam_pos calculation.
   - elevation angle (in degrees for better readability): val==0 means same height as lookat, val>0 means above lookat
   - orientation angle (in degrees): val==0 means cam sits on XZ-plane,
    increasing val shifts the pos clockwise when looking from above
   - distance: length of displacement vector from lookat
"""

CONFIG = {
    "room": {
        "prob_assembled": 0.3
    },
    "decorator": {
        "decorations": CONSTANTS.CHAIRS,
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
                "x_min": 0.8,
                "x_max": 1.0,
                "y_min": -0.2,
                "y_max": 0.2,
                "z_min": 0.1,
                "z_max": 0.15,
            },
            "endeffector_pos": {
                "x": 1.2,
                "y_1": -0.3,
                "y_2": 0.3,
                "z_min": -0.15,
                "z_max": -0.1,
            },
            "waypoint_pos": {
                "x_min": 0.55,
                "x_max": 0.65,
                "z_min": -0.15,
                "z_max": -0.1,
            },
            "velocity": {},
            "other": {}
        },
    }
}
