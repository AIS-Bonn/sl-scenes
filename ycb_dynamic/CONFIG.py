"""
Configurations and variables for each scene.
"""
import torch

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
    "scenes": {
        # BILLARDS SCENE
        "billards": {
            "camera": {
                "elevation_angle_min": 15,
                "elevation_angle_max": 15,
                "orientation_angle_min": 0,
                "orientation_angle_max": 360,
                "orientation_angle_default": 256,
                "distance_min": 2.45,
                "distance_max": 2.45,
                "lookat": torch.Tensor([0, 0, 0])
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
                "orientation_angle_default": 294,
                "distance_min": 2.0,
                "distance_max": 2.0,
                "lookat": torch.Tensor([0, 0, 0])
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
                "elevation_angle_min": 7,
                "elevation_angle_max": 7,
                "orientation_angle_min": 0,
                "orientation_angle_max": 360,
                "orientation_angle_default": 294,
                "distance_min": 2.45,
                "distance_max": 2.45,
                "lookat": torch.Tensor([0, 0, 0])
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
        # STACK SCENE
        "stack": {
            "camera": {
                "elevation_angle_min": 15,
                "elevation_angle_max": 15,
                "orientation_angle_min": 0,
                "orientation_angle_max": 360,
                "orientation_angle_default": 294,
                "distance_min": 2.45,
                "distance_max": 2.45,
                "lookat": torch.Tensor([0, 0, 0])
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
        # THROW SCENE
        "throw": {
            "camera": {
                "elevation_angle_min": 15,
                "elevation_angle_max": 15,
                "orientation_angle_min": 0,
                "orientation_angle_max": 360,
                "orientation_angle_default": 256,
                "distance_min": 2.45,
                "distance_max": 2.45,
                "lookat": torch.Tensor([0, 0, 0])
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
        # DICE ROLL SCENE
        "dice_roll": {
            "camera": {
                "elevation_angle_min": 15,
                "elevation_angle_max": 15,
                "orientation_angle_min": 0,
                "orientation_angle_max": 360,
                "orientation_angle_default": 294,
                "distance_min": 2.45,
                "distance_max": 2.45,
                "lookat": torch.Tensor([0, 0, 0])
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
        }
    }
}
