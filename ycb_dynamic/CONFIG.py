"""
Configurations and variables for each scene.
"""
import torch

CONFIG = {
    "scenes": {
        # BILLARDS SCENE
        "billards": {
            "camera": {},
            "pos": {  # bowling ball position
                "x_min": -0.9,
                "x_max": -0.9,
                "y_min": 0.,
                "y_max": 0.,
                "z_min": 0.64,
                "z_max": 0.64,
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
            "camera": {},
            "position": {},
            "velocity": {},
            "other": {
                "min_objs": 2,
                "max_objs": 7
            },
        },
        # BOWLING SCENE
        "bowling": {
            "camera": {},
            "pos": {  # bowling ball position
                "x_min": -0.9,
                "x_max": -0.9,
                "y_min": 0.,
                "y_max": 0.,
                "z_min": 0.64,
                "z_max": 0.64,
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
            "camera": {},
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
            "camera": {},
            "pos": {
                "x_min": -0.6,
                "x_max": 0.6,
                "y_min": -1,
                "y_max": -1,
                "z_min": 0.6,
                "z_max": 1.0,
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
            "camera": {},
            "pos": {
                "x_min": -1.2,
                "x_max": -1.2,
                "y_min": -0.4,
                "y_max": 0.4,
                "z_min": 0.6,
                "z_max": 1.0,
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
