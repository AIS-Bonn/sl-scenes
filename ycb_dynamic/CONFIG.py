"""
Configurations and variables
"""
import torch

CONFIG = {
    "scenes": {
        "stack": {
            "stacks_min": 1,
            "stacks_max": 3,
            "x_disp_max": 1,
            "x_disp_min": -1,
            "x_disp_step": 0.5,
            "y_disp_max": 0.5,
            "y_disp_min": -0.5,
            "y_disp_step": 0.2,
            "pos_noise_mean": 0,
            "pos_noise_std": 0.1,
        },
        "throw": {
            "linear_velocity": torch.Tensor([0, 2, 1.5]),
            "linear_noise_mean": 0,
            "linear_noise_std": 0.3,
            "angular_velocity": torch.Tensor([0, 0, 0]),
            "angular_noise_mean": 0,
            "angular_noise_std": 0.1,
        }
    }
}
