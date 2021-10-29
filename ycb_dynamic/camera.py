import numpy as np
import torch


class Camera(object):
    def __init__(self, name: str, start_pos: torch.Tensor, start_lookat: torch.Tensor,
                 moving: bool, stereo_deviation: float=0):
        self.name = name  # can be used e.g. to name the corresponding output directories
        self.start_pos = start_pos
        self.start_lookat = start_lookat
        self.moving = moving
        self.stereo_deviation = stereo_deviation
        self.reset_cam()

    def reset_cam(self):
        self.t = 0
        self.pos = self.start_pos
        self.lookat = self.start_lookat

    def step(self):
        if not self.moving:
            return
        else:
            raise NotImplementedError("TODO implement position/lookat change on step() invocation")
            # TODO respect stereo lookat deviation in movement!

def create_coplanar_stereo_cams(name: str, cam_pos: torch.Tensor, cam_lookat: torch.Tensor,
                                stereo_cam_dist: float, moving: bool):
    """
    Creates a pair of coplanar stereo cameras.
    Both cameras deviate by half the specified distance from the given position and lookat.
    """
    cam_up = torch.tensor([0., 0., 1.])  # TODO variable up vectors
    deviation_magnitude = stereo_cam_dist / 2
    deviation_vec = torch.cross((cam_lookat - cam_pos), cam_up)
    deviation_vec *= deviation_magnitude / torch.linalg.norm(deviation_vec)
    cam_pos_left, cam_pos_right = cam_pos - deviation_vec, cam_pos + deviation_vec
    cam_lookat_left, cam_lookat_right = cam_lookat - deviation_vec, cam_lookat + deviation_vec  # only for coplanar st.
    return Camera(name + "_left", cam_pos_left, cam_lookat_left, moving=moving, stereo_deviation=deviation_magnitude), \
           Camera(name + "_right", cam_pos_right, cam_lookat_right, moving=moving, stereo_deviation=deviation_magnitude)

def cam_pos_from_config(cam_lookat: torch.Tensor, elevation_angle: float, orientation_angle: float, distance: float):
    """
    Calculate the camera position from given lookat position, camera distance
    and elevation/orientation angle (in degrees)
    """
    cam_x = np.cos(orientation_angle * np.pi / 180.) * np.cos(elevation_angle * np.pi / 180.)
    cam_y = np.sin(orientation_angle * np.pi / 180.) * np.cos(elevation_angle * np.pi / 180.)
    cam_z = np.sin(elevation_angle * np.pi / 180.)
    cam_xyz = torch.tensor([cam_x, cam_y, cam_z])
    cam_pos = cam_lookat + cam_xyz * distance
    return cam_pos