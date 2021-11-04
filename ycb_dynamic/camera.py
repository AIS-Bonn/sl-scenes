import numpy as np
import torch
from copy import deepcopy
from typing import List


class Camera(object):
    def __init__(self, name: str, elev_angle: float, ori_angle: float, distance: float, lookat: torch.Tensor,
                 stereo_pair_dist: float, stereo_positions: List[str]=["mono"], moving: bool=False):
        self.name = name # can be used e.g. to name the corresponding output directories
        self.moving = moving
        self.stereo_pair_dist = stereo_pair_dist
        self.stereo_positions = stereo_positions
        self.start_elev_angle = elev_angle
        self.start_ori_angle = ori_angle
        self.start_distance = distance
        self.start_base_lookat = lookat
        self.start_base_pos = cam_pos_from_config(self.start_base_lookat, self.start_elev_angle,
                                                  self.start_ori_angle, self.start_distance)

        self.reset_cam()

    def reset_cam(self):
        self.t = 0.0  # in seconds
        self.base_lookat = self.start_base_lookat
        self.elev_angle = self.start_elev_angle
        self.ori_angle = self.start_ori_angle
        self.distance = self.start_distance

    def stereo_deviation(self, vec, stereo_position):

        # TODO use cam's up vector instead of [0, 0, 1]. This is OK as long as camera is not rolled
        deviation_vec = torch.cross(
            (self.base_lookat - self.base_pos).double(), torch.tensor([0.0, 0.0, 1.0]).double()
        ).float()
        deviation_vec *= self.stereo_pair_dist / (2 * torch.linalg.norm(deviation_vec))
        return vec - deviation_vec if stereo_position == "left" else vec + deviation_vec

    @property
    def base_pos(self):
        return self.start_base_pos if not self.moving else \
            cam_pos_from_config(self.base_lookat, self.elev_angle, self.ori_angle, self.distance)

    def get_pos(self, stereo_position="mono"):
        pos = self.base_pos
        return pos if stereo_position == "mono" else self.stereo_deviation(pos, stereo_position)

    def get_lookat(self, stereo_position="mono"):
        lookat = self.base_lookat
        return lookat if stereo_position == "mono" else self.stereo_deviation(lookat, stereo_position)

    def get_posed_name(self, stereo_position="mono"):
        return f"{self.name}_{stereo_position}"

    def step(self, dt):
        self.t += dt
        if not self.moving:
            return
        else:
            raise NotImplementedError("TODO implement position/lookat change on step() invocation")
            # TODO respect stereo lookat deviation in movement!


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
