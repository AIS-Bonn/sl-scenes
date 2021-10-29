import numpy as np
import torch
from copy import deepcopy


class Camera(object):
    def __init__(self, name: str, elev_angle: float, ori_angle: float, distance: float, lookat: torch.Tensor,
                 up: torch.Tensor=torch.tensor([0., 0., 1.]), moving: bool=False, stereo_position: str="mono",
                 stereo_pair_dist: float=0):
        self.name = f"{name}_{stereo_position}" # can be used e.g. to name the corresponding output directories
        self.moving = moving
        self.stereo_position = stereo_position
        self.stereo_pair_dist = stereo_pair_dist
        self.start_elev_angle = elev_angle
        self.start_ori_angle = ori_angle
        self.start_distance = distance
        self.start_up = up
        self.start_base_lookat = lookat
        self.start_base_pos = cam_pos_from_config(self.start_base_lookat, self.start_elev_angle,
                                                  self.start_ori_angle, self.start_distance)

        self.reset_cam()

    def reset_cam(self):
        self.t = 0
        self.base_lookat = self.start_base_lookat
        self.elev_angle = self.start_elev_angle
        self.ori_angle = self.start_ori_angle
        self.distance = self.start_distance
        self.up = self.start_up

    def stereo_deviation(self, vec):
        deviation_vec = torch.cross((self.base_lookat - self.base_pos).double(), self.up.double()).float()
        deviation_vec *= self.stereo_pair_dist / (2 * torch.linalg.norm(deviation_vec))
        return vec - deviation_vec if self.stereo_position == "left" else vec + deviation_vec

    @property
    def base_pos(self):
        return self.start_base_pos if not self.moving else \
            cam_pos_from_config(self.base_lookat, self.elev_angle, self.ori_angle, self.distance)

    @property
    def pos(self):
        pos = self.base_pos
        return pos if self.stereo_position == "mono" else self.stereo_deviation(pos)

    @property
    def lookat(self):
        lookat = self.base_lookat
        return lookat if self.stereo_position == "mono" else self.stereo_deviation(lookat)

    def step(self):
        if not self.moving:
            return
        else:
            raise NotImplementedError("TODO implement position/lookat change on step() invocation")
            # TODO respect stereo lookat deviation in movement!

def create_coplanar_stereo_cams(name: str, elev_angle: float, ori_angle: float, distance: float,
                                lookat: torch.Tensor, stereo_pair_dist: float,
                                up: torch.Tensor=torch.tensor([0., 0., 1.]), moving: bool=False):
    """
    Creates a pair of coplanar stereo cameras.
    Both cameras deviate by half the specified distance from the given position and lookat.
    """
    return Camera(name, elev_angle, ori_angle, distance, lookat, up=up, moving=moving,
                  stereo_position="left", stereo_pair_dist=stereo_pair_dist), \
           Camera(name, elev_angle, ori_angle, distance, lookat, up=up, moving=moving,
                  stereo_position="right", stereo_pair_dist=stereo_pair_dist),

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