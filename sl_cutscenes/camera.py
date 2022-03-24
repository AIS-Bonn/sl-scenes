import numpy as np
import torch
from typing import List
from sl_cutscenes.constants import SCENARIO_DEFAULTS
from sl_cutscenes.utils.camera_utils import ConstFunc, LinFunc, LinFuncOnce, SinFunc, TanhFunc

camera_movement_constraints = SCENARIO_DEFAULTS["camera_movement"]


class Camera(object):
    '''
    The camera object provides a range of functionalities to better model real-world (stereo) cameras
    that might move around or behave otherwise as time passes.
    '''
    def __init__(self, name: str, cam_dt: float, elev_angle: float, ori_angle: float, distance: float,
                 lookat: torch.Tensor, stereo_pair_dist: float, stereo_positions: List[str], movement_complexity: int):
        self.name = name  # can be used e.g. to name the corresponding output directories
        self.cam_dt = cam_dt
        self.movement_complexity = movement_complexity
        self.moving = self.movement_complexity > 0
        self.stereo_pair_dist = stereo_pair_dist
        self.stereo_positions = stereo_positions
        self.start_elev_angle = elev_angle
        self.start_ori_angle = ori_angle
        self.start_distance = distance
        self.start_base_lookat = lookat
        self.start_up_vec = torch.tensor([0.0, 0.0, 1.0])  # TODO adjustable up vector instead of [0, 0, 1]

        self.reset_cam()
        self.setup_cam_pos_func()

    def reset_cam(self):
        self.t = 0.0  # in seconds
        self.base_lookat = self.start_base_lookat

    def setup_cam_pos_func(self):
        # for each attribute that can be animated, generate a probability number.
        # Keep the <self.movement_complexity> highest ones.
        if self.moving:
            probs_for_movement = np.random.uniform(size=3)
            # the following line doesn't work for N=0. Therefore, the if-statement has to be used.
            probs_for_movement[probs_for_movement.argsort()[:-1 * self.movement_complexity]] = 0
        else:
            probs_for_movement = np.zeros(shape=(3,))
        prob_elev, prob_ori, prob_dist = probs_for_movement

        # Use previously generated random numbers to decide whether a
        # Non-constant movement function will be used for that specific parameter.
        #  - movement function for elevation angle attribute
        if np.random.uniform() >= prob_elev:
            self.elev_angle_func = ConstFunc(self.start_elev_angle, None, None, None)
        else:
            start_val = self.start_elev_angle
            end_val = self.cmc_random_uniform("delta_elev") + start_val
            start_t = self.cmc_random_uniform("t_start")
            end_t = self.cmc_random_uniform("t_duration") + start_t
            elev_func = np.random.choice([SinFunc, LinFuncOnce, TanhFunc])
            self.elev_angle_func = elev_func(start_val, end_val, start_t, end_t)

        #  - movement function for orientation angle attribute
        if np.random.uniform() >= prob_ori:
            self.ori_angle_func = ConstFunc(self.start_ori_angle, None, None, None)
        else:
            start_val = self.start_ori_angle
            end_val = self.cmc_random_uniform("delta_ori") + start_val
            start_t = self.cmc_random_uniform("t_start")
            end_t = self.cmc_random_uniform("t_duration") + start_t
            ori_func = np.random.choice([SinFunc, LinFuncOnce, TanhFunc, LinFunc])
            self.ori_angle_func = ori_func(start_val, end_val, start_t, end_t)

        #  - movement function for distance attribute
        if np.random.uniform() >= prob_dist:
            self.distance_func = ConstFunc(self.start_distance, None, None, None)
        else:
            start_val = self.start_distance
            end_val = self.cmc_random_uniform("delta_dist") + start_val
            start_t = self.cmc_random_uniform("t_start")
            end_t = self.cmc_random_uniform("t_duration") + start_t
            distance_func = np.random.choice([SinFunc, LinFuncOnce, TanhFunc])
            self.distance_func = distance_func(start_val, end_val, start_t, end_t)

    # TODO: use config
    def cmc_random_uniform(self, parameter):
        assert self.movement_complexity > 0
        return np.random.uniform(
            camera_movement_constraints[parameter]["min"][self.movement_complexity - 1],
            camera_movement_constraints[parameter]["max"][self.movement_complexity - 1]
        )

    @property
    def elev_angle(self): return np.clip(self.elev_angle_func.get_value(self.t), 0, 89).item()

    @property
    def ori_angle(self): return self.ori_angle_func.get_value(self.t) % 360

    @property
    def distance(self): return np.clip(self.distance_func.get_value(self.t), 0.8, 5.0).item()

    def stereo_deviation(self, vec, stereo_position):

        deviation_vec = torch.cross(
            (self.base_lookat - self.base_pos).double(), self.start_up_vec.double()
        ).float()
        deviation_vec *= self.stereo_pair_dist / (2 * torch.linalg.norm(deviation_vec))
        return vec - deviation_vec if stereo_position == "left" else vec + deviation_vec

    @property
    def base_pos(self):
        """
        Calculate the camera position from given lookat position, camera distance
        and elevation/orientation angle (in degrees)
        """
        cam_x = np.cos(self.ori_angle * np.pi / 180.) * np.cos(self.elev_angle * np.pi / 180.)
        cam_y = np.sin(self.ori_angle * np.pi / 180.) * np.cos(self.elev_angle * np.pi / 180.)
        cam_z = np.sin(self.elev_angle * np.pi / 180.)
        cam_xyz = torch.tensor([cam_x, cam_y, cam_z])
        cam_pos = self.base_lookat + cam_xyz * self.distance
        return cam_pos

    def get_pos(self, stereo_position="mono"):
        pos = self.base_pos
        return pos if stereo_position == "mono" else self.stereo_deviation(pos, stereo_position)

    def get_lookat(self, stereo_position="mono"):
        lookat = self.base_lookat
        return lookat if stereo_position == "mono" else self.stereo_deviation(lookat, stereo_position)

    def get_posed_name(self, stereo_position="mono"):
        return f"{self.name}_{stereo_position}"

    def step(self, dt=None):
        dt = dt or self.cam_dt
        self.t += dt
