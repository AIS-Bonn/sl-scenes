import numpy as np
import torch
from typing import List

camera_movement_constraints = {
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
}


class Camera(object):
    def __init__(self, name: str, elev_angle: float, ori_angle: float, distance: float, lookat: torch.Tensor,
                 stereo_pair_dist: float, stereo_positions: List[str], movement_complexity: int):
        self.name = name  # can be used e.g. to name the corresponding output directories
        self.movement_complexity = movement_complexity
        self.moving = self.movement_complexity > 0
        self.stereo_pair_dist = stereo_pair_dist
        self.stereo_positions = stereo_positions
        self.start_elev_angle = elev_angle
        self.start_ori_angle = ori_angle
        self.start_distance = distance
        self.start_base_lookat = lookat
        self.start_base_pos = cam_pos_from_config(self.start_base_lookat, self.start_elev_angle,
                                                  self.start_ori_angle, self.start_distance)

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


class TimeDependentCamParamFunc(object):
    def __init__(self, start_val, end_val, start_t, end_t):
        self.start_val = start_val
        self.end_val = end_val
        self.start_t = start_t
        self.end_t = end_t

    def get_value(self, t):
        raise NotImplementedError


class ConstFunc(TimeDependentCamParamFunc):
    def __init__(self, start_val, end_val, start_t, end_t):
        super(ConstFunc, self).__init__(start_val, end_val, start_t, end_t)

    def get_value(self, t):
        return self.start_val


class LinFunc(TimeDependentCamParamFunc):
    def __init__(self, start_val, end_val, start_t, end_t):
        super(LinFunc, self).__init__(start_val, end_val, start_t, end_t)
        self.slope = (self.end_val - self.start_val) / (self.end_t - self.start_t)

    def get_value(self, t):
        return self.slope * t + self.start_val


class SinFunc(TimeDependentCamParamFunc):
    def __init__(self, start_val, end_val, start_t, end_t):
        super(SinFunc, self).__init__(start_val, end_val, start_t, end_t)
        self.amp = self.end_val - self.start_val
        self.freq_mod = np.random.uniform(0.3, 0.6)

    def sin(self, x):
        return np.sin(self.freq_mod * (self.end_t - self.start_t) * x)

    def get_value(self, t):
        return self.start_val + self.amp * self.sin(t)


class LinFuncOnce(TimeDependentCamParamFunc):
    def __init__(self, start_val, end_val, start_t, end_t):
        super(LinFuncOnce, self).__init__(start_val, end_val, start_t, end_t)

    def get_value(self, t):
        t_rel = (t - self.start_t) / (self.end_t - self.start_t)
        t_rel = min(1, max(0, t_rel))
        return t_rel * self.end_val + (1 - t_rel) * self.start_val


class TanhFunc(TimeDependentCamParamFunc):
    def __init__(self, start_val, end_val, start_t, end_t):
        super(TanhFunc, self).__init__(start_val, end_val, start_t, end_t)
        self.freq_mod = np.random.uniform(4, 8)

    def tanh(self, x):
        return np.tanh(self.freq_mod * x - (self.freq_mod / 2))

    def get_value(self, t):
        t_rel = (t - self.start_t) / (self.end_t - self.start_t)
        return self.start_val + (self.end_val - self.start_val) * (1 + self.tanh(t_rel))
