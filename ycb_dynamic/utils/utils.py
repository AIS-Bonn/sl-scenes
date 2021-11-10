"""
Utils methods
"""

import os
import random
import datetime
import torch
import argparse

import ycb_dynamic.CONSTANTS as CONSTANTS

PI = torch.acos(torch.tensor(-1))


def clear_cmd():
    """Clearning command line window"""
    os.system("cls" if os.name == "nt" else "clear")
    return


def timestamp():
    """ Obtaining the current timestamp in an human-readable way """

    timestamp = (
        str(datetime.datetime.now()).split(".")[0].replace(" ", "_").replace(":", "-")
    )

    return timestamp


def get_rot_matrix(angles):
    """ Generating a rotation matrix given the rotation angles """
    yaw = torch.tensor([
        [torch.cos(angles[0]), -torch.sin(angles[0]), 0],
        [torch.sin(angles[0]), torch.cos(angles[0]), 0],
        [0, 0, 1]
    ])
    pitch = torch.tensor([
        [torch.cos(angles[1]), 0, torch.sin(angles[1])],
        [0, 1, 0],
        [-torch.sin(angles[1]), 0, torch.cos(angles[1])],
    ])
    roll = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(angles[2]), -torch.sin(angles[2])],
        [0, torch.sin(angles[2]), torch.cos(angles[2])],
    ])
    rot_matrix = yaw @ pitch @ roll
    return rot_matrix


def get_angle_from_mat(mat, deg=False):
    """ Obtaining angle from 2D rot mat """
    if(mat[0, 0] < 1e-4 and mat[1, 0].abs() > 0.99):
        ang = PI / 2
    else:
        ang = torch.atan(mat[1, 0] / mat[0, 0])

    ang = ang * 180. / PI if deg else ang
    return ang


def get_rand_num(N=1, low=0, high=1):
    """ Get N random uniformly distributed numbers on the specified range"""
    nums = torch.rand(N,) * (high - low) + low
    return nums


def get_noisy_vect(v, mean, std):
    """ Adding AWGN to the vector v"""
    noise = torch.randn(v.shape) * std + mean
    noisy_v = v + noise
    return noisy_v


def get_surface_height(surface):
    """
    Getting the height of the surface object, e.g. table
    """
    pose = surface.pose
    height = pose[2, -1]
    return height


def positive_integer(var):
    var = int(var)
    if var < 1:
        raise argparse.ArgumentTypeError(f"invalid parameter: {var}")
    return var


def randomize():
    """ Re-randomizing the objects in the room to avoid always having the same textures/objs """
    CONSTANTS.TABLE = [random.choice(CONSTANTS.TABLES)]
    CONSTANTS.BOWL = [random.choice(CONSTANTS.BOWLS)]
    CONSTANTS.ROOM = [random.choice(CONSTANTS.ROOMS)]
    CONSTANTS.FLOOR = [random.choice(CONSTANTS.FLOORS)]
    CONSTANTS.WALL = [random.choice(CONSTANTS.WALLS)]
    CONSTANTS.FURNITURE = [random.choice(CONSTANTS.FURNITURES)]
    return
