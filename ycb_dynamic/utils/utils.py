"""
Utils methods
"""
import sys

sys.path.append(".")

import os
import random
import datetime
import shutil

import torch
import argparse
import stillleben as sl

import ycb_dynamic.CONSTANTS as CONSTANTS
import ycb_dynamic.OBJECT_INFO as OBJECT_INFO
import nimblephysics as nimble
from pathlib import Path

PI = torch.acos(torch.tensor(-1))
TAB = "    "

def clear_cmd():
    """Clearing command line window"""
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

def get_rpy_from_mat(mat : torch.Tensor):
    """Get roll-pitch-yaw angle (ZYX euler angle convention) from rotation matrix"""
    # TODO test this! What about singularity handling?
    sy = torch.sqrt(mat[2, 1] ** 2 + mat[2, 2] ** 2)
    yaw = torch.atan2(mat[1, 0], mat[0, 0])
    pitch = torch.atan2(-mat[2, 0], sy)
    roll = 0 if sy < 1e-6 else torch.atan2(mat[2, 1], mat[2, 2])
    return torch.stack([roll, pitch, yaw])

def get_mat_from_rpy(rpy : torch.Tensor):
    """Get rotation matrix from roll-pitch-yaw angle (ZYX euler angle convention)"""
    # TODO test this!
    roll, pitch, yaw = rpy
    Rz_y = torch.tensor([   torch.cos(yaw), -torch.sin(yaw),                0,
                            torch.sin(yaw),  torch.cos(yaw),                0,
                                         0,               0,                1]).view(3, 3)
    Ry_p = torch.tensor([ torch.cos(pitch),               0, torch.sin(pitch),
                                         0,               1,                0,
                         -torch.sin(pitch),               0, torch.cos(pitch)]).view(3, 3)
    Rx_r = torch.tensor([                1,               0,                0,
                                         0, torch.cos(roll), -torch.sin(roll),
                                         0, torch.sin(roll),  torch.cos(roll)]).view(3, 3)
    R = torch.mm(Rz_y, torch.mm(Ry_p, Rx_r))
    return R

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
    '''
    argparse helper for a 'positive integer' type
    '''
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


def copy_overwrite(src, dst):
    if os.path.exists(dst):
        os.remove(dst)
    shutil.copy(src, dst)



def get_absolute_mesh_path(obj_info : OBJECT_INFO):
    floor_or_wall = obj_info.name.endswith("_floor") or obj_info.name.endswith("_wall")
    path = CONSTANTS.TEXT_BASE_DIR if floor_or_wall else CONSTANTS.MESH_BASE_DIR
    return str((path / obj_info.mesh_fp).resolve())


def sl_object_to_nimble(obj : sl.Object, obj_info : OBJECT_INFO):

    skel = nimble.dynamics.Skeleton()
    skel.setName(str(obj.instance_index))
    position, velocity = [], []
    if obj.static:
        skel_joint, skel_body = skel.createWeldJointAndBodyNodePair()
    else:
        skel_joint, skel_body = skel.createFreeJointAndBodyNodePair()
        pose = obj.pose()
        t = pose[:3, 3]
        rpy = get_rpy_from_mat(pose[:3, :3])
        # TODO do the angular velocities need to be converted as well?
        position.extend([rpy, t])
        velocity.extend([obj.angular_velocity, obj.linear_velocity])
    if False:  # obj_info.flags % 2 == 1:  # object is concave -> initialize with sub-parts # TODO deal with non-convex parts
        convex_parts = getattr(obj.mesh, "convex_parts", [None])
        raise NotImplementedError
    else:
        scale = torch.tensor([obj_info.scale] * 3)
        mesh_path = get_absolute_mesh_path(obj_info)
        skel_shape = skel_body.createShapeNode(nimble.dynamics.MeshShape(scale=scale, path=mesh_path))
        # inertia_moment1 = skel_shape.getShape().computeInertia(obj.mass)
        inertia_moment = torch.diag(obj.inertia)
        # print(inertia_moment1)  # roughly equal to obj.inertia
        # print(inertia_moment)
        obj_center_of_mass = obj.inertial_frame[:3, 3]
        skel_body.setInertia(nimble.dynamics.Inertia(obj.mass, obj_center_of_mass, inertia_moment))
    return skel, position, velocity


def dump_sl_scene_to_urdf(scene: sl.Scene, out_fp : str):
    """ Dumps given stillleben scene to a urdf file that can be used by robotics simulators """
    # TODO handle robots here or just the 'environment'? If only env -> exclude robots from scene_objects!
    scene_objects = scene.objects
    urdf_lines = ['<robot name="scene">', TAB + '<link name="world"/>']
    for obj in scene_objects:
        obj_name = str(obj.instance_index)
        obj_pose = obj.pose()
        obj_x, obj_y, obj_z = obj_pose[:3, 3]
        obj_r, obj_p, obj_y = get_rpy_from_mat(obj_pose[:3, :3])

        # link
        urdf_lines.append(1 * TAB + f'<link name="{obj_name}">')

        urdf_lines.append(2 * TAB + '<inertial>')
        urdf_lines.append(3 * TAB + f'<mass value="{str(obj.mass)}"/>')
        urdf_lines.append(3 * TAB + '<origin xyz="0 0 0" rpy="0 0 0"/>')
        urdf_lines.append(3 * TAB + f'<inertia ixx="{0}" ixy="{0}" ixz="{0}" iyy="{0}" iyz="{0}" izz="{0}"/>')
        urdf_lines.append(2 * TAB + '</inertial>')

        urdf_lines.append(2 * TAB + '<visual>')
        urdf_lines.append(3 * TAB + f'<origin xyz="{obj_x} {obj_y} {obj_z}" rpy="{obj_r} {obj_p} {obj_y}"/>')
        urdf_lines.append(3 * TAB + '<geometry>')
        urdf_lines.append(4 * TAB + f'<mesh filename="{obj.mesh.filename}"/>')
        urdf_lines.append(3 * TAB + '</geometry>')
        urdf_lines.append(2 * TAB + '</visual>')

        urdf_lines.append(2 * TAB + '<collision>')
        urdf_lines.append(3 * TAB + f'<origin xyz="{obj_x} {obj_y} {obj_z}" rpy="{obj_r} {obj_p} {obj_y}"/>')
        urdf_lines.append(3 * TAB + '<geometry>')
        urdf_lines.append(4 * TAB + f'<mesh filename="{obj.mesh.filename}"/>')
        urdf_lines.append(3 * TAB + '</geometry>')
        urdf_lines.append(2 * TAB + '</collision>')

        urdf_lines.append(2 * TAB + '<contact_coefficients>')

        urdf_lines.append(1 * TAB + '</link>')

        # joint
        joint_type = "fixed" if obj.static else "floating"  # alt: "fixed"/"floating"?
        urdf_lines.append(1 * TAB + f'<joint name="joint_{obj_name}" type="{joint_type}">')
        urdf_lines.append(2 * TAB + f'<origin xyz="{obj_x} {obj_y} {obj_z}" rpy="{obj_r} {obj_p} {obj_y}"/>')
        urdf_lines.append(2 * TAB + '<parent link="world"/>')
        urdf_lines.append(2 * TAB + f'<child link="{obj_name}"/>')
        urdf_lines.append(1 * TAB + '</joint>')
    urdf_lines.append('</robot>')

    # add linesep and write to file
    urdf_lines = [line + "\n" for line in urdf_lines]
    with open(out_fp, "w") as urdf_file:
        urdf_file.writelines(urdf_lines)
