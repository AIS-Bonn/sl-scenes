"""
Utils methods
"""

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

def get_rpy_from_mat(mat : torch.Tensor):
    """Get roll-pitch-yaw angle from rotation matrix"""
    # TODO test this!
    yaw = torch.atan2(mat[1, 0], mat[0, 0])
    pitch = torch.atan2(-mat[2, 0], torch.sqrt(mat[2, 1] ** 2 + mat[2, 2] ** 2))
    roll = torch.atan2(mat[2, 1], mat[2, 2])
    return roll, pitch, yaw

def get_mat_from_rpy(rpy : torch.Tensor):
    """Get rotation matrix from roll-pitch-yaw angle"""
    raise NotImplementedError

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


def sl_object_to_nimble(obj : sl.Object, obj_info : OBJECT_INFO):

    skel = nimble.dynamics.Skeleton()
    skel.setName(str(obj.instance_index))
    if obj.static:
        skel_joint, skel_body = skel.createWeldJointAndBodyNodePair()
    else:
        skel_joint, skel_body = skel.createFreeJointAndBodyNodePair()
    if obj_info.flags % 1 == 1:  # object is concave -> initialize with sub-parts
        convex_parts = getattr(obj.mesh, "convex_parts", [None])
        raise NotImplementedError  # TODO deal with non-convex parts
    else:
        scale = obj_info.scale
        skel_shape = skel_body.createShapeNode(nimble.dynamics.MeshShape(scale=torch.tensor([scale] * 3),
                                                                       path=obj_info.mesh_fp))
        inertia_moment = skel_shape.getShape().computeInertia(obj.mass)
        skel_body.setInertia(nimble.dynamics.Inertia(obj.mass, obj.inertial_frame[:3, 3], inertia_moment))

    pose = obj.pose()
    t = pose[:3, 3]
    rpy = get_rpy_from_mat(pose[:3, :3])
    skel_state = torch.cat([t, rpy, obj.linear_velocity, obj.angular_velocity]).cpu()
    # TODO do the angular velocities need to be converted as well?
    skel.setState(skel_state)
    return skel


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