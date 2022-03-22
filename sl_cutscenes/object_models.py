import stillleben as sl
import random
import torch
from typing import List

from sl_cutscenes.objects.occupancy_matrix import OccupancyMatrix
import sl_cutscenes.utils.utils as utils
import sl_cutscenes.CONSTANTS as CONSTANTS
from sl_cutscenes.CONFIG import CONFIG
import sl_cutscenes.OBJECT_INFO as OBJECT_INFO


class MeshLoader:
    """
    Class to load the meshes for the objects in a scene
    """

    def __init__(self):
        """Module initializer"""
        self.base_dir = CONSTANTS.MESH_BASE_DIR
        self.text_dir = CONSTANTS.TEXT_BASE_DIR
        self.reset()

    def reset(self):
        self.loaded_meshes = []

    def get_meshes(self):
        """ """
        extract_singular = lambda x: x[0] if len(x) == 1 else x
        return [extract_singular(item) for item in self.loaded_meshes]

    def load_meshes(self, obj_info: List[OBJECT_INFO.ObjectInfo], **kwargs):
        """
        Loads the meshes whose information is given in parameter 'obj_info.
        Each call of this method APPENDS a list to the loaded_meshes attribute.
        :param obj_info: The object information of the meshes to be loaded.
        :param kwargs: additional mesh modifiers such as scale, specified with a leading 'mod_'
        """
        paths = []
        for obj in obj_info:
            path = self.text_dir if obj.name.endswith("_floor") or obj.name.endswith("_wall") else self.base_dir
            paths.append((path / obj.mesh_fp).resolve())
        scales = [obj.scale for obj in obj_info]
        class_ids = [obj.class_id for obj in obj_info]
        mod_scales = kwargs.get("mod_scale", [1.0] * len(scales))
        scales = [s * ms for (s, ms) in zip(scales, mod_scales)]
        flags = [mesh_flags(obj) for obj in obj_info]
        meshes = sl.Mesh.load_threaded(filenames=paths, flags=flags)

        # Setup class IDs
        for _, (mesh, scale, class_id) in enumerate(zip(meshes, scales, class_ids)):
            pt = torch.eye(4)
            pt[:3, :3] *= scale
            mesh.pretransform = pt
            mesh.class_index = class_id

        info_mesh_tuples = list(zip(obj_info, meshes))
        self.loaded_meshes.append(info_mesh_tuples)


def mesh_flags(info: OBJECT_INFO.ObjectInfo):
    if info.flags >= OBJECT_INFO.FLAG_CONCAVE:
        return sl.Mesh.Flag.NONE
    else:
        return sl.Mesh.Flag.PHYSICS_FORCE_CONVEX_HULL


class ObjectLoader:
    """
    Class to load the objects in a scene
    """

    def __init__(self):
        """Module initializer"""
        self.reset()

    def reset(self):
        self.instance_idx = 0
        self.loaded_objects = dict()

    @property
    def static_objects(self):
        return [obj for obj in self.loaded_objects.values() if obj.static]

    @property
    def dynamic_objects(self):
        return [obj for obj in self.loaded_objects.values() if not obj.static]

    def create_object(self, object_info: OBJECT_INFO.ObjectInfo, mesh: sl.Mesh, is_static: bool, **obj_mod):
        """
        Proper object setup
        :param mesh:
        :param object_info:
        :param is_static:
        :param obj_mod: Optional object modifiers, specified with a leading 'mod_'.
            IMPORTANT: scaling is done during mesh loading!!!
        :return:
        """
        ins_idx = self.instance_idx + 1
        self.instance_idx += 1
        obj = sl.Object(mesh)
        mod_weight = obj_mod.get("mod_weight", obj_mod.get("mod_scale", 1.0) ** 3)
        obj.mass = object_info.weight * mod_weight
        obj.metallic = object_info.metallic
        obj.roughness = object_info.roughness
        obj.restitution = object_info.restitution
        obj.static_friction = object_info.static_friction
        obj.dynamic_friction = object_info.dynamic_friction
        pose = obj_mod.get("mod_pose", torch.eye(4))
        mod_R = obj_mod.get("mod_R", torch.eye(3))
        pose[:3, :3] = torch.mm(mod_R, pose[:3, :3])
        mod_t = obj_mod.get("mod_t", torch.tensor([obj_mod.get("mod_x", 0.0),
                                                   obj_mod.get("mod_y", 0.0),
                                                   obj_mod.get("mod_z", 0.0)]))
        pose[:3, 3] += mod_t
        obj.set_pose(pose)
        obj.linear_velocity = obj_mod.get("mod_v_linear", torch.tensor([0.0, 0.0, 0.0]))
        obj.angular_velocity = obj_mod.get("mod_v_angular", torch.tensor([0.0, 0.0, 0.0]))
        obj.static = is_static
        obj.instance_index = ins_idx
        self.loaded_objects[ins_idx] = obj

        return obj

    def remove_object(self, instance_id, decrement_ins_idx=True):
        obj = self.loaded_objects.pop(instance_id, None)
        if decrement_ins_idx and obj is not None:
            self.instance_idx -= 1
        return obj


class DecoratorLoader:
    """
    Class to add random decorative objects to the scene, which do not participate of the scene dynamics.
    It is based on creating an occupancy matrix of the scene, finding empty locations and placing stuff there
    """

    def __init__(self, scene):
        """ Object initializer """
        self.config = CONFIG["decorator"]
        decorations = self.config["decorations"]
        bounds = self.config["bounds"]
        self.bounds = bounds
        self.pi = torch.acos(torch.zeros(1))

        self.scene = scene
        self.mesh_loader = MeshLoader()
        self.mesh_loader.load_meshes(decorations),
        self.meshes = self.mesh_loader.get_meshes()[0]

        self.x_vect = torch.arange(bounds["min_x"], bounds["max_x"] + bounds["res"], bounds["res"])
        self.y_vect = torch.arange(bounds["min_y"], bounds["max_y"] + bounds["res"], bounds["res"])

        return

    def add_object(self, object_loader, object_id):
        """ Loading an object and adding to the loader """
        obj_info, obj_mesh = self.meshes[object_id]
        pose = torch.eye(4)
        obj_mod = {"mod_pose": pose}
        obj = object_loader.create_object(obj_info, obj_mesh, True, **obj_mod)
        self.scene.add_object(obj)

        # shifting object to a free position and adjusting z-coord to be aligned with the table
        position = self.occ_matrix.find_free_spot(obj=obj)
        pose[:2, -1] = position if position is not None else torch.ones(2)
        pose[2, -1] += obj.mesh.bbox.max[-1]

        # Rotating object in yaw direction
        yaw_angle = random.choice([torch.tensor([i * CONSTANTS.PI / 2]) for i in range(4)])
        angles = torch.cat([yaw_angle, torch.zeros(2)])
        rot_matrix = utils.get_rot_matrix(angles=angles)
        pose[:3, :3] = pose[:3, :3] @ rot_matrix

        obj.set_pose(pose)
        self.occ_matrix.update_occupancy_matrix(obj)
        self.occ_matrix.add_object_margings()
        return

    def decorate_scene(self, object_loader):
        """ Randomly adding some decoderation to a scene """
        # initializing occupancy matrix
        self.occ_matrix = OccupancyMatrix(bounds=self.bounds, objects=self.scene.objects)

        # iteratively placing objects while avoiding collision
        N = torch.randint(low=self.config["min_objs"], high=self.config["max_objs"], size=(1,))
        for i in range(N):
            id = torch.randint(low=0, high=len(self.meshes), size=(1,))
            self.add_object(object_loader, object_id=id)

        return

#
