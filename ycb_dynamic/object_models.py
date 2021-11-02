import stillleben as sl
import torch
import torch.nn.functional as F
from math import ceil
from typing import List
from matplotlib import pyplot as plt

import ycb_dynamic.utils.utils as utils
import ycb_dynamic.CONSTANTS as CONSTANTS
from ycb_dynamic.CONFIG import CONFIG
import ycb_dynamic.OBJECT_INFO as OBJECT_INFO


class MeshLoader:
    """
    Class to load the meshes for the objects in a scene
    """

    def __init__(self):
        """Module initializer"""
        self.base_dir = CONSTANTS.MESH_BASE_DIR
        self.reset()

    def reset(self):
        self.class_idx = 0
        self.loaded_meshes = []

    def get_meshes(self):
        """ """
        extract_singular = lambda x: x[0] if len(x) == 1 else x
        return [extract_singular(item) for item in self.loaded_meshes]

    def load_meshes(self, obj_info : List[OBJECT_INFO.ObjectInfo], **kwargs):
        """
        Loads the meshes whose information is given in parameter 'obj_info.
        Each call of this method APPENDS a list to the loaded_meshes attribute.
        :param obj_info: The object information of the meshes to be loaded.
        :param kwargs: additional mesh modifiers such as scale, specified with a leading 'mod_'
        """
        paths = [(self.base_dir / obj.mesh_fp).resolve() for obj in obj_info]
        scales = [obj.scale for obj in obj_info]
        mod_scales = kwargs.get("mod_scale", [1.0] * len(scales))
        scales = [s * ms for (s, ms) in zip(scales, mod_scales)]
        flags = [mesh_flags(obj) for obj in obj_info]
        meshes = sl.Mesh.load_threaded(filenames=paths, flags=flags)

        # Setup class IDs
        for _, (mesh, scale) in enumerate(zip(meshes, scales)):
            pt = torch.eye(4)
            pt[:3, :3] *= scale
            mesh.pretransform = pt
            mesh.class_index = self.class_idx + 1
            self.class_idx += 1

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
        decorations = CONFIG["decorator"]["decorations"]
        self.bounds = CONFIG["decorator"]["bounds"]
        self.pi = torch.acos(torch.zeros(1))

        self.scene = scene
        self.mesh_loader = MeshLoader()
        self.mesh_loader.load_meshes(decorations),
        self.meshes = self.mesh_loader.get_meshes()[0]

        self.x_vect = torch.arange(
                self.bounds["min_x"], self.bounds["max_x"] + self.bounds["res"], self.bounds["res"]
            )
        self.y_vect = torch.arange(
                self.bounds["min_y"], self.bounds["max_y"] + self.bounds["res"], self.bounds["res"]
            )
        self.grid_y, self.grid_x = torch.meshgrid(self.x_vect, self.y_vect)
        self.occupancy_matrix = torch.zeros(
                int((self.bounds["max_x"] + self.bounds["res"] - self.bounds["min_x"]) / self.bounds["res"]),
                int((self.bounds["max_y"] + self.bounds["res"] - self.bounds["min_y"]) / self.bounds["res"])
            )

        n_cells = int(self.bounds["dist"] / self.bounds["res"]) + 1
        self.margin_kernel = torch.ones(1, 1, n_cells, n_cells) / (n_cells ** 2)
        self.pad = (n_cells // 2, n_cells // 2, n_cells // 2, n_cells // 2)

        self.table_height = None
        return

    def init_occupancy_matrix(self, objects):
        """ Obtaining an occupancy matrix with empty and occupied positions"""
        occ_matrix = self.occupancy_matrix.clone()
        for _, obj in objects.items():
            if("table" in obj.mesh.filename):
                self.table_height = obj.mesh.bbox.max[-1]
            occ_matrix = self.update_occupancy_matrix(occ_matrix, obj)
        occ_matrix = self.add_object_margings(occ_matrix)
        return occ_matrix

    def update_occupancy_matrix(self, occ_matrix, obj):
        """ Updating occupancy matrix given object """
        pos_x, pos_y = obj.pose()[:2, -1]
        min_x, min_y = obj.mesh.bbox.min[0] + pos_x, obj.mesh.bbox.min[1] + pos_y
        max_x, max_y = obj.mesh.bbox.max[0] + pos_x, obj.mesh.bbox.max[1] + pos_y
        y_coords = (self.grid_y >= min_y) & (self.grid_y < max_y)
        x_coords = (self.grid_x >= min_x) & (self.grid_x < max_x)
        occ_coords = y_coords & x_coords
        occ_matrix[occ_coords] = 1

        return occ_matrix

    def add_object_margings(self, occ_matrix):
        """ Adding margin to objects in occupancy matrix. Indicated with value 0.5"""
        orig_pos = occ_matrix > 0.5
        occ_matrix[occ_matrix <= 0.5] = 0
        occ_matrix = F.pad(occ_matrix, self.pad).unsqueeze(0).unsqueeze(0)
        occ_matrix = F.conv2d(occ_matrix, self.margin_kernel, stride=1)[0, 0]
        occ_matrix[occ_matrix > 0] = 0.5
        occ_matrix[orig_pos] = 1
        return occ_matrix

    def find_free_spot(self, obj, occ_matrix):
        """ Finding a position in the occupancy matrix where the object wont collide with anything """
        position = None
        max = ceil((obj.mesh.bbox.max[:2].max().item() + self.bounds["dist"]) / self.bounds["res"] + 1)
        max = max if max % 2 != 0 else max + 1
        aux_matrix = F.conv2d(
                occ_matrix.unsqueeze(0).unsqueeze(0),
                torch.ones(1, 1, max, max),
                padding=max//2
            )[0, 0]
        free_positions = torch.where(aux_matrix == 0)
        if(len(free_positions[0]) > 0):
            id = torch.randint(0, len(free_positions[0]), (1,))
            pos_y, pos_x = free_positions[0][id], free_positions[1][id]
            position = torch.cat([self.x_vect[pos_x], self.y_vect[pos_y]])

        return position

    def add_object(self, object_loader, occ_matrix, object_id):
        """ Loading an object and adding to the loader """
        obj_info, obj_mesh = self.meshes[object_id]
        pose = torch.eye(4)
        obj_mod = {"mod_pose": pose}
        obj = object_loader.create_object(obj_info, obj_mesh, True, **obj_mod)
        self.scene.add_object(obj)

        # shifting object to a free position and adjusting z-coord to be aligned with the table
        position = self.find_free_spot(obj=obj, occ_matrix=occ_matrix)
        pose[:2, -1] = position if position is not None else torch.ones(2, 1)
        pose[2, -1] += obj.mesh.bbox.max[-1] - self.table_height

        # Rotating object in yaw direction
        yaw_angle = (torch.rand((1,)) - 0.5) * self.pi  # [-pi / 2, pi / 2]
        angles = torch.cat([yaw_angle, torch.Tensor([0.]), torch.Tensor([0.])])
        rot_matrix = utils.get_rot_matrix(angles=angles)
        pose[:3, :3] = pose[:3, :3] @ rot_matrix

        obj.set_pose(pose)

        occ_matrix = self.update_occupancy_matrix(occ_matrix, obj)
        occ_matrix = self.add_object_margings(occ_matrix)

        return obj, occ_matrix

    def decorate_scene(self, object_loader):
        """ Randomly adding some decoderation to a scene """
        objects = object_loader.loaded_objects
        occ_matrix = self.init_occupancy_matrix(objects)

        N = torch.randint(low=3, high=7, size=(1,))
        for i in range(N):
            id = torch.randint(low=0, high=len(self.meshes), size=(1,))
            obj, occ_matrix = self.add_object(object_loader, occ_matrix, object_id=id)

        # For displaying the occupancy matrix after filling the room
        # plt.figure()
        # plt.imshow(occ_matrix)
        # plt.title(f"Occupacy Matrix after Decoration #{i+1}")
        # plt.colorbar()
        # plt.show()

        return


#
