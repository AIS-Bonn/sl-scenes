import stillleben as sl
import torch

import ycb_dynamic.CONSTANTS as CONSTANTS


class MeshLoader:
    """
    Class to load the meshes for the objects in a scene
    """

    def __init__(self):
        """Module initializer"""
        self.base_dir = CONSTANTS.MESH_BASE_DIR
        self.class_idx = 0
        self.loaded_meshes = []
        # store weights separately to pass to object construction with loaded meshes
        self.loaded_mesh_weights = []

    def get_meshes(self):
        """ """
        return self.loaded_meshes

    def get_mesh_weights(self):
        """ """
        return self.loaded_mesh_weights

    def load_meshes(self, objects):
        """
        Loads the meshes corresponding to given namedtuples 'objects'.
        :param objects: The object information of the meshes to be loaded.
        :param class_index_start: If specified, class index values are assigned starting from this number.
        :return: The loaded meshes as a list, or the loaded mesh object itself if it's only one.
        """
        paths = [(self.base_dir / obj.mesh_fp).resolve() for obj in objects]
        scales = [obj.scale for obj in objects]
        weights = [obj.weight for obj in objects]
        flags = [mesh_flags(obj) for obj in objects]
        meshes = sl.Mesh.load_threaded(filenames=paths, flags=flags)

        # Setup class IDs
        for i, (mesh, scale) in enumerate(zip(meshes, scales)):
            pt = torch.eye(4)
            pt[:3, :3] *= scale
            mesh.pretransform = pt
            mesh.class_index = self.class_idx + i + 1
            self.class_idx += 1

        meshes = meshes if len(meshes) != 1 else meshes[0]
        weights = weights if len(weights) != 1 else weights[0]
        self.loaded_meshes.append(meshes)
        self.loaded_mesh_weights.append(weights)
        return


def mesh_flags(info: CONSTANTS.ObjectInfo):
    if info.flags >= CONSTANTS.FLAG_CONCAVE:
        return sl.Mesh.Flag.NONE
    else:
        return sl.Mesh.Flag.PHYSICS_FORCE_CONVEX_HULL


def load_table_and_ycbv():
    """
    Loads the meshes required for the Table scenario:
        YCBV objects falling on top of a table.
    :return: The loaded meshes as a tuple: (table_mesh, ycb_video_meshes)
    """
    meshLoader = MeshLoader()
    meshLoader.load_meshes(CONSTANTS.TABLE),
    meshLoader.load_meshes(CONSTANTS.YCBV_OBJECTS),
    return meshLoader.get_meshes(), meshLoader.get_mesh_weights()


def load_bowling():
    """
    Loads the meshes required for the Bowling Scenario:
        A ball smashes through a tower of wooden blocks
    :return: The loaded meshes as a tuple: (table_mesh, ycb_video_meshes)
    """
    meshLoader = MeshLoader()
    meshLoader.load_meshes(CONSTANTS.TABLE),
    meshLoader.load_meshes(CONSTANTS.BOWLING_BALL),
    meshLoader.load_meshes(CONSTANTS.WOOD_BLOCK),
    return meshLoader.get_meshes(), meshLoader.get_mesh_weights()


def load_billiards():
    """
    Loads the meshes required for the Billiards Scenario:
        A ball smashes through a bunch of objects placed in a pool-ball triangle manner
    :return: The loaded meshes as a tuple: (table_mesh, ycb_video_meshes)
    """
    meshLoader = MeshLoader()
    meshLoader.load_meshes(CONSTANTS.TABLE),
    meshLoader.load_meshes(CONSTANTS.BOWLING_BALL),
    meshLoader.load_meshes(CONSTANTS.BILLIARDS_OBJECTS),
    return meshLoader.get_meshes(), meshLoader.get_mesh_weights()


#
