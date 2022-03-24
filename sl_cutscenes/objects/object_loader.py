import stillleben as sl
import torch
import threading

from sl_cutscenes import object_info


class ObjectLoader:
    """
    Class to load the objects in a scene
    """
    # These 'global' variables determine the loaded objects from ALL instantiated object loaders.
    # They can be re-set by the scenario so that multiple object loaders don't lead to duplicates
    # IMPORTANT: update them only inside a 'with ObjectLoader.lock:' statement!
    scenario_objects_loaded = 0
    loaded_objects = dict()
    lock = threading.Lock()

    def __init__(self, scenario_reset=False):
        """Module initializer"""
        self.reset(scenario_reset)

    def reset(self, new_scenario):
        if new_scenario:
            with ObjectLoader.lock:
                ObjectLoader.scenario_objects_loaded = 0
                ObjectLoader.loaded_objects = dict()

    @property
    def all_objects(self):
        with ObjectLoader.lock:
            objs = ObjectLoader.loaded_objects.values()
        return objs

    @property
    def static_objects(self):
        with ObjectLoader.lock:
            static_objs = [obj for obj in ObjectLoader.loaded_objects.values() if obj.static]
        return static_objs

    @property
    def dynamic_objects(self):
        with ObjectLoader.lock:
            dynamic_objs = [obj for obj in ObjectLoader.loaded_objects.values() if not obj.static]
        return dynamic_objs

    def create_object(self, object_info: object_info.ObjectInfo, mesh: sl.Mesh, is_static: bool, **obj_mod):
        """
        Proper object setup
        :param obj_mod: Optional object modifiers, specified with a leading 'mod_'.
            IMPORTANT: scaling is done during mesh loading!!!
        :return:
        """
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

        with ObjectLoader.lock:
            ObjectLoader.scenario_objects_loaded += 1
            ins_idx = ObjectLoader.scenario_objects_loaded
            obj.instance_index = ins_idx
            ObjectLoader.loaded_objects[ins_idx] = obj
        return obj

    def remove_object(self, instance_id, decrement_ins_idx=True):
        with ObjectLoader.lock:
            obj = ObjectLoader.loaded_objects.pop(instance_id, None)
            if decrement_ins_idx and obj is not None:
                ObjectLoader.scenario_objects_loaded -= 1
        return obj