import stillleben as sl


class Scenario(object):
    def __init__(self, name, scene, cameras, static_objects, dynamic_objects):
        self.name = name
        self.scene = scene
        self.cameras = cameras
        self.static_objects = static_objects
        self.dynamic_objects = dynamic_objects

    def simulation_step(self, dt):
        self.scene.simulate(dt)


def add_obj_to_scene(scene: sl.Scene, obj: sl.Object):
    obj.instance_index = len(scene.objects) + 1
    scene.add_object(obj)
