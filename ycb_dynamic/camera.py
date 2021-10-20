import torch


class Camera(object):
    def __init__(self, name: str, start_pos: torch.Tensor, start_lookat: torch.Tensor, moving: bool):
        self.name = name  # can be used e.g. to name the corresponding output directories
        self.start_pos = start_pos
        self.start_lookat = start_lookat
        self.moving = moving
        self.reset_cam()


    def reset_cam(self):
        self.t = 0
        self.pos = self.start_pos
        self.lookat = self.start_lookat


    def step(self):
        if not self.moving:
            return
        else:
            raise NotImplementedError("TODO implement position/lookat change on step() invocation")