from __future__ import annotations

from . import RenderSession


class GaussianTrainer:
    def __init__(self, scene_path: str, width: int, height: int, spp: int = 1):
        self.renderer = RenderSession(width=width, height=height)
        self.renderer.load_scene(scene_path)
        self.spp = spp

    def render(self):
        return self.renderer.render(self.spp)

    def set_opacity(self, index: int, opacity: float) -> None:
        self.renderer.set_gaussian_opacity(index, opacity)
