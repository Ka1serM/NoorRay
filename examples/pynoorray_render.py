"""Minimal Python equivalent of NoorRay's CLI scene setup and render path."""

import numpy as np

import pynoorray


session = pynoorray.RenderSession(width=1280, height=720)
session.import_file("assets/slanted_edge_target.glb")
session.add_perspective_camera(
    position=np.array([0.0, 2.0, 5.0], dtype=np.float32),
    focal_length_mm=50.0,
)
session.set_samples(4)
session.set_max_bounces(10)
session.set_exposure(0.0)

rgba = session.render(spp=64)
print(rgba.shape, rgba.dtype, rgba.min(), rgba.max())
