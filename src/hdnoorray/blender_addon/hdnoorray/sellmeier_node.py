# SPDX-License-Identifier: GPL-3.0-or-later
"""Blender UI node for NoorRay's spectral dielectric extension."""

import bpy
from bpy.types import ShaderNode


class NoorRaySellmeierIOR(ShaderNode):
    """A scalar Blender IOR facade carrying a Sellmeier model."""

    bl_idname = "NoorRaySellmeierIOR"
    bl_label = "Sellmeier IOR"
    bl_description = (
        "Spectral IOR model. The output is the scalar IOR used by Blender; "
        "NoorRay exports all Sellmeier coefficients."
    )
    bl_icon = "NODE_MATERIAL"
    bl_width_default = 190
    _REFERENCE_WAVELENGTH_NM = 546.074

    _INPUTS = (
        ("B1", 1.03961212),
        ("B2", 0.231792344),
        ("B3", 1.01046945),
        ("C1 (um²)", 0.00600069867),
        ("C2 (um²)", 0.0200179144),
        ("C3 (um²)", 103.560653),
    )

    def init(self, _context):
        for name, default in self._INPUTS:
            socket = self.inputs.new("NodeSocketFloat", name)
            socket.default_value = default
        output = self.outputs.new("NodeSocketFloat", "IOR")
        output.description = "IOR at 546.074 nm for Blender's scalar IOR socket"
        self._update_scalar_ior()

    def update(self):
        self._update_scalar_ior()

    def _update_scalar_ior(self):
        if not self.outputs or not self.inputs:
            return
        try:
            values = [float(self.inputs[name].default_value) for name, _ in self._INPUTS]
            wavelength_um = self._REFERENCE_WAVELENGTH_NM * 0.001
            lambda_squared = wavelength_um * wavelength_um
            n_squared = 1.0
            for index in range(3):
                denominator = lambda_squared - values[index + 3]
                if abs(denominator) > 1.0e-8:
                    n_squared += values[index] * lambda_squared / denominator
            self.outputs["IOR"].default_value = max(n_squared, 1.0) ** 0.5
        except (AttributeError, IndexError, TypeError, ValueError):
            pass

    def draw_buttons(self, _context, layout):
        layout.label(text="n(546.074 nm) derived")


CLASSES = (NoorRaySellmeierIOR,)
