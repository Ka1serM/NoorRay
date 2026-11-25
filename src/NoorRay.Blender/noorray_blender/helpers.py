import math
import os
from functools import lru_cache

import bpy
import mathutils
import numpy as np


@lru_cache(maxsize=None)
def color_processor(space):
    import PyOpenColorIO

    path = os.environ.get("OCIO") or os.path.join(bpy.utils.system_resource("DATAFILES"), "colormanagement", "config.ocio")
    return PyOpenColorIO.Config.CreateFromFile(path).getProcessor(space, "scene_linear").getDefaultCPUProcessor()


#! Converts RGBA float texels of a Blender colour space in place.
def to_scene_linear(space, texels):
    color_processor(space).applyRGBA(np.frombuffer(texels, np.float32))


#! The file of a sequence at the frame of an image user.
def frame_path(image, user):
    return image.filepath_from_user(image_user=user)


#! The longest side of the dense grid that a sparse grid reads into, in voxels.
GRID_SIDE = 256


#! The values of a grid, x fastest, with the matrix from its voxel indices to the object; None when the source has no such grid, and a message when the grids do not load.
def volume_grid(source, name):
    if isinstance(source, str):
        return vdb_grid(source, name)
    return fluid_grid(source, name)


#! The OpenVDB file of a volume data-block at its frame; Blender gives no values of a volume that geometry nodes make.
def volume_path(volume):
    if not volume.filepath or not volume.grids.load():
        return ""
    return bpy.path.abspath(volume.grids.frame_filepath or volume.filepath, library=volume.library)


def vdb_grid(path, name):
    try:
        import openvdb
    except ImportError:
        return "this Blender has no openvdb module, so volumes do not render; the builds of blender.org have it"
    if not os.path.isfile(path):
        return f"the volume file '{path}' does not exist" if path else "a volume of geometry nodes has no file and does not render"

    if name not in {metadata.name for metadata in openvdb.readAllGridMetadata(path)}:
        return None
    grid = openvdb.read(path, name)
    low, high = grid.evalActiveVoxelBoundingBox()
    if any(high[axis] < low[axis] for axis in range(3)):
        return None
    size = [high[axis] - low[axis] + 1 for axis in range(3)]
    stride = max(1, math.ceil(max(size) / GRID_SIDE))
    cells = [math.ceil(side / stride) for side in size]
    vector = not isinstance(grid.background, (float, int, bool))
    result = np.empty(cells, np.float32)
    for x in range(cells[0]):
        slab = np.zeros((stride, cells[1] * stride, cells[2] * stride) + ((3,) if vector else ()), np.float32)
        grid.copyToArray(slab, ijk=(low[0] + x * stride, low[1], low[2]))
        if vector:
            slab = slab.mean(axis=3)
        result[x] = slab.reshape(stride, cells[1], stride, cells[2], stride).mean(axis=(0, 2, 4))
    #! A VDB voxel has its centre at its index.
    start = [low[axis] - 0.5 for axis in range(3)]
    origin = mathutils.Vector(grid.transform.indexToWorld(start))
    matrix = mathutils.Matrix.Identity(4)
    matrix.translation = origin
    for axis in range(3):
        end = list(start)
        end[axis] += stride
        matrix.col[axis] = (*(mathutils.Vector(grid.transform.indexToWorld(end)) - origin), 0.0)
    return cells, matrix, result.transpose(2, 1, 0).tobytes()


def fluid_domain(object):
    for modifier in object.modifiers:
        if modifier.type == "FLUID" and modifier.fluid_type == "DOMAIN" and modifier.domain_settings.domain_type == "GAS":
            return modifier.domain_settings
    return None


def fluid_grid(object, name):
    domain = fluid_domain(object)
    if name not in {"density", "flame", "heat", "temperature"}:
        return None
    property = getattr(domain, f"{name}_grid")
    size = tuple(domain.domain_resolution)
    if len(property) == 0 or 0 in size:
        return None
    scale = round((len(property) / math.prod(size)) ** (1.0 / 3.0))
    cells = [side * scale for side in size]
    values = np.empty(len(property), np.float32)
    property.foreach_get(values)
    low, high = object.bound_box[0], object.bound_box[6]
    matrix = mathutils.Matrix.Identity(4)
    for axis in range(3):
        matrix[axis][axis] = (high[axis] - low[axis]) / cells[axis]
        matrix[axis][3] = low[axis]
    return cells, matrix, values.tobytes()


#! The hair strands of the particle systems of an object: for each system the material slot, points per strand, positions in the object, and radii along a strand.
def particle_hair(object, render):
    strands = []
    inverse = object.matrix_world.inverted()
    for modifier in object.modifiers:
        if modifier.type != "PARTICLE_SYSTEM" or not (modifier.show_render if render else modifier.show_viewport):
            continue
        system = modifier.particle_system
        settings = system.settings
        if settings.type != "HAIR" or settings.render_type != "PATH":
            continue
        steps = 2 ** (settings.render_step if render else settings.display_step) + 1
        parents, children = len(system.particles), len(system.child_particles)
        first = 0 if children == 0 or settings.use_parent_particles else parents
        positions = np.array([system.co_hair(object, particle_no=particle, step=step) for particle in range(first, parents + children)
                              for step in range(steps)], np.float32).reshape(-1, 3)
        if len(positions) == 0:
            continue
        positions = (positions @ np.array(inverse.to_3x3(), np.float32).T + np.array(inverse.translation, np.float32)).astype(np.float32)
        along = np.linspace(1.0, 0.0, steps)
        shape = settings.shape
        curve = along if shape == 0.0 else along ** (1.0 + shape if shape < 0.0 else 1.0 / max(1.0 - shape, 1e-3))
        #! The radii of the particle settings are diameters.
        radius = 0.5 * settings.radius_scale * (settings.tip_radius + (settings.root_radius - settings.tip_radius) * curve)
        if settings.use_close_tip:
            radius[-1] = 0.0
        strands.append((max(settings.material - 1, 0), steps, positions.tobytes(), radius.astype(np.float32).tobytes()))
    return strands


#! A value of an attribute as RGBA; a name in brackets is a custom property, and any other name is one or a property path.
def rgba(value):
    if isinstance(value, (bool, int, float)):
        return (float(value),) * 3 + (1.0,)
    try:
        parts = [float(part) for part in value]
    except (TypeError, ValueError):
        return None
    return tuple(parts[:4] + [0.0] * (3 - len(parts)) + [1.0] * (len(parts) < 4)) if 1 < len(parts) else None


def lookup(owner, name):
    if owner is None:
        return None
    if name in owner.keys():
        return rgba(owner[name])
    try:
        return rgba(owner.path_resolve(name))
    except ValueError:
        return None


def object_attribute(object, name):
    return lookup(object, name) or lookup(object.data, name) or (0.0,) * 4


def view_layer_attribute(depsgraph, name):
    layer = depsgraph.view_layer_eval
    return lookup(layer, name) or lookup(depsgraph.scene_eval, name) or lookup(depsgraph.scene_eval.world, name) or (0.0,) * 4


#! The values of an attribute of each instance that geometry nodes make, or None.
def instance_attribute(parent, name):
    points = parent.evaluated_geometry().instances_pointcloud() if hasattr(parent, "evaluated_geometry") else None
    attribute = points.attributes.get(name) if points else None
    if attribute is None:
        return None
    return [rgba(item.color if "COLOR" in attribute.data_type else item.vector if "VECTOR" in attribute.data_type or attribute.data_type == "FLOAT2" else item.value) or (0.0,) * 4
            for item in attribute.data]
