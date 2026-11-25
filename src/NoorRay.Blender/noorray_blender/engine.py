import math
import time

import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader
from gpu_extras.presets import draw_texture_2d

from . import noorray
from .properties import cycles_settings

#! Updates of these data-blocks change nothing that the session holds.
IGNORED_UPDATES = (bpy.types.Screen, bpy.types.WorkSpace)
SHUTTER_CURVE_SAMPLES = 256
FINAL_UPDATE_INTERVAL = 1.0
MAX_REPORTED_WARNINGS = 8
#! The weight of the newest frame time in the viewport's frame rate.
FRAME_RATE_SMOOTHING = 0.1
#! Blender is Z-up and NoorRay is Y-up.
Y_UP = np.array(((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, -1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)))


def rows(matrix):
    return tuple(float(value) for value in matrix[:3].reshape(-1))


#! The border as pixel columns and rows, left to right and bottom to top; at least one pixel.
def border_pixels(border, width, height):
    left, right, bottom, top = border
    x = min(max(int(left * width), 0), width - 1)
    y = min(max(int(bottom * height), 0), height - 1)
    return x, y, max(min(int(right * width), width), x + 1), max(min(int(top * height), height), y + 1)


#! The part of the scene camera's image that the view covers: left, top, right, bottom.
def camera_window(projection, scene):
    corners = np.array([tuple(corner) + (1.0,) for corner in scene.camera.data.view_frame(scene=scene)], dtype=np.float64)
    clip = corners @ projection.T
    frame = clip[:, :2] / clip[:, 3:]
    (left, bottom), (right, top) = frame.min(axis=0), frame.max(axis=0)
    return tuple(float(value) for value in ((-1.0 - left) / (right - left), (top - 1.0) / (top - bottom), (1.0 - left) / (right - left),
                                            (top + 1.0) / (top - bottom)))


#! Reads the camera out of the projection matrix, so that the zoom and offset of a camera view carry over.
def view_camera(region_data, scene, projection):
    camera_to_world = Y_UP @ np.array(region_data.view_matrix.inverted(), dtype=np.float64)
    view = {}
    if region_data.is_perspective:
        view["vertical_fov"] = 2.0 * math.atan(1.0 / projection[1][1])
        view["shift_x"], view["shift_y"] = projection[0][2], projection[1][2]
        view["near"] = projection[2][3] / (projection[2][2] - 1.0)
        view["far"] = projection[2][3] / (projection[2][2] + 1.0)
    else:
        view["orthographic"] = True
        view["orthographic_height"] = 2.0 / projection[1][1]
        view["shift_x"], view["shift_y"] = -projection[0][3], -projection[1][3]
        near = (projection[2][3] + 1.0) / projection[2][2]
        far = (projection[2][3] - 1.0) / projection[2][2]
        #! An orthographic view clips from a negative near plane, but NoorRay starts its rays at the camera.
        camera_to_world = camera_to_world @ np.array(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, -near), (0, 0, 0, 1)), dtype=np.float64)
        view["near"], view["far"] = 0.0, far - near
    camera = scene.camera
    if region_data.view_perspective == "CAMERA" and camera is not None and camera.type == "CAMERA":
        view["through_camera"] = True
        if camera.data.type == "PANO":
            view["window"] = camera_window(projection, scene)
    view["transform"] = rows(camera_to_world)
    return view


#! A selection change updates objects without a change of their transform, geometry or shading.
def changes_scene(update):
    if isinstance(update.id, IGNORED_UPDATES):
        return False
    if not isinstance(update.id, bpy.types.Object):
        return True
    return update.is_updated_transform or update.is_updated_geometry or update.is_updated_shading


#! The film rows run top down and Blender's bottom up; foreach_set copies in milliseconds where an assignment of rect takes seconds.
def show(combined, pixels):
    combined.rect.foreach_set(pixels[::-1].ravel())


#! Cycles' mist falls from 1 in the distance to 0 at the camera.
def mist(scene, depth):
    settings = scene.world.mist_settings if scene.world is not None else None
    start, length, falloff = (settings.start, settings.depth, settings.falloff) if settings is not None else (5.0, 25.0, "QUADRATIC")
    value = np.clip((depth - start) / max(length, 1e-6), 0.0, 1.0)
    if falloff == "LINEAR":
        return value
    return value * value if falloff == "QUADRATIC" else np.sqrt(value)


def write_passes(scene, session, passes):
    shape = (session.height, session.width)
    for name in ("Normal", "Diffuse Color"):
        if name in passes:
            values = np.empty((*shape, 3), np.float32)
            session.read_pass(name, values)
            show(passes[name], values)
    if "Depth" not in passes and "Mist" not in passes:
        return
    depth = np.empty((*shape, 1), np.float32)
    session.read_pass("Depth", depth)
    if "Depth" in passes:
        show(passes["Depth"], depth)
    if "Mist" in passes:
        show(passes["Mist"], np.ascontiguousarray(mist(scene, depth)))


#! Blender binds the texture of a GPUTexture when it draws with it, and the session finds the OpenGL texture by that binding.
def bind_texture(texture):
    shader = gpu.shader.from_builtin("IMAGE")
    corner = ((0.0, 0.0),) * 3
    batch = batch_for_shader(shader, "TRIS", {"pos": corner, "texCoord": corner})
    shader.uniform_sampler("image", texture)
    batch.draw(shader)


#! The frame offsets that motion blur samples: equal shares of the area of the shutter curve.
def shutter_times(scene):
    render = scene.render
    if not render.use_motion_blur:
        return [0.0]
    mapping = render.motion_blur_shutter_curve
    mapping.initialize()
    positions = (np.arange(SHUTTER_CURVE_SAMPLES) + 0.5) / SHUTTER_CURVE_SAMPLES
    weights = np.array([max(mapping.evaluate(mapping.curves[0], position), 0.0) for position in positions])
    if weights.sum() <= 0.0:
        return [0.0]
    steps = min(scene.noorray.motion_steps, scene.noorray.samples)
    area = np.concatenate(((0.0,), np.cumsum(weights) / weights.sum()))
    opened = np.interp((np.arange(steps) + 0.5) / steps, area, np.linspace(0.0, 1.0, SHUTTER_CURVE_SAMPLES + 1))
    start = {"START": 0.0, "CENTER": -0.5, "END": -1.0}[render.motion_blur_position]
    return list((opened + start) * render.motion_blur_shutter)


def report_warnings(engine, session):
    warnings = list(dict.fromkeys(session.warnings))
    for warning in warnings[:MAX_REPORTED_WARNINGS]:
        engine.report({"WARNING"}, f"NoorRay: {warning}")
    if len(warnings) > MAX_REPORTED_WARNINGS:
        engine.report({"WARNING"}, f"NoorRay: {len(warnings) - MAX_REPORTED_WARNINGS} more warnings")


class NoorRayRenderEngine(bpy.types.RenderEngine):
    bl_idname = "NOORRAY"
    bl_label = "NoorRay"
    bl_use_preview = False
    bl_use_shading_nodes_custom = False
    bl_use_eevee_viewport = True

    def __init__(self, *arguments, **keywords):
        super().__init__(*arguments, **keywords)
        self.session = None
        self.settings = None
        self.view = None
        self.pixels = None
        self.texture = None
        #! Only a drawing OpenGL Blender shares the film on the GPU; after a failure the film goes through the CPU until the region size changes.
        self.shares = not bpy.app.background and gpu.platform.backend_type_get() == "OPENGL"
        self.unshared = None
        self.reported = set()
        self.error = None
        #! False until a sync succeeds; a failed sync clears the scene, so the next update syncs all of it again.
        self.synced = False
        #! The smoothed seconds per viewport frame, and the time of the last rendered frame; None until two frames follow each other.
        self.frame_time = None
        self.rendered = None

    def update_render_passes(self, scene, render_layer):
        self.register_pass(scene, render_layer, "Combined", 4, "RGBA", "COLOR")
        if render_layer.use_pass_z:
            self.register_pass(scene, render_layer, "Depth", 1, "Z", "VALUE")
        if render_layer.use_pass_mist:
            self.register_pass(scene, render_layer, "Mist", 1, "Z", "VALUE")
        if render_layer.use_pass_normal:
            self.register_pass(scene, render_layer, "Normal", 3, "XYZ", "VECTOR")
        if render_layer.use_pass_diffuse_color:
            self.register_pass(scene, render_layer, "Diffuse Color", 3, "RGB", "COLOR")

    def render(self, depsgraph):
        scene = depsgraph.scene
        if scene.camera is None:
            self.report({"ERROR"}, "NoorRay: the scene has no active camera")
            return
        scale = scene.render.resolution_percentage / 100.0
        width = max(int(scene.render.resolution_x * scale), 1)
        height = max(int(scene.render.resolution_y * scale), 1)
        border = None
        if scene.render.use_border:
            border = border_pixels((scene.render.border_min_x, scene.render.border_max_x, scene.render.border_min_y, scene.render.border_max_y), width, height)
        try:
            self.render_film(depsgraph, width, height, border)
        except RuntimeError as error:
            self.report({"ERROR"}, f"NoorRay: {error}")

    def render_film(self, depsgraph, width, height, border):
        scene = depsgraph.scene
        settings = scene.noorray.engine_settings(final=True)
        left, bottom, right, top = border or (0, 0, width, height)
        #! With Persistent Data Blender keeps this engine for the next frame, whose sync reads only what changed.
        session = self.session
        if session is None:
            session = noorray.Engine(right - left, top - bottom)
        elif border is None:
            border = (0, 0, width, height)
        session.resize(right - left, top - bottom)
        self.session = session if scene.render.use_persistent_data else None
        session.configure(**settings)
        times = shutter_times(scene)
        frame = scene.frame_current_final
        if len(times) > 1:
            self.set_time(frame + times[0])
        session.sync(depsgraph, approximate=True)
        if border is not None:
            session.crop(left / width, 1.0 - top / height, right / width, 1.0 - bottom / height)
        report_warnings(self, session)
        pixels = np.empty((top - bottom, right - left, 4), np.float32)
        result = self.begin_result(0, 0, right - left, top - bottom, layer=depsgraph.view_layer.name)
        finished = False
        #! Blender crashes on a result left open.
        try:
            passes = result.layers[0].passes
            limit = cycles_settings(scene).time_limit
            started = time.monotonic()
            #! With motion blur each time of the shutter interval gets an equal share of the samples and of the time.
            for step, offset in enumerate(times, 1):
                if step > 1:
                    if self.test_break():
                        break
                    self.set_time(frame + offset)
                    session.sync(depsgraph, approximate=True, accumulates=True)
                deadline = started + limit * step / len(times) if limit > 0.0 else math.inf
                self.accumulate(session, settings["samples"], -(-settings["samples"] * step // len(times)), deadline, result, pixels)
            #! Like Cycles, a final render denoises once at the end, also when the time limit stopped it.
            if not self.test_break():
                session.denoise()
            session.read(pixels)
            show(passes["Combined"], pixels)
            write_passes(scene, session, passes)
            finished = True
        finally:
            self.end_result(result, cancel=not finished)
            if len(times) > 1:
                self.set_time(frame)

    def set_time(self, moment):
        frame = math.floor(moment)
        self.frame_set(frame, moment - frame)

    #! Cycles' Time Limit is in seconds, and 0 has none.
    def accumulate(self, session, samples, target, deadline, result, pixels):
        combined = result.layers[0].passes["Combined"]
        shown = time.monotonic()
        first = True
        while not self.test_break() and (first or (session.samples < target and not session.complete and time.monotonic() < deadline)):
            first = False
            session.render()
            self.update_stats("", f"Sample {session.samples}/{samples}")
            self.update_progress(session.samples / samples)
            if time.monotonic() - shown < FINAL_UPDATE_INTERVAL:
                continue
            session.read(pixels)
            show(combined, pixels)
            self.update_result(result)
            shown = time.monotonic()

    def view_update(self, context, depsgraph):
        try:
            settings = depsgraph.scene.noorray.engine_settings(final=False)
            if self.session is None:
                self.session = noorray.Engine(context.region.width, context.region.height)
            if settings != self.settings:
                self.session.configure(**settings)
                self.settings = settings
            if self.synced and not any(changes_scene(update) for update in depsgraph.updates):
                return
            self.synced = False
            self.session.sync(depsgraph, approximate=True)
            self.synced = True
            self.error = None
            report_warnings(self, self.session)
        except RuntimeError as error:
            self.error = f"NoorRay: {error}"

    def view_draw(self, context, depsgraph):
        if self.error is not None or self.session is None:
            self.update_stats("", self.error or "")
            return
        region = context.region
        projection = np.array(context.region_data.window_matrix, dtype=np.float64)
        left, bottom, right, top = self.view_border(context, depsgraph.scene, projection)
        width, height = right - left, top - bottom
        #! A film of a part of the view has the part of the projection.
        scale = np.array((region.width / width, region.height / height))
        centre = np.array(((left + right) / region.width - 1.0, (bottom + top) / region.height - 1.0))
        crop = np.diag((scale[0], scale[1], 1.0, 1.0))
        crop[:2, 3] = -centre * scale
        projection = crop @ projection
        try:
            self.session.resize(width, height)
            camera = view_camera(context.region_data, depsgraph.scene, projection)
            if (camera, width, height) != self.view:
                self.session.set_camera(**camera)
                self.view = (camera, width, height)
            texture = self.share(width, height) if self.shares else None
            if not self.session.complete:
                self.session.render()
                self.count_frame()
                self.tag_redraw()
            else:
                self.frame_time = self.rendered = None
            if texture is not None:
                texture = self.present(texture)
            if texture is None:
                texture = self.copy(width, height)
        except RuntimeError as error:
            self.error = f"NoorRay: {error}"
            self.update_stats("", self.error)
            return
        gpu.state.blend_set("ALPHA_PREMULT")
        self.bind_display_space_shader(depsgraph.scene)
        draw_texture_2d(texture, (left, top), width, -height)
        self.unbind_display_space_shader()
        gpu.state.blend_set("NONE")
        #! The viewport draws each line of the stats text under the last.
        stats = f"Sample {self.session.samples}"
        if self.frame_time is not None:
            stats += f"\n{1.0 / self.frame_time:.0f} fps"
        if self.session.compiling:
            stats += f"\n{self.session.compiling}"
        self.update_stats("", stats)

    #! The pixels of the region that the film covers: a render border in a camera view or a render region in the others.
    def view_border(self, context, scene, projection):
        region = context.region
        size = region.width, region.height
        if context.region_data.view_perspective != "CAMERA":
            space = context.space_data
            if not space.use_render_border:
                return 0, 0, *size
            border = space.render_border_min_x, space.render_border_max_x, space.render_border_min_y, space.render_border_max_y
            return border_pixels(border, *size)
        if not scene.render.use_border or scene.camera is None or scene.camera.type != "CAMERA":
            return 0, 0, *size
        left, top, right, bottom = camera_window(projection, scene)
        return border_pixels(((scene.render.border_min_x - left) / (right - left), (scene.render.border_max_x - left) / (right - left),
                              (scene.render.border_min_y - 1.0 + bottom) / (bottom - top), (scene.render.border_max_y - 1.0 + bottom) / (bottom - top)), *size)

    def count_frame(self):
        now = time.monotonic()
        if self.rendered is not None:
            elapsed = now - self.rendered
            self.frame_time = elapsed if self.frame_time is None else self.frame_time + FRAME_RATE_SMOOTHING * (elapsed - self.frame_time)
        self.rendered = now

    def share(self, width, height):
        if self.texture is not None and (self.texture.width, self.texture.height) == (width, height):
            return self.texture
        if self.unshared == (width, height):
            return None
        try:
            self.texture = gpu.types.GPUTexture((width, height), format="RGBA16F")
            bind_texture(self.texture)
            self.session.open_gl()
            return self.texture
        except RuntimeError as error:
            self.stop_sharing(error, width, height)
            return None

    def present(self, texture):
        try:
            self.session.present_gl()
            return texture
        except RuntimeError as error:
            self.stop_sharing(error, texture.width, texture.height)
            return None

    def stop_sharing(self, error, width, height):
        self.unshared = (width, height)
        self.texture = None
        message = f"NoorRay: the viewport copies the film through the CPU at {width}x{height}: {error}"
        if message in self.reported:
            return
        self.reported.add(message)
        print(message)
        self.report({"WARNING"}, message)

    def copy(self, width, height):
        if self.pixels is None or len(self.pixels) != width * height * 4:
            self.pixels = gpu.types.Buffer("FLOAT", width * height * 4)
        self.session.read(self.pixels)
        return gpu.types.GPUTexture((width, height), format="RGBA32F", data=self.pixels)
