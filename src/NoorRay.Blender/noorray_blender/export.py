import bpy
from bpy_extras.io_utils import ExportHelper

from . import noorray


class NOORRAY_OT_export_scene(bpy.types.Operator, ExportHelper):
    bl_idname = "noorray.export_scene"
    bl_label = "Export NoorRay Scene"
    bl_description = "Write a NoorRay scene that the NoorRay app opens"
    bl_options = {"PRESET"}

    filename_ext = ".nrs"
    filter_glob: bpy.props.StringProperty(default="*.nrs", options={"HIDDEN"})
    strict: bpy.props.BoolProperty(name="Strict Materials", description="Fail on unsupported shader nodes instead of writing the closest fixed material",
                                   default=False)

    frames: bpy.props.EnumProperty(name="Frames", default="CURRENT",
                                   items=(("CURRENT", "Current Frame", "Write the current frame as a still scene"),
                                          ("SCENE", "Scene Frame Range", "Write the animation from the start to the end frame of the scene")))

    #! The export reads the viewport depsgraph, so modifiers and visibility follow their viewport settings.
    def execute(self, context):
        scene = context.scene
        frames = (scene.frame_start, scene.frame_end) if self.frames == "SCENE" else None
        try:
            warnings = noorray.export_scene(context.evaluated_depsgraph_get(), self.filepath, approximate=not self.strict, frames=frames)
        except RuntimeError as error:
            self.report({"ERROR"}, f"NoorRay: {error}")
            return {"CANCELLED"}
        for warning in warnings:
            self.report({"WARNING"}, f"NoorRay: {warning}")
        self.report({"INFO"}, f"NoorRay: wrote {self.filepath}")
        return {"FINISHED"}


class NOORRAY_PT_export(bpy.types.Panel):
    bl_label = "Export"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "NoorRay"

    def draw(self, context):
        self.layout.operator(NOORRAY_OT_export_scene.bl_idname, icon="EXPORT")


def draw_export_menu(self, context):
    self.layout.operator(NOORRAY_OT_export_scene.bl_idname, text="NoorRay Scene (.nrs)")


classes = (NOORRAY_OT_export_scene, NOORRAY_PT_export)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_export.append(draw_export_menu)


def unregister():
    bpy.types.TOPBAR_MT_file_export.remove(draw_export_menu)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
