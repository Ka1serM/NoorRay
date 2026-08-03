# SPDX-License-Identifier: GPL-3.0-or-later
"""Incremental MaterialX transport for Blender's Hydra render engine."""

from dataclasses import dataclass
from array import array
import ctypes
from pathlib import Path
from xml.sax.saxutils import escape as _xml_escape

import bpy
from bpy.app.handlers import persistent

from .materialx_export import export_material


SETTING_PREFIX = "noorray:materialx:"


class _MemoryTextureApi:
    """Thin ABI shim for the already-loaded hdNoorRay plugin."""

    def __init__(self) -> None:
        library = ctypes.CDLL(
            str(Path(__file__).resolve().parent / "plugin" / "libhdnoorray.so")
        )
        self._register = library.HdNoorRayRegisterMemoryTextureF32
        self._register.argtypes = (
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
        )
        self._register.restype = ctypes.c_int
        self._unregister = library.HdNoorRayUnregisterMemoryTexture
        self._unregister.argtypes = (ctypes.c_char_p,)
        self._unregister.restype = None

    def register(self, uri: str, pixels: array, width: int, height: int) -> bool:
        data = (ctypes.c_float * len(pixels)).from_buffer(pixels)
        return bool(self._register(uri.encode("utf-8"), data, len(pixels), width, height))

    def unregister(self, uri: str) -> None:
        self._unregister(uri.encode("utf-8"))

_revision = 0
_full_revision = 0
_material_revisions: dict[int, int] = {}
_image_revisions: dict[int, int] = {}


def hydra_material_leaf(material) -> str:
    """Return the leaf Blender 5.2 uses in PopulateContext::material_prim_id."""
    return f"M_0x{material.as_pointer():x}"


def _original(datablock):
    return getattr(datablock, "original", None) or datablock


def _next_revision() -> int:
    global _revision
    _revision += 1
    return _revision


def invalidate_all() -> None:
    global _full_revision
    _full_revision = _next_revision()
    _material_revisions.clear()
    _image_revisions.clear()


@persistent
def _depsgraph_update_post(_scene, depsgraph) -> None:
    changes = tuple(depsgraph.updates)
    if not changes:
        return
    revision = _next_revision()
    for update in changes:
        datablock = _original(update.id)
        pointer = datablock.as_pointer()
        if isinstance(datablock, bpy.types.Material):
            _material_revisions[pointer] = revision
        elif isinstance(datablock, bpy.types.Image):
            _image_revisions[pointer] = revision
        elif isinstance(datablock, bpy.types.NodeTree):
            # A shared shader group can feed any number of materials.  Groups
            # are not expanded by the lean exporter yet, but a broad
            # invalidation here makes adding that support safe.
            global _full_revision
            _full_revision = revision


@persistent
def _reset_after_file_operation(_dummy) -> None:
    invalidate_all()


_RESET_HANDLERS = (
    bpy.app.handlers.load_post,
    bpy.app.handlers.undo_post,
    bpy.app.handlers.redo_post,
)


def register_handlers() -> None:
    if _depsgraph_update_post not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_depsgraph_update_post)
    for handlers in _RESET_HANDLERS:
        if _reset_after_file_operation not in handlers:
            handlers.append(_reset_after_file_operation)


def unregister_handlers() -> None:
    if _depsgraph_update_post in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_depsgraph_update_post)
    for handlers in _RESET_HANDLERS:
        if _reset_after_file_operation in handlers:
            handlers.remove(_reset_after_file_operation)
    invalidate_all()


@dataclass(slots=True)
class _Content:
    document: str
    image_revisions: dict[int, int]
    material_revision: int
    full_revision: int
    frame: float | None


class MaterialXRenderSettingsSync:
    """Per-Hydra-engine delivery state with shared depsgraph invalidation."""

    __slots__ = ("_content", "_sent", "_image_paths", "_memory_texture_api")

    def __init__(self):
        self._content: dict[tuple[int, int], _Content] = {}
        self._sent: dict[str, str] = {}
        self._image_paths: dict[int, tuple[int, str]] = {}
        self._memory_texture_api: _MemoryTextureApi | None = None

    def _memory_textures(self) -> _MemoryTextureApi:
        if self._memory_texture_api is None:
            self._memory_texture_api = _MemoryTextureApi()
        return self._memory_texture_api

    @staticmethod
    def _identity(material) -> tuple[int, int]:
        original = _original(material)
        return original.as_pointer(), int(getattr(original, "session_uid", 0))

    @staticmethod
    def _animated_frame(material, scene) -> float | None:
        original = _original(material)
        if getattr(original, "animation_data", None) is not None:
            return float(getattr(scene, "frame_current_final", scene.frame_current))
        return None

    def _resolve_image_path(self, image, node) -> str:
        image = _original(image)
        pointer = image.as_pointer()
        revision = _image_revisions.get(pointer, 0)
        cached = self._image_paths.get(pointer)
        if cached is not None and cached[0] == revision:
            return cached[1]

        # Keep the MaterialX document small: its <image file> contains only a
        # stable URI while this side channel copies Blender's pixel buffer
        # straight into the already-loaded render delegate.
        try:
            width, height = (int(value) for value in image.size[:2])
            if width <= 0 or height <= 0:
                raise ValueError("image has no pixels")
            pixels = array("f", [0.0]) * (width * height * 4)
            image.pixels.foreach_get(pixels)
            path = f"noorray-memory://blender/{pointer:x}/{revision}"
            if not self._memory_textures().register(path, pixels, width, height):
                raise RuntimeError("hdNoorRay rejected the image buffer")
            if cached is not None:
                self._memory_textures().unregister(cached[1])
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            # A non-readable image (for example a missing source) still gets
            # the normal resolver path.  This is a fallback only: Blender
            # images with accessible pixels never get written to disk.
            path = ""

        if not path:
            try:
                path = image.filepath_from_user(image_user=node.image_user)
            except (AttributeError, RuntimeError, TypeError):
                path = getattr(image, "filepath", "") or getattr(
                    image, "filepath_raw", ""
                )
            if path:
                try:
                    path = bpy.path.abspath(path, library=image.library)
                except (AttributeError, RuntimeError, TypeError):
                    path = bpy.path.abspath(path)

        self._image_paths[pointer] = (revision, path)
        return path

    @staticmethod
    def _fallback_document(material, nonce: int) -> str:
        color = getattr(material, "diffuse_color", (0.8, 0.8, 0.8, 1.0))
        values = ",".join(format(float(component), ".9g") for component in color[:3])
        opacity = format(float(color[3] if len(color) > 3 else 1.0), ".9g")
        revision = (
            f"<!--noorray-image-revision:{int(nonce)}-->" if nonce else ""
        )
        return (
            '<?xml version="1.0"?><materialx version="1.39">'
            + revision
            + '<open_pbr_surface name="n0" type="surfaceshader">'
            + f'<input name="base_color" type="color3" value="{_xml_escape(values)}"/>'
            + f'<input name="geometry_opacity" type="float" value="{opacity}"/>'
            + '</open_pbr_surface><surfacematerial name="n1" type="material">'
            + '<input name="surfaceshader" type="surfaceshader" nodename="n0"/>'
            + "</surfacematerial></materialx>"
        )

    def _document_for(self, material, scene) -> tuple[tuple[int, int], str]:
        identity = self._identity(material)
        original_pointer = identity[0]
        material_revision = _material_revisions.get(original_pointer, 0)
        frame = self._animated_frame(material, scene)
        cached = self._content.get(identity)

        changed_images: tuple[int, ...] = ()
        if cached is not None:
            changed_images = tuple(
                pointer
                for pointer, seen_revision in cached.image_revisions.items()
                if _image_revisions.get(pointer, 0) > seen_revision
            )
        stale = (
            cached is None
            or cached.full_revision != _full_revision
            or cached.material_revision < material_revision
            or cached.frame != frame
            or bool(changed_images)
        )
        if not stale:
            return identity, cached.document

        # A comment nonce changes the transport payload when packed/generated
        # pixels change at a stable cache path.  MaterialX ignores comments,
        # so shader topology and compiled-program sharing remain unchanged.
        nonce = max(
            (_image_revisions.get(pointer, 0) for pointer in changed_images),
            default=0,
        )
        try:
            exported = export_material(
                material,
                self._resolve_image_path,
                transport_nonce=nonce,
            )
            if exported.unsupported_nodes:
                print(
                    "NoorRay MaterialX exporter: unsupported Blender nodes in "
                    f"{material.name}: "
                    + ", ".join(sorted(exported.unsupported_nodes))
                )
            document = exported.document
            image_revisions = {
                pointer: _image_revisions.get(pointer, 0)
                for pointer in exported.image_pointers
            }
        except Exception:
            document = self._fallback_document(material, nonce)
            image_revisions = {}

        self._content[identity] = _Content(
            document=document,
            image_revisions=image_revisions,
            material_revision=material_revision,
            full_revision=_full_revision,
            frame=frame,
        )
        return identity, document

    def collect(self, depsgraph, scene) -> dict[str, str]:
        """Return only changed documents and empty-string removal tombstones."""
        current: dict[str, str] = {}
        used_identities: set[tuple[int, int]] = set()

        # ``depsgraph.ids`` contains every material datablock reachable from
        # the scene, including materials on hidden objects and collections
        # outside the render view.  Uploading those documents also registers
        # their textures with the delegate, which can exhaust GPU memory on
        # large Blender scenes before the first ray is traced.  Hydratize only
        # materials referenced by renderable evaluated objects instead.
        materials = {
            slot.material
            for obj in depsgraph.objects
            if obj.type in {"MESH", "CURVE", "SURFACE", "FONT", "META"}
            and not obj.hide_render
            for slot in obj.material_slots
            if slot.material is not None
        }

        for datablock in materials:
            identity, document = self._document_for(datablock, scene)
            used_identities.add(identity)
            current[hydra_material_leaf(datablock)] = document

        pending = {
            SETTING_PREFIX + leaf: document
            for leaf, document in current.items()
            if self._sent.get(leaf) != document
        }
        pending.update(
            {
                SETTING_PREFIX + leaf: ""
                for leaf in self._sent.keys() - current.keys()
            }
        )

        self._sent = current
        if len(self._content) != len(used_identities):
            self._content = {
                identity: content
                for identity, content in self._content.items()
                if identity in used_identities
            }
        return pending

    def clear(self) -> None:
        if self._memory_texture_api is not None:
            for _revision, uri in self._image_paths.values():
                self._memory_texture_api.unregister(uri)
        self._content.clear()
        self._sent.clear()
        self._image_paths.clear()
