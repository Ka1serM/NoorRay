import bpy


def state(label, image):
    props = {}
    for name in dir(image):
        if any(part in name.lower() for part in ("dirty", "update", "bind", "pack", "tag")):
            try:
                value = getattr(image, name)
                if not callable(value):
                    props[name] = repr(value)
            except Exception as exc:
                props[name] = f"<{type(exc).__name__}>"
    packed = image.packed_file
    props["packed_pointer"] = packed.as_pointer() if packed is not None else 0
    props["packed_size"] = packed.size if packed is not None else 0
    print(label, props)


image = bpy.data.images.new("Probe", width=2, height=2)
state("new", image)
image.pixels = [1.0, 0.0, 0.0, 1.0] * 4
state("pixels", image)
image.update()
state("update", image)
image.pack()
state("pack", image)
try:
    import _bpy_hydra

    print(
        "cache path",
        _bpy_hydra.cache_or_get_image_file(
            bpy.context.as_pointer(), image.as_pointer()
        ),
    )
except Exception as exc:
    print("cache error", type(exc).__name__, str(exc))
state("cache", image)
image.pixels = [0.0, 0.0, 1.0, 1.0] * 4
image.update()
state("second update", image)
image.pack()
state("second pack", image)
try:
    print(
        "second cache path",
        _bpy_hydra.cache_or_get_image_file(
            bpy.context.as_pointer(), image.as_pointer()
        ),
    )
except Exception as exc:
    print("second cache error", type(exc).__name__, str(exc))
state("second cache", image)

unpacked = bpy.data.images.new("UnpackedProbe", width=2, height=2)
unpacked.pixels = [1.0, 1.0, 0.0, 1.0] * 4
unpacked.update()
state("unpacked update", unpacked)
try:
    print(
        "unpacked cache path",
        _bpy_hydra.cache_or_get_image_file(
            bpy.context.as_pointer(), unpacked.as_pointer()
        ),
    )
except Exception as exc:
    print("unpacked cache error", type(exc).__name__, str(exc))
state("unpacked cache", unpacked)
