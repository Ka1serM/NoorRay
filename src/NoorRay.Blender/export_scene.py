import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy

from noorray_blender import helpers, noorray


def parse_frames(text, scene):
    if text is None:
        return scene.frame_start, scene.frame_end
    first, _, last = text.partition("-")
    return int(first), int(last or first)


def main():
    arguments = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(prog="blender -b <file.blend> --python export_scene.py --")
    parser.add_argument("output", help="the NoorRay scene file (.nrs) to write")
    parser.add_argument("--frames", help="the frame or frame range A-B to export; the default is the scene frame range")
    parser.add_argument("--approximate", action="store_true", help="export unsupported shader nodes as the closest fixed material, with a warning")
    options = parser.parse_args(arguments)
    scene = bpy.context.scene
    first, last = parse_frames(options.frames, scene)
    if last < first:
        parser.error("the frame range ends before it starts")
    start = time.monotonic()
    noorray.set_helpers(helpers)
    try:
        warnings = noorray.export_scene(bpy.context.evaluated_depsgraph_get(), os.path.abspath(options.output), approximate=options.approximate,
                                        frames=(first, last))
    except RuntimeError as error:
        print(f"NoorRay export failed: {error}", file=sys.stderr)
        sys.exit(1)
    finally:
        noorray.set_helpers(None)
    for warning in warnings:
        print(f"NoorRay export: {warning}", file=sys.stderr)
    print(f"NoorRay export: wrote {options.output} in {time.monotonic() - start:.1f} s")


main()
