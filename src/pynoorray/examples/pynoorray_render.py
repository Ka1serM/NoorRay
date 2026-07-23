"""Render a slanted-edge target at several object distances."""

from pathlib import Path
from time import perf_counter

import pynoorray as nr


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TARGET = REPOSITORY_ROOT / "src/pynoorray/examples/assets/slanted_edge_target.glb"
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "build/slanted-edge-depths"
ROSS_RESOURCES = REPOSITORY_ROOT / "external/ROSS/resources"
LENS = ROSS_RESOURCES / "lenses/lg-innotek/ADAS_NarrowViewing.zmx"
GLASS_CATALOGS = [
    ROSS_RESOURCES / "glasscatalogs/schott.AGF",
    ROSS_RESOURCES / "glasscatalogs/ohara.AGF",
    ROSS_RESOURCES / "glasscatalogs/misc.agf",
]
SENSOR = ROSS_RESOURCES / "sensors/onsemi_AR0237.json"


def main():
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    session = nr.NoorRaySession()
    session.scene.import_file(str(TARGET))

    sensor = nr.RectangularSensor()
    sensor.load(str(SENSOR))
    camera = nr.RealisticCamera(sensor)
    camera.load(
        lens_path=str(LENS),
        glass_catalog_paths=[str(path) for path in GLASS_CATALOGS],
    )

    session.scene.add_camera(camera)
    camera_instance = session.scene.active_camera
    session.scene.render_settings.samples = 32
    session.scene.render_settings.max_bounces = 64

    target = session.scene.get_object(session.scene.active_object_id)
    if target is None or camera_instance is None:
        raise RuntimeError("The imported target and camera are required")
    camera_instance.position = nr.Vector3(0.0, 0.0, 10.09)

    target.position = nr.Vector3(0.0, 0.0, 0.0)
    camera_instance.camera.focus_distance_cm = 1009.0
    camera_instance.camera.aperture_diameter_mm = 2.0

    output = OUTPUT_DIRECTORY / "slanted-edge.exr"
    print(f"Rendering to {output} ...", flush=True)
    start_time = perf_counter()
    session.raytracer.render_frame()
    beauty = session.raytracer.beauty
    beauty.save(str(output))
    elapsed_seconds = perf_counter() - start_time
    print(
        f"Saved {beauty.width}x{beauty.height} image to {output} "
        f"in {elapsed_seconds:.2f} s",
        flush=True,
    )


if __name__ == "__main__":
    main()
