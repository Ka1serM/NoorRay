"""Render a slanted-edge target at several object distances."""

from pathlib import Path
from time import perf_counter

import pynoorray as nr


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TARGET = REPOSITORY_ROOT / "assets/tests/slanted_edge_target.glb"
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "build/slanted-edge-depths"
ROSS_RESOURCES = REPOSITORY_ROOT / "external/ROSS/resources"
LENS = ROSS_RESOURCES / "lenses/canon_automotive_fisheye/canon_automotive_fisheye.zmx"
GLASS_CATALOGS = [
    ROSS_RESOURCES / "glasscatalogs/schott.AGF",
    ROSS_RESOURCES / "glasscatalogs/ohara.AGF",
    ROSS_RESOURCES / "glasscatalogs/misc.agf",
]
SENSOR = ROSS_RESOURCES / "sensors/onsemi_AR0237.json"
HDRI = REPOSITORY_ROOT / "assets/textures/studio_small_03_2k.hdr"


def main():
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    session = nr.NoorRaySession()
    session.scene.import_file(str(TARGET))

    # Environment lighting and background settings.
    environment = session.scene.environment
    environment.color = nr.Vector3(1.0, 1.0, 1.0)
    environment.lighting_exposure = 10.0
    environment.visible_exposure = 0.0
    environment.rotation = 30.0
    environment.visible = True

    # Load a native texture, add it to the scene, then select it as the HDRI.
    hdri = nr.Texture(session.context, str(HDRI))
    hdri = session.scene.add(hdri)
    environment.set_hdri_texture(hdri)

    sensor = nr.RectangularSensor()
    sensor.load(str(SENSOR))
    sensor.resolution = (1280, 720)
    camera = nr.RealisticCamera(sensor)
    camera.load(
        lens_path=str(LENS),
        glass_catalog_paths=[str(path) for path in GLASS_CATALOGS],
    )

    session.scene.add_camera(camera)
    camera_instance = session.scene.active_camera
    session.scene.render_settings.samples = 32
    session.scene.render_settings.max_bounces = 10

    target = session.scene.get_object(session.scene.active_object_id)
    if target is None or camera_instance is None:
        raise RuntimeError("The imported target and camera are required")
    camera_instance.position = nr.Vector3(0.0, 0.0, 3.0)

    # Target Z, optical focus distance in meters, and aperture diameter in millimeters.
    render_settings = (
        (0.0, 12.0, 4.0),
        (1.0, 11.0, 8.0),
        (-1.0, 13.0, 12.0),
    )

    for target_z, focus_distance, aperture_diameter in render_settings:
        output = OUTPUT_DIRECTORY / (
            f"slanted-edge-z{target_z:g}-focus{focus_distance:g}m-"
            f"aperture{aperture_diameter:g}mm.exr"
        )
        print(f"Rendering to {output} ...", flush=True)
        start_time = perf_counter()

        target.position = nr.Vector3(0.0, 0.0, target_z)
        camera_instance.camera.focus_distance = focus_distance
        camera_instance.camera.aperture_diameter = aperture_diameter
        session.raytracer.render()
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
