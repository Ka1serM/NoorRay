# TODO

## Camera & Render Resolution

- **Sensor / render resolution as camera setting**: Make render resolution a first-class camera property, like `sensorWidthMm` already is. Currently `renderWidth/renderHeight` in `CameraBase` are set externally by the raytracer and `RealisticCamera` drives the resolution through `getPreferredRenderSize()` via a loaded sensor file. Instead, expose width/height directly in `CameraSettings` (and the camera UI) so any camera type can declare a fixed output resolution. The raytracer picks it up the same way it already does for realistic cameras.

- **Sensor resolution editable in UI**: `RealisticCamera::sensorResolutionWidth/Height` is currently read-only (loaded from the sensor JSON). Add UI drag inputs so it can be overridden without editing the sensor file.

## Performance

- **`updateSceneSettings` allocates a new buffer every frame reset**: Calling this on every camera move creates/destroys a GPU buffer each time. Use a persistently-mapped host-visible buffer and just memcpy the new data instead.

- **Same issue for `meshBuffer`** in `updateMeshes` and `instancesBuffer` in `updateTLAS`.

- **Too many memory barriers per wavefront bounce**: `computeBarrier` inserts a full compute→compute barrier between every pass (Generate, Advance, Extend×N, Shade×N, Connect×N). Consider whether finer-grained buffer barriers on specific bindings would reduce pipeline stalls.

- **`bindAllDescriptorSets` called twice on LUT generation frame**: When the ray LUT is generated, descriptor sets are re-bound before and after the LUT dispatch. The second bind is only needed because the LUT generator uses its own pipeline; consider keeping the main pipeline bound and only re-binding once after the LUT dispatch.

- **Advance pass dispatches 1×1×1**: The Advance shader runs a single workgroup to flip the ray queue pointers and reset counters. If Advance grows, switch to indirect dispatch so the CPU doesn't need to know active ray counts.

## Misc

- **`Scene::replaceObject` always sets `TLAS` dirty even for camera swaps**: Cameras don't affect the TLAS. Distinguish camera replacements from mesh instance replacements.
