# NoorRay Agent Guide

## Build
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON
cmake --build build -j$(nproc)
```

## Architecture

### Camera
- `CameraInstance` stores a tagged pointer to a concrete camera allocated in unified memory via `nr::rstd::allocator<T>`
- FreeCamera dispatches on tagged concrete type, using the matching typed allocator
- Camera sensor provides the single source of truth for render resolution; Raytracer reads it internally
- Concrete camera types: `FisheyeCamera`, `PerspectiveCamera`, `ThinLensCamera`, `OrthographicCamera`, `RealisticCamera`

### Renderer
- `Raytracer` constructor takes `(Context&, Scene&)` — reads resolution from active camera sensor, fallback 1280×720
- `NoorRay` constructor takes `(int windowWidth, int windowHeight)` — window size only

### Viewport (was Tonemapper)
- Handles UI tonemapping + selection outlines
- `runCli()` does NOT run viewport shader; saves raw HDR via `stbi_write_hdr`

### Scene
- `Types.h` deleted — each header includes GLM directly and uses qualified or file-local `using` declarations
- GPU light structs — one per type:
  - `PointLightGpu.h` — position, color, intensity, range, sourceRadius, falloff
  - `SpotLightGpu.h` — + direction, inner/outer cone angle
  - `RectLightGpu.h` — + width, height, twoSided
- `LightInstance` — single SceneObject subclass, holds only a type discriminant + index into Scene's unified arrays
  - `renderUi()` reads/writes directly to unified memory — no local data copy
  - `onTransformUpdated()` syncs position/direction into the unified slot
  - Uses `initData` private member only for bootstrapping new slots (copy-on-init, stale afterward)
- Scene owns three growable unified-memory arrays (`cudaMallocManaged`), one per light type
  - `registerLight()` — grow, copy from initData
  - `unregisterLight()` — swap-remove, update displaced instance's index
- `Environment::textureIndex` defaults to `-1`; set to `0` in `Nooray.cpp` after HDRI load

### Tests
- `tests/` directory, `BUILD_TESTING=ON` enables `add_subdirectory(tests)`
- Test runner compiles as `noorray_test`, shells out to `NoorRay` binary, checks output file

## Key Decisions
- No DPI scaling for render resolution — sensor is the single source
- CLI saves only HDR output (no viewport pass)
- Camera must exist in scene before Raytracer is constructed
- CameraInstance uses direct tagged pointer storage (no move-construct into unified memory)
- Lights use per-type GPU structs in unified memory arrays (not a fat LightGpu with type discriminant)
- LightInstance holds only an index — reads/write directly to unified memory, no local GPU data copy
