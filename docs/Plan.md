# NoorRay rewrite plan

NoorRay becomes a small, clean rendering library with thin integrations, in the style of NoorRHI, RmlKit, UnrealKit
and verylia. The work starts on a new orphan branch. Code is ported module by module from
`origin/vulkan-headers-submodule` (the newest line; it contains `bsdf`, `wavefront`, `gaussians` and `hdNoorRay`),
plus Gaussian training from `origin/gaussians` and adaptive sampling from `origin/gpgpu`.

The whole plan is implemented first. The build and the tests come after the implementation.

## 1. Decisions

| Topic | Decision |
|---|---|
| Repository | New orphan branch in the NoorRay repo. Same layout, AGENTS.md, tools and hooks as NoorRHI/RmlKit/UnrealKit. |
| GPU API | NoorRHI only. Vulkan on the desktop, WebGPU (Dawn native and browser). No raw Vulkan, no VMA. |
| Ray tracing | Spectral only. Raster is RGB. Gaussian training is RGB. |
| Ray tracing architecture | Wavefront with inline ray queries (`traceRayInline`) on both backends. No ray tracing pipelines, no SBT. |
| Feature switches | Compile-time link constants (`extern static const bool`) per entry file. Variants are chosen from the scene content. No runtime feature branches in hot kernels. |
| Materials | No MaterialX. Blender node tree, USD PreviewSurface and glTF PBR become one material IR. The IR compiles to SPIR-V functions at runtime. Closure model: Principled BSDF + Emission + one Mix Shader of two Principled BSDFs. |
| Renderer | One `Renderer`. Primary visibility (raster, traced), shading (raster RGB, path traced spectral), reuse, denoiser, accumulation and adaptive sampling are settings. Presets `Viewport` (raster) and `PathTraced` are setting values. `PathTraced` is used for preview and final frames: unbiased reuse, display-only denoising, converges. |
| ReSTIR | RTXDI (own fork `Ka1serM/RTXDI-Library`), ReSTIR DI and ReSTIR PT, made spectral with an `RTXDI_COLOR` macro. Used for realtime and progressive rendering. |
| Denoiser | Own Slang-only port of NRD RELAX with SH (diffuse + specular), as used in NoorRay before. No NRD library. |
| Upscaler | None for now. No FSR3. |
| Gaussians | Ray-traced proxy geometry, stochastic sorting-free traversal (arXiv 2603.23637). Differentiable with Slang autodiff. |
| UI | RmlKit app. No ImGui. Materials are authored in Blender, the app has no node editor. |
| Unreal-type features | Landscape meshes, spline mesh deformation and similar go to UnrealKit/verylia. NoorRay has only the generic mesh API. |
| Removed | MaterialX, SVM interpreter, runtime Slang, OptiX/CUDA, ImGui, ImGuizmo, ImNodeFlow, VMA, Vulkan-Headers, ShaderMake, CMakeRC, MathLib, NRD, FidelityFX, legacy `.nrscene`. |

## 2. Repository layout

```
NoorRay/
  AGENTS.md  README.md  LICENSE  CMakeLists.txt  CMakePresets.json
  cmake/            CompilerOptions.cmake, NoorRayShaders.cmake
  tools/            check_comments.py, pre_commit.py, githooks/, serve_web.py
  external/         NoorRHI, RmlKit, RTXDI-Library (fork), tinygltf, tinyobjloader, stb, tinyexr, flatbuffers,
                    oneTBB (from RmlKit). nanobind comes with NoorRay.Python. OpenUSD is a prebuilt tree that NOORRAY_USD_ROOT names.
  schema/           NoorRayScene.fbs, MaterialProgram.fbs
  web/              shell page, service worker, test page
  tests/            Test.h (NoorRHI style), unit tests, image tests, benchmarks, scenes/
  src/
    NoorRay/                    rendering library, no UI code
    NoorRay.App/                RmlKit app, desktop and web
    NoorRay.Cli/                headless renderer (scene in, EXR out); the image-test driver
    NoorRay.Hydra/              Hydra render delegate (desktop only)
    NoorRay.Blender/            Blender addon (Python): node tree to material IR, Hydra engine glue
    NoorRay.Python/             nanobind module
```

Presets: `linux-release`, `linux-relwithdebinfo`, `linux-webgpu-release`, `windows-release`, `windows-webgpu-release`,
`web-release`. The `web-release` preset builds the library, the app and the CLI-free test page only.

Namespaces: root `NoorRay`, one namespace per module (`NoorRay::Scene`, `NoorRay::Render`, ...). Files and folders
PascalCase. `src/` is the include root (`"NoorRay/Scene/Scene.h"`). An object that can fail has a default constructor
and `open()` returning `std::expected<void, Core::Error>`. No exceptions, no comments except `//!` facts.

## 3. Library modules (`src/NoorRay/`)

```
Core/        Error, Log, Math (float3/float4x4 over NoorRHI types), Handle<T>, SlotMap, Hash
Scene/       Scene, Transform, Instance, Mesh, GaussianCloud, Volume, Light, Camera, Lens, Sensor, Environment,
             RenderSettings, SceneChange (dirty tracking per kind)
Materials/   MaterialProgram (IR), MaterialLibrary, Texture, TextureCache,
             Ir/Builder, Ir/Validator, Ir/ConstantFolding,
             Spirv/Emitter (IR -> SPIR-V function), Spirv/Templates (embedded shade kernel templates),
             Spirv/Linker (patches the stub function through NoorRHI SpirvModule)
Import/      UsdImporter (desktop), GltfImporter, ObjImporter, PlyGaussianImporter, PbrtImporter,
             PreviewSurface (USD -> IR), GltfMaterial (glTF -> IR)
Format/      SceneFile (flatbuffers NoorRayScene, read in place, memory-mapped on desktop), SceneWriter
Gpu/         GpuScene (owns all GPU data of a Scene), MeshBuffers, GaussianBuffers, VolumeBuffers,
             LightBuffers (alias table, light BVH later), EnvironmentBuffers (importance CDF),
             TextureTable, MaterialKernels (compiled shade kernels per material),
             AccelerationStructures (BLAS per mesh, BLAS per Gaussian cloud, one TLAS)
Spectral/    SpectralTables (basis + film weight per wavelength), WavelengthSampler, Sellmeier, Conductors (n,k),
             Blackbody
Render/      Renderer, RendererSettings (+ presets), FrameGraph (passes, transient and temporal resources),
             Targets (film half-buffers, AOVs), Accumulation, SampleRate, Reprojection, Exposure, Timings,
             VariantKey, PipelineCache
  Wavefront/   PathPool, Queues, Generate, Extend, Shade, Shadow, Accumulate, Regenerate
  Raster/      VisibilityBuffer, RasterShading (RGB PBR), GaussianSplatting (sorted raster splats), Sky, ToneMap,
               Bloom, AmbientOcclusion
  Reuse/       Restir (RTXDI DI + PT host side, temporal and spatial passes, validation)
  Denoise/     Relax (RELAX SH denoiser passes), Composite (display blend, TAA while moving)
  Training/    GaussianTrainer, Dataset (COLMAP, transforms.json), Loss (L1 + D-SSIM), Adam, Densify
Shaders/     Slang modules and entry files (section 6)
Session/     Session (device + scene + renderer, the API that App, Cli, Hydra and Python use)
```

### 3.1 Scene

- Host data in SoA layout. Each kind (mesh, instance, light, Gaussian cloud, volume, material, camera) lives in a
  `SlotMap` with stable handles.
- `SceneChange` records what changed: transform only, material values, topology, new or removed objects. `GpuScene`
  uses it to choose between a range upload, a TLAS update, a BLAS refit, or a rebuild.
- Cameras: perspective, orthographic, thin lens, fisheye, realistic lens (Kolb, lens snapshot from the old branch),
  rectangular sensor with configurable origin. Port from `src/libnoorray/Camera` and `Optics`.
- Lights: point, spot, rect, disk, sphere, directional, mesh emitters, environment. Port analytic intersection from
  `gaussians`.
- Volumes: homogeneous media and dense grids. Delta tracking and ratio tracking.

### 3.2 Scene file

- `schema/NoorRayScene.fbs`: meshes (quantized positions optional, octahedral normals, half UVs), instances,
  materials (IR blobs), textures (BC7/BC6H plus mip chain), Gaussian clouds (compact: position, scale, rotation,
  opacity, SH), volumes, lights, cameras, environment.
- The desktop writes it from any importer. The web reads only this file. It is read in place without copies.
- Textures need `texture-compression-bc` on WebGPU. A device without it gets a clear error at scene load.

## 4. Materials

### 4.1 IR

- `schema/MaterialProgram.fbs`: a flat SSA list of instructions, constants, parameters (editable values), textures
  and outputs.
- Ops: constants, parameters, Math (all Blender Math modes), VectorMath, Mix (float/vector/color), MapRange, Clamp,
  ColorRamp (baked to a 1D table), RGBCurves (baked), ImageTexture (wrap, filter, color space), TexCoord (UV0-3,
  generated, object), Geometry (position, normal, tangent, incoming), Attribute (vertex color, instance attributes),
  NormalMap, Bump, Fresnel, LayerWeight, SeparateXYZ/CombineXYZ, HueSaturationValue, Invert, Gamma, BrightContrast,
  noise later.
- Outputs: the Principled BSDF inputs (base color, metallic, roughness, IOR, transmission, coat, sheen, specular
  tint, anisotropy, normal, alpha, emission color and strength, thin film thickness and IOR), plus an optional
  second Principled set with a mix factor.
- `Validator` rejects unknown ops, cycles, and type errors with a clear message. Unsupported Blender nodes fail at
  export in the addon.
- `ConstantFolding` folds constant subgraphs. Constant colors become spectral basis coefficients at compile time.

### 4.2 SPIR-V compilation

- Each shade kernel template (`Shaders/Wavefront/Shade.slang`, `Shaders/Reuse/...`, `Shaders/Raster/...`) calls a
  stub `[noinline] BsdfInputs evaluateMaterial(ShadingContext context)` and `[noinline] float evaluateOpacity(...)`.
  The stubs are compiled with names kept.
- `Spirv/Emitter` writes the function body for one material: instructions, texture handles from `TextureTable`,
  parameters from a per-material constant block (value edits need no recompile).
- `Spirv/Linker` replaces the stub body in the template module through NoorRHI `SpirvModule::replaceFunction`
  (section 7). The result opens as a normal `ComputePipeline` with `openAsync`. On WebGPU NoorRHI translates it to
  WGSL at open, so the same path works in the browser.
- `MaterialKernels` caches kernels by IR hash and variant key. The pipeline cache persists them on disk.
- While a kernel compiles, the material shades with the previous kernel of that material, or with a neutral grey
  kernel on first load. This is a visible state in the UI, not a hidden substitution.

### 4.3 Sources

- Blender addon: walks the node tree, writes the IR, stores it on the USD material prim as `noorray:program`
  (binary asset). The Hydra delegate and the USD importer read it.
- USD importer: `noorray:program` if present, else `UsdPreviewSurface` through `PreviewSurface`.
- glTF importer: metallic-roughness PBR, KHR_materials_transmission, ior, emissive_strength, clearcoat, sheen.

## 5. Rendering

### 5.1 Renderer and settings

There is one `Renderer`. Raster, hybrid, realtime and progressive rendering are settings of it.

| Setting | Values | Notes |
|---|---|---|
| Primary visibility | `Raster`, `Traced` | `Raster` writes the visibility buffer. |
| Shading | `Raster`, `PathTraced` | `Raster`: RGB PBR, environment map, AO, splatted Gaussians, no secondary rays. `PathTraced`: spectral wavefront path tracing. |
| Reuse | `Off`, `Di`, `DiPt` | `PathTraced` only. |
| Denoiser | `Off`, `Relax` | `PathTraced` only. Display only, never written into the accumulation (5.6). |
| Accumulation | on, off | Resets on camera, scene or setting changes. |
| Adaptive sampling | on, off | `PathTraced` only. Per-pixel candidate rate from reuse confidence and, with accumulation, the error (5.4). |
| Path budget | paths per frame | Default: one per pixel. Adaptive sampling moves it, it does not add to it. |
| Samples, bounces, error target | numbers | The render stops at the sample limit or when the error target is met. |

Invalid combinations are fixed in `RendererSettings::validated()`: `Raster` shading forces `Raster` primary
visibility, and reuse, denoiser and adaptive sampling become `Off`.

Presets are plain `RendererSettings` values, not separate code:

| Setting | Viewport | PathTraced |
|---|---|---|
| Primary visibility | Raster | Raster (Traced when the camera has depth of field or motion blur) |
| Shading | Raster | PathTraced |
| Reuse | Off | DiPt |
| Denoiser | Off | Relax |
| Accumulation | off | on |
| Adaptive sampling | off | on (rate `[1/8, 4]`) |

`PathTraced` is the one setup for preview and final frames. It is fast while the camera moves and converges to the
unbiased image while the camera stands still. There is no separate "fast" reuse mode. The reuse is always
unbiased:

| Reuse rule | Value | Why |
|---|---|---|
| Visibility rays in the shift | always | Without them the shift is biased. Cost: about one shadow ray per spatial neighbor. |
| Reuse MIS | pairwise | Unbiased, and O(N) in the neighbor count. |
| Boiling filter | none | It is biased. The M-cap and random neighbors keep fireflies from spreading. |
| Temporal M-cap | 20 (setting) | Limits the correlation length between frames. |
| Spatial neighbors | new random neighbors each frame | Decorrelation. |
| Raster primary | jittered sub-pixel position each frame | Same estimator as a traced pinhole primary ray. |

- The `Renderer` builds a `FrameGraph` each frame from the settings (model: kajiya `kajiya-rg`). A pass declares its
  reads and writes. The graph orders the passes, places the NoorRHI barriers, and aliases transient targets. A
  pass that a setting does not use is not recorded, and its pipelines and targets are not opened. Settings change
  only the frame graph, targets and pipelines. The scene and the GPU data stay.
- Temporal resources (film half-buffers, reservoirs, RELAX history, TAA history, previous depth and normal) live in
  the graph under a fixed key and survive between frames. A resize or a setting that changes their format drops
  only the affected keys.
- Debug view: the app can show any graph resource of the last frame (pass name and resource name), with a value
  range. This replaces separate debug AOV code.
- One `Reprojection` pass per frame writes the reprojection map (motion vector + validity from depth and normal
  tests). RTXDI temporal reuse, RELAX and TAA all read it. No pass computes its own reprojection.
- Acceleration structures are built on first use. With `Raster` shading and no traced primary, no BLAS or TLAS is
  built, so the `Viewport` preset has no ray tracing cost.
- Reuse off is plain wavefront path tracing: the resampling passes are not recorded, and the link constant
  `writesReservoirs` removes the reservoir writes from the path tracing kernels.
- The visibility buffer stores instance, primitive and barycentrics: the same record that the extend kernel writes.
  Raster primary visibility replaces only the first extend. Then the primary ray traces only the Gaussians (instance
  mask), with tMax at the raster depth.
- Pipelines for the likely next settings open with `openAsync` ahead of use.
- `Raster` shading is RGB, `PathTraced` shading is spectral. Both write the same RGB film, so tone mapping and the
  app do not see a difference.

### 5.2 Variants

- Link constants: `hasGaussians`, `hasVolumes`, `hasDispersion`, `hasAlphaTest`, `writesReservoirs` (from the
  reuse setting, not from the scene). Each entry file sets them with
  `export static const bool`. Dead code and its registers disappear.
- `VariantKey` comes from `GpuScene` content. A scene without Gaussians never opens a Gaussian variant.
- Entry files for each kernel and each used combination are generated by `cmake/NoorRayShaders.cmake` from a small
  list, so no combination is written by hand.
- Small switches stay runtime uniforms: camera type, AOV set, adaptive sampling, bounce limits, clamps.

### 5.3 Spectral core

- One hero sample `u` per path from the Owen-scrambled Sobol sampler (port), 4 equally spaced wavelengths,
  importance-sampled with an analytic visible-range pdf with a closed-form inverse.
- RGB to spectrum: the linear sRGB basis (Mallett and Yuksel 2019). At path start, one fetch per wavelength from
  `SpectralTables` gives the basis (float3) and the film weight (CMF times XYZ-to-linear-RGB divided by pdf, float3).
  Every upsampling is then 4 dot3. The film conversion is 4 FMA.
- Emission, environment and Gaussian radiance sum in RGB and convert once, because the basis is linear.
- Payload and path state store `u` as 16 bit. A spectral value is one `float4`.
- Dispersion (Sellmeier) only in dispersive materials, under `hasDispersion`. Secondary wavelengths end there.
- Conductors: small n,k tables per metal, sampled at the 4 wavelengths. Blackbody: analytic Planck.
- Environment maps and float tables use `Rgba16Float` (filterable on every WebGPU device).

### 5.4 Wavefront path tracer

```
generate -> extend -> sort/bin -> shade[material] -> shadow -> accumulate
   ^                                                              |
   +------------- regenerate finished paths into the pool -------+
```

- `PathPool`: packed SoA path state (about 64-96 B per path). The size is fixed (default 1M, lower when the heap is
  small) and does not depend on the resolution.
- `Queues`: `AtomicPtr<uint>` counters. No indirect dispatch: every stage is a direct dispatch of one fixed grid
  (`gridGroups` workgroups, sized to fill the GPU, at most the pool) whose threads stride over the stage's GPU count.
  An empty stage costs one grid of threads that exit at once, and Dawn validates no indirect arguments. Each
  kernel reads its exact work count (for example `Regenerate` reads 0 when regeneration is off).
- `Extend`: only traversal and the hit filter (stochastic opacity via `evaluateOpacity`, stochastic Gaussian
  acceptance). Writes the hit record.
- Texture filtering: ray cones (Akenine-Möller et al. 2021) give the mip level at each hit. The cone starts from the
  pixel footprint and grows with the BSDF roughness. On the web this also cuts texture cache misses.
- Sort and bin: by material id for shade, by direction octant for the next extend.
- `Shade[material]`: one compiled kernel per material with work in the queue. BSDF sampling, NEE (light alias table,
  environment CDF), MIS, Russian roulette, writes the shadow ray and the next ray.
- `Shadow`: any-hit traversal, first hit ends the search.
- With reuse on (`writesReservoirs`), the path state records the ReSTIR PT reconnection data while it traces:
  reconnection vertex, partial Jacobian, `rcWiPdf`, suffix radiance, path flags and the hero `u`. At path end,
  `Accumulate` builds the initial reservoir with the RTXDI functions (`RTXDI_MakePTReservoir`, streaming). The DI
  initial samples come from `Shade` in the same way.
- `Accumulate`: film (RGB float) as two half-buffers (odd and even frames), AOVs (albedo, normal, depth, position,
  cryptomatte, Gaussian overdraw).
- `SampleRate` (one pass, replaces separate adaptive sampling masks): writes a per-pixel candidate rate `r` in
  `[rMin, rMax]` (default `[1/8, 4]`). `Generate` turns it into a compacted path queue: `floor(r)` paths plus one
  more with probability `fract(r)`, in a 16x16 tile pattern so that every pixel gets a new candidate at least every
  `1/rMin` frames. The wavefront queue already compacts, so skipped pixels cost no idle lanes and the trace cost
  scales linearly with the rate. The rate comes from:
  - with reuse (idea from Hu 2026, "Adaptive ReSTIR"): reservoir confidence `M`, disocclusion and invalid
    reprojection (rate 1 or more), motion length, the RTXDI duplication score, highly specular or transmissive
    primary hits (rate at least 1/2 or 1), a black reservoir, and `M = 0` from `Validate` after a scene change;
  - with accumulation: the relative error from the two half-buffers against the error target. It stays correct
    when reuse makes frames correlated, where a per-pixel Welford estimate would stop pixels too early. A pixel
    below the target gets rate 0 and stops accumulating, but keeps its reservoir for the neighbors.
  - The rates are scaled so that their sum meets the frame budget (paths per frame, a setting). The budget that
    converged pixels free goes to disoccluded and high-error pixels (`r > 1`).
- Unbiased sub-1 rates: the thesis skips candidates and keeps the old sample, which loses paths that only the
  canonical sample can reach, so dark corners and highlights drift. Here a candidate made with probability `r` has
  its contribution weight divided by `r` (Russian roulette on the canonical candidate). The expected contribution
  weight is unchanged, so GRIS stays unbiased. `rMin` limits the `1/r` weight and the fireflies it would cause. The
  image test "reuse on versus off" covers this.
- Progressive PT submits in bounded chunks of samples, never one long dispatch (browser and OS watchdogs).
- Volumes: medium tracking in the path state under `hasVolumes`, free-flight and ratio tracking in `Extend` and
  `Shadow`.

### 5.5 Gaussians

- One BLAS per Gaussian cloud with world-space proxy triangles (icosphere, configurable triangle count).
  `primitiveIndex / proxyTriangles` gives the Gaussian id. No TLAS instance per Gaussian.
- Rendering: stochastic sorting-free traversal. Each candidate is accepted with its opacity in the hit filter, the
  nearest accepted hit wins. Shading: SH in RGB, converted with the basis, GI mode or direct-colour mode (port).
- `Raster` shading: sorted splat rasterization (GPU radix sort by depth, tile binning), RGB.
- Gaussian vertices are not reconnection vertices in ReSTIR PT.

### 5.6 Reuse and denoising

- `Restir`: RTXDI host contexts for ReSTIR DI (light presampling, temporal, spatial) and ReSTIR PT (temporal and
  spatial with the hybrid shift). The initial samples come from the wavefront path tracer (5.4), not from RTXDI's own
  path loop. Bridge `RAB_*` on `Ptr<T>` and NoorRHI handles. The reuse is always unbiased (5.1): shift visibility rays,
  pairwise MIS, no boiling filter.
- ReSTIR PT Enhanced (Lin, Kettunen, Wyman 2026, reported 2-3x faster than ReSTIR PT) parts that the fork already
  has and NoorRay uses: footprint-based reconnection criteria (`RTXDI_RESTIRPT_RECONNECTION_MODE_FOOTPRINT`),
  duplication maps and decorrelation (they also shorten the correlation between accumulated frames, and give the
  duplication score for `SampleRate`), and the heuristic spatial neighbor selection.
- Resampling kernels that replay path prefixes need material evaluation. They use one switch kernel over the
  scene's material functions, with inputs sorted by material.
- With accumulation, frames are correlated by the temporal reuse. Each frame is unbiased, so the average converges.
  The M-cap and random spatial neighbors keep the correlation short. With a still camera the temporal reprojection
  is the identity, so no reprojection error enters the history.
- Denoising never changes the converged result. `Accumulate` always stores the raw frames (two half-buffers). The
  display is `lerp(raw, denoised, w)` per pixel, where `w` falls from 1 to 0 as the half-buffer relative error falls
  to the error target. A converged pixel shows the raw unbiased average. While the camera moves, `w` is 1 and the
  image is the RELAX output. The final image output option `denoise final` is explicit and off by default.
- RELAX gets the raw frame, not the accumulation, so its own history works the same while moving and standing.
- Validation (idea from kajiya, made unbiased): the reconnection shift reuses the cached suffix radiance. After a
  light, material or object change, that cache is stale and the reuse would be biased. On such a frame, a
  `Validate` pass replays the suffix of every temporal reservoir with its stored random seed (RTXDI random replay)
  in the new scene and writes the new radiance. Reservoirs whose path became invalid get `M = 0`. The replay is the
  same estimator in the new scene, so the reuse stays unbiased. A camera-only change needs no validation.
- For animation renders, each frame validates, then starts with the reservoirs of the previous animation frame
  (temporal reuse through the reprojection map), then accumulates to the error target.
- `Composite` TAA: while `w > 0`, the denoised image goes through a TAA pass (reprojection map, color box
  clamping) that resolves the per-frame sub-pixel jitter. The raw accumulation never goes through TAA.
- `Relax`: RELAX diffuse + specular with SH, ported pass by pass to Slang (pre-pass, temporal accumulation, history
  fix, history clamping, a-trous spatial filter, anti-firefly, split screen debug). Inputs: diffuse and specular
  radiance + hit distance as SH (RGB after the film conversion), normal + roughness, view Z, motion vectors.
  Settings and pass order port from `Realtime/Denoiser.cpp`.
- `Composite`: denoised lighting times albedo plus emission, display blend, TAA, exposure, tone map, output.

### 5.7 Raster

- Port from verylia `UI/Viewport`: scene shaders, environment map, tone mapper, bloom, ambient occlusion.
- `Exposure` (both shading modes): GPU luminance histogram, read back one frame late, clipped percentiles, smooth
  adaptation. Manual exposure for final renders. Bloom: blur pyramid and reverse pyramid (kajiya `post`), off by
  default for final renders.
- Visibility buffer pass for raster primary visibility. `Raster` shading uses the same compiled material functions
  through a raster shade template (RGB). Viewport overlays: grid, light billboards, selection outline, transform gizmo.

### 5.8 Training

- `GaussianTrainer`: forward with stochastic traversal, backward with the Monte Carlo estimator of arXiv 2603.23637
  in `[Differentiable]` Slang (`bwd_diff`), gradients with `atomicAdd` on `AtomicPtr<float>`.
- `Adam` in compute, `Densify` (clone, split, prune by gradient and opacity), BLAS refit after each step, rebuild
  after densify.
- `Dataset`: COLMAP and `transforms.json`. `Loss`: L1 + D-SSIM in compute.
- RGB only. Port the math from `origin/gaussians` `Training/Kernels`.

## 6. Shaders (`src/NoorRay/Shaders/`)

```
Common/      Types.h (shared C++/Slang records), Sampling, Sobol, Random, Frame, RayDifferential, Packing
Spectral/    Wavelengths, Basis, Film, Sellmeier, Conductors, Blackbody
Bsdf/        Principled (diffuse, GGX conductor, dielectric, coat, sheen, thin film), EnergyLut, Mix
Lights/      AliasTable, Analytic, Environment, Emitters
Scene/       SceneSurface (hit record to surface), Gaussians, Volumes, Camera, Lens
Wavefront/   SampleRate, Generate, Extend, Shade (template), Shadow, Accumulate, Regenerate, Sort
Reuse/       RtxdiBridge, RestirDi, RestirPt, Validate
Denoise/     Relax/*, Composite, Taa
Render/      Reprojection, Histogram, RayCone
Raster/      VisibilityBuffer, Shade (template), GaussianSplat, Sky, ToneMap, Bloom, Ao, Overlays
Training/    Forward, Backward, Adam, Loss, Densify
Entries/     generated entry files per kernel and variant
```

- Every `[shader(...)]` entry is a thin wrapper in an entry file. The logic is in imported modules.
- Textures only use `SampleLevel`/`SampleGrad` (ray differentials). No `Sample()` in compute.
- No reliance on f16, subgroups or timestamps. Timestamps only for `Timings`.
- Shared C++/Slang records in headers like NoorRHI's `ShaderTypes.h`, with `static_assert` on size and offsets.

## 7. Changes in other repositories

### NoorRHI

1. Fix the failing Vulkan test "GPU culling on the async queue feeds draws with a count".
2. Public hit filter for inline queries:
   `traceRayInline<F : IHitFilter>(scene, flags, mask, ray, inout F filter, out RayHit hit)`. The filter gets the
   candidate (t, barycentrics, instance index and id, primitive, geometry, front face) and the object-space ray.
   Vulkan: `RayQuery` `Proceed()` loop. Fallback: the existing `INoorRHIHitFilter` traversal. Tests on both backends:
   ignore, accept, accept and end, instance mask, cull flags, first-hit flag.
3. Software traversal speed: the BLAS LBVH is collapsed on the GPU into 8-wide quantized nodes (96 B, CWBVH style:
   8-bit boxes, octant-ordered slots, leaves of up to 3 triangles), in indirect rounds. The TLAS stays binary
   because it is rebuilt every frame. A SAH-binned builder comes later, after benchmarks in
   `tests/benchmarks/RayTracingBenchmark.cpp`.
4. `SpirvModule` builds on both backends and has `replaceFunction(name, donor)`. It merges types, constants,
   capabilities and extended instruction sets, copies donor callees, and links a donor callee to a host function
   with the same name and type. Host stubs and helpers are `[noinline]`. Test: patch a compute stub, run it on both
   backends.
5. Runtime `Shader` from SPIR-V words: no NoorRHI change. NoorRay owns the words and fills `Shader::code` with spans.

### RmlKit

- Use the parent's NoorRHI when `NoorRHI::NoorRHI` already exists (`if(NOT TARGET ...)` around
  `add_subdirectory(NoorRHI)`).

### RTXDI fork

Branch `spectral` in `external/RTXDI`.

- `RTXDI_COLOR` and `RTXDI_COLOR_COMPONENTS` (default `float3`, 3) for every color value in `PT/` and
  `RTXDI_BrdfRaySample`. `RTXDI_Luminance(float4)` is the mean of four pre-weighted spectral samples.
  `RTXDI_ColorFromFloat3` keeps the packed NEE light data in the radiance slot. `GI/` stays RGB; NoorRay does not use it.
- With 4 components the reservoir has `wavelengthSample`, copied on resampling, packed in a fifth `uint4`.
- Hooks, empty by default: `RTXDI_PT_FINISH_INITIAL_SAMPLE(reservoir)` stores the path's wavelength sample,
  `RTXDI_PT_BEGIN_SHIFT(reservoir)` sets the wavelengths before a shift evaluates the sample.
- `RTXDI_BrdfRaySampleProperties::SetNotReconnectible()`: dispersive and Gaussian vertices get a zero footprint, so
  they are never reconnection vertices. In fixed-threshold mode `RAB_GetSurfaceRoughness` returns 0 for them.
- The RGB default packs and behaves like upstream.

## 8. Integrations

### NoorRay.App (RmlKit)

- Views: viewport (verylia `ViewportElement` pattern), outliner, inspector (transform, light, camera, material
  parameters, Gaussian cloud), render settings (preset, primary visibility, shading, reuse, samples, bounces, adaptive, denoiser), statistics (pass
  timings), training view (dataset, loss curve, controls).
- Input: orbit, pan, zoom, fly, pick (cryptomatte AOV), transform gizmo, drag-and-drop scene files.
- Desktop: open USD, glTF, OBJ, PLY, PBRT, NoorRay scene files. Web: NoorRay scene files and PLY.

### NoorRay.Cli

- `noorray render scene --spp 1024 --error 0.01 --out image.exr` (single settings override the preset), `noorray convert in.usd out.nrs`,
  `noorray train dataset --out cloud.ply`. Image tests and benchmarks use it.

### NoorRay.Hydra and NoorRay.Blender

- Render delegate: meshes, instancers, lights, cameras, render settings, Gaussian splats, materials from
  `noorray:program`, render buffers (beauty, AOVs). Export to Blender through external memory and semaphore
  (NoorRHI `Interop`, Vulkan only).
- Addon: node tree to IR exporter (replaces `materialx_export.py`), Hydra engine registration, render settings UI.

### NoorRay.Python

- nanobind: Session, Scene (create, edit, import), Renderer (settings, presets, render, read beauty and AOVs as numpy),
  Trainer.

## 9. Tests

- Unit (NoorRHI-style `Test.h`): IR validator and folding, SPIR-V emitter output per op (run on GPU and compare with
  CPU evaluation), scene file round trip, spectral tables (basis round trip, film weights sum), wavelength sampler
  pdf, alias table, Sobol.
- Image tests through `NoorRay.Cli` on Vulkan and Dawn, compared with stored references (FLIP or relative MSE):
  white furnace, Cornell box, glass prism (dispersion), textured scene, Gaussian scene, volume scene, one scene per
  preset, adaptive sampling convergence. `PathTraced` with reuse on and with reuse off converge to the
  same image (unbiasedness check). The displayed image converges to the raw average (denoiser blend weight reaches 0). The half-buffer variance with reuse on stays within a set factor of the true
  error. Sub-1 candidate rates (`rMin` 1/8, fixed and adaptive) converge to the same image as rate 1.
- Browser test page: the same scenes in Chrome, image compare, shader compile time and frame time printed.
- Benchmarks: traversal (primary, diffuse, shadow), frame time per preset, training step time.

## 10. Implementation order

Everything is written before the first build. The order follows the dependencies:

1. NoorRHI changes (section 7) and their tests.
2. RmlKit and RTXDI fork changes.
3. Orphan branch skeleton: CMake, presets, AGENTS.md, tools, hooks, external submodules, schemas.
4. `Core`, `Scene`, `Spectral`.
5. `Materials` (IR, folding, emitter, linker), `Format`, `Import`.
6. `Gpu` (GpuScene and all buffers, acceleration structures, texture table, material kernels).
7. Shaders: `Common`, `Spectral`, `Bsdf`, `Lights`, `Scene`.
8. `Render` core, `Wavefront`, adaptive sampling.
9. `Renderer`, `RendererSettings`, `FrameGraph`, `Raster` shading and raster primary visibility.
10. `Reuse` (RTXDI bridge, reservoir recording in the wavefront kernels, ReSTIR DI and PT) and `Denoise` (RELAX SH,
    composite), the presets.
11. `Training`.
12. `Session`, `NoorRay.Cli`, `NoorRay.App`, `NoorRay.Python`, `NoorRay.Hydra`, `NoorRay.Blender`.
13. Tests and reference scenes.

Build order after the implementation: NoorRHI tests on both backends, then the library on Vulkan, the CLI image tests
on Vulkan, the same on Dawn, the web build in Chrome, the app, then Hydra, Python and Blender.

## 11. Ported code sources

| New module | Source |
|---|---|
| Camera, Lens, Sensor | `vulkan-headers-submodule:src/libnoorray/Camera`, `Optics` |
| Lights, alias table | `src/libnoorray/Lights`, `Shaders/Raytracer/AnalyticLights.slang` |
| BSDF, energy LUTs | `Shaders/Raytracer/*Lobe.slang`, `CompositeBsdf.slang`, `Ggx.slang`, `EnergyLut*`, `Materials/Shading` |
| Spectral | `Spectrum.slang`, `Sellmeier.slang`, `RgbToSpectrum.slang` (replaced by the linear basis), `SpectralTables.h` |
| Gaussians | `GaussianHit.slang`, `GaussianAnyHit.slang`, `GaussianSplat.slang`, `Mesh/Assets` |
| Volumes | `Shaders/Viewport/ViewportVolumes.slang`, scene volume code |
| Import | `src/libnoorray/Scene/Import` (USD, glTF, OBJ, PBRT), `IO` |
| ReSTIR, RELAX | `Realtime/Restir.cpp`, `Realtime/Denoiser.cpp`, `Shaders/RealtimeRaytracer`, NRD RELAX shaders |
| Raster | verylia `src/Verylia/UI/Viewport` |
| Training | `origin/gaussians:src/libnoorray/Training` |
| Adaptive sampling | `origin/gpgpu:src/Kernels/Finalize.cu`, `Generate.cu`, `KernelHelpers.h` |
| Hydra, addon | `src/hdnoorray` |
| Python | `src/pynoorray` |

### 11.1 Inspiration: Adaptive ReSTIR (Xiao yi Hu, 2026 thesis)

Repositories `XiaoyiHu1998/Adaptive_ReSTIR` (Falcor, on Lin et al. ReSTIR PT) and `Adaptive_ReSTIR_Thesis_Report`.

Taken:
- Candidate rates below 1 per pixel, with a tile pattern so every pixel gets new candidates regularly (16x16 tiles).
- Heuristic inputs: `M`, disocclusion by a depth test, motion length, duplication score, specular and
  transmissive primary hits, black output.
- Its recommended direction: rates above 1 for low-history pixels (its future work), and a high minimum rate.

Changed:
- Unbiased: `1/r` on the canonical candidate weight, so the thesis' darkening (lost rare paths, darkening of
  reused samples with a moving camera) becomes variance, not bias.
- Slow light updates (its 7.3.5): `Validate` after a scene change sets `M = 0`, which raises the rate there.
- One rate pass for realtime and accumulation, and a fixed path budget instead of a fixed rate.

### 11.2 Inspiration: kajiya (EmbarkStudios)

Taken:
- Render graph with temporal resources by key, and a debug view of any graph resource (`kajiya-rg`).
- One shared reprojection map for all temporal passes.
- ReSTIR sample validation after scene changes, made unbiased with random replay.
- Ray cones for texture LOD in traced paths.
- GPU histogram auto exposure, blur pyramid bloom.
- Built-in furnace test switch (here: an image test and a debug setting).

Not taken, because it does not converge to the reference or the plan already covers it better:
- Biased ReSTIR GI with screen-space occlusion in the spatial pass, half-resolution tracing with a resolve, the
  rank-voted irradiance cache, single-bounce reflections. NoorRay traces full paths and uses unbiased ReSTIR PT.
- Firefly and caustic suppression by roughness bias. Biased. A later option only, off by default.
- Forced 4% F0 dielectrics and the packed G-buffer. NoorRay keeps the full Principled BSDF and a visibility buffer.
- Rust GPU shaders and DLSS.

## 12. Open points

- RELAX is derived from NRD, and NRD and RTXDI use the NVIDIA RTX SDK license. All 58 RTXDI headers in the fork
  carry `LicenseRef-NvidiaProprietary`. The license allows distribution only "in object code format" inside an
  application, with the notice "This software contains source code provided by NVIDIA Corporation." The web build
  ships WGSL text, which is close to source. Check this before a public web build. Fallback: an own ReSTIR PT in
  Slang from the papers (GRIS, ReSTIR PT, ReSTIR PT Enhanced), with the same bridge interface.
- ReSTIR PT Enhanced unifies DI and GI into one reservoir (2 x 64 B per pixel instead of 2 x (88 + 16) B, one
  pass less). The fork has separate DI and PT reservoirs. The plan keeps them separate; the unification is a later
  fork change.
- Material IR shared with UnrealKit's `MaterialProgram`: not now. Can come later through an UnrealKit exporter that
  writes NoorRay IR.
- Camera pose and exposure gradients in training: later.
- ROSS camera integration from `origin/gpgpu`: decide if it is still needed.
