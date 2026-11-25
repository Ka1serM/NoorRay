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
| Denoiser | NRD RELAX (diffuse + specular, no SH, and a second RELAX diffuse for fog light), unmodified upstream. The NRD C++ library plans the dispatches; the RELAX shaders compile offline with slangc to SPIR-V and WGSL through NoorRay wrappers that map the NRD resource macros to NoorRHI bindings. |
| Upscaler | AMD FSR 3.1 upscaler, unmodified upstream FidelityFX SDK. The FidelityFX C++ library plans the passes through a NoorRHI backend; the shaders compile offline like RELAX. Realtime only (no accumulation). |
| Gaussians | Ray-traced proxy geometry, stochastic sorting-free traversal (arXiv 2603.23637). Differentiable with Slang autodiff. |
| UI | RmlKit app. No ImGui. Materials are authored in Blender, the app has no node editor. |
| Blender | A native `bpy.types.RenderEngine` on `NoorRay.Python`. The C++ module reads the depsgraph through bpy and fills the session scene directly: no scene file, no USD, no Hydra. |
| Python | CPython stable ABI (abi3, Python 3.11 and later) through the C API. One binary loads into every Blender from 4.2 on. No nanobind: it needs exceptions and RTTI. |
| Unreal-type features | Landscape meshes, spline mesh deformation and similar go to UnrealKit/verylia. NoorRay has only the generic mesh API. |
| Removed | MaterialX, SVM interpreter, runtime Slang, OptiX/CUDA, ImGui, ImGuizmo, ImNodeFlow, VMA, Vulkan-Headers, CMakeRC, legacy `.nrscene`, the Hydra delegate (`hdNoorRay`). |

## 2. Repository layout

```
NoorRay/
  AGENTS.md  README.md  LICENSE  CMakeLists.txt  CMakePresets.json
  cmake/            CompilerOptions.cmake, NoorRayShaders.cmake
  tools/            check_comments.py, pre_commit.py, githooks/, serve_web.py
  external/         NoorRHI, RmlKit, RTXDI-Library (fork), NRD, MathLib and ShaderMake (for NRD), FidelityFX-SDK,
                    Compat (Emscripten headers for NRD), tinygltf, tinyobjloader, stb, tinyexr,
                    oneTBB (from RmlKit). OpenUSD is a prebuilt tree that NOORRAY_USD_ROOT names.
  web/              shell page, service worker, test page
  tests/            Test.h (NoorRHI style), unit tests, image tests, benchmarks, scenes/
  src/
    NoorRay/                    rendering library, no UI code
    NoorRay.App/                RmlKit app, desktop and web
    NoorRay.Cli/                headless renderer (scene in, EXR out); the image-test driver
    NoorRay.Python/             Python module `noorray` (CPython stable ABI, desktop only)
    NoorRay.Blender/            Blender addon (Python render engine) and the standalone .nrs exporter
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
Format/      SceneFile, SceneWriter (the binary .nrs format of SceneCodec.h, memory-mapped on desktop), Animation
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
  Denoise/     Relax (NRD instance, pools and RELAX dispatches)
  Upscale/     Fsr (FidelityFX FSR 3.1 context on a NoorRHI backend)
  Training/    GaussianTrainer, Dataset (COLMAP, transforms.json), Loss (L1 + D-SSIM), Adam, Densify
Shaders/     Slang modules and entry files (section 6)
Session/     Session (device + scene + renderer, the API that App, Cli and Python use)
```

### 3.1 Scene

- Host data in SoA layout. Each kind (mesh, instance, light, Gaussian cloud, volume, material, camera) lives in a
  `SlotMap` with stable handles.
- `SceneChange` records what changed: transform only, material values, topology, new or removed objects. `GpuScene`
  uses it to choose between a range upload, a TLAS update, a BLAS refit, or a rebuild.
- Cameras: perspective, orthographic, thin lens, fisheye, realistic lens (Kolb, lens snapshot from the old branch),
  rectangular sensor with configurable origin. Port from `src/libnoorray/Camera` and `Optics`.
- Lights: point, spot, rect, disk, sphere, directional, mesh emitters, environment. Port analytic intersection from
  `gaussians`. A spot with a radius is an area light: a disk of that radius that faces the receiver, radiance
  `I / (pi r^2)` with the spot mask, so BSDF rays hit it and NEE gets an MIS weight. Before, it was a delta light,
  and its reflections on the sharp metal bevels of the hero shot were the largest fireflies (pixel values up to 2700).
- Volumes: homogeneous media and dense grids. Delta tracking and ratio tracking.

### 3.2 Scene file

- `.nrs` is a little-endian binary file, read through `Core::MappedFile`. Sizes have 64 bits. Arrays start at 16-byte
  file offsets and copy with `memcpy`. `Format/SceneCodec.h` lists the fields of every object once, for the writer
  (`SceneEncoder`) and for the bounds-checked reader (`SceneDecoder`).
- Layout: a 32-byte header (`NRSC`, version, file size, offset of the shared objects, offset of the animation
  directory), the scene, the frame area, the shared objects, the animation directory.
- The scene is lossless: textures as GPU texels, materials, meshes with their topology, instances, Gaussian clouds,
  volumes, lights, cameras, the active camera, the environment and the node hierarchy. Handles are dense indices.
- Shared objects are the graphs, programs and node trees (with their groups). Each distinct one is in the file once,
  and the reader gives one `shared_ptr` for it, so materials share pipelines.
- The animation directory holds the key channels and the place of each key frame. The frame area holds the key frames
  of object keys and mesh sequences. `Animation::apply` decodes a key frame only when it shows it.
- The desktop writes the file from any importer. The web reads only this file.
- Textures need `texture-compression-bc` on WebGPU. A device without it gets a clear error at scene load.

## 4. Materials

### 4.1 IR

- `Materials::Graph` (done): a flat SSA list of float vector values (width 1 to 4). Each instruction is a constant,
  a hit input (`GraphInput`), a parameter slot, a texture sample, a call of a host function in
  `Shaders/Materials/Graph.slang` (noise, voronoi, wave, magic, checker, brick, gradient, white noise, HSV, blackbody,
  wavelength, Fresnel, layer weight), an extract or combine, or a math op. `Graph::open` validates widths, operand
  order, slots and outputs. `GraphBuilder` folds constants and merges equal values.
- Outputs (`GraphOutput`): the members of `BsdfInputs` (the Principled inputs), the emission, the chosen closure
  weight over its probability with a translucent flag, and the transparency. `BsdfInputs.smoothNormal` is not a graph
  output: the material kernel sets it to the interpolated normal before bump and normal maps.
- Bent normals as in Cycles (bump map correction on): the specular, metal, glass and coat lobes use the normal that
  `ensure_valid_specular_reflection` turns towards Ng (`clampShadingNormal`); diffuse and sheen use the bent normal.
  `bump_shadowing_term` drops a light on the other side of the smooth normal and softens diffuse light at the bump
  terminator. Diffuse and sheen samples below Ng are dropped, but their evaluation takes such light.
- Tangents as in Cycles: an unlinked Tangent input of an anisotropic BSDF is the radial tangent about Z, and a
  Tangent node can be radial about X, Y or Z. The radial tangent comes from the generated coordinates. It turns to
  world space like a normal through the `worldToObject` rows of `ShadingContext`, and is then made normal to N.
- Closures: the translator flattens Mix and Add Shader into weighted leaves. The graph selects one leaf per hit with
  `ClosureRandom` in proportion to the luminance of its weight. Emission and transparency sum over all leaves.
- Unlinked inputs become parameters, so materials with the same node setup share one graph, and value edits do not
  recompile. Inputs that decide if a material is transparent or emissive stay constants at 0 and 1, so the material
  flags fold.
- `Materials::Program` stays the interpreted register program for displacement
  (`docs/BlenderAnimationPlan.md`, phase 2c).
- `.nrs`: a material references its shared graph and holds the current parameter values, the graph textures, the fixed
  parameters and the node tree.

### 4.2 Shader compilation

- `Shaders/Materials/Graph.slang` declares `[noinline] evaluateMaterialGraph(ShadingContext, uint4 material)`. The
  helpers `graphParameter` and `graphTexture` get the parameter and texture addresses from `material`; the graph code
  has no static variables.
- `Materials::writeGraphCode` writes the body of one graph: SPIR-V in `GraphSpirv.cpp`, WGSL in `GraphWgsl.cpp`.
  `PipelineCache` splices it into `Entries/EvaluateMaterials.slang` and compiles one compute kernel per distinct
  graph in the background (0.5 to 1.5 s uncached on Vulkan). `evaluateFixedMaterials` evaluates the materials
  without a graph.
- A material kernel only evaluates the material: it writes a packed closure (`PackedClosure`) per path slot. All
  other kernels are plain compute kernels without material code, so each compiles once, independent of the scene.
- `binPaths` puts the surface hits in one bin per material program (bin 0 holds the Gaussian and volume hits).
  `advanceBins` writes the thread groups of each bin, and the material kernels run concurrently with
  `dispatchIndirect`. Then the shade kernels read the closures.
- `PipelineCache` keeps the material kernels of the last four graph sets. Value edits do not compile again.
- WebGPU uses the same kernels. Uncached Dawn compile of the ReSTIR kernels: 60 s in total (`restirResample` 45 s),
  before 519 s with the old callable link (ReSTIR temporal and spatial about 210 s each).
- The raster visibility pass does not run graphs: its alpha test uses the fixed parameters.

### 4.3 Sources

- Blender addon: `NoorRay.Python` walks the node tree through bpy (`Blender/GraphCompiler` for the surface,
  `Blender/MaterialReader` for the fixed fallback, `Blender/ProgramCompiler` for displacement) and sets the material in
  the session scene. `noorray.export_scene` writes the same scene as a `.nrs` file.
  `export_scene.py` is the older Python exporter; it stays for frame-range export until the C++ export writes animation.
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
| Upscaler | `Off`, `Fsr` | `PathTraced` without accumulation only. Render scale and sharpness are settings. |
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
- Temporal resources (film half-buffers, reservoirs, RELAX history, FSR history, previous depth and normal) live in
  the graph under a fixed key and survive between frames. A resize or a setting that changes their format drops
  only the affected keys.
- Debug view: the app can show any graph resource of the last frame (pass name and resource name), with a value
  range. This replaces separate debug AOV code.
- One `Reprojection` pass per frame writes the reprojection map (motion vector + validity from depth and normal
  tests). RTXDI temporal reuse, RELAX and FSR all read it. No pass computes its own reprojection.
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

- `PathPool`: SoA path state, about 256 B per slot with the queues and the packed closure. The default is 2M slots
  (512 MB): one path per pixel of 1080p, so a real-time frame needs one round. The loop runs
  `maxDepth * ceil(budget / pool)` iterations; regeneration stays on while the sample list can pass the pool.
- `Queues`: `AtomicPtr<uint>` counters. Each `advance` phase writes the indirect dispatch of the next grid stages
  (`gridDispatch`: active, bin source, surface, medium, shadow, free, binned) from their counts, at most the fixed
  grid (`gridGroups` workgroups), whose threads stride over the rest. An empty stage launches no group. Measured on
  the heavy test scene against direct dispatch of the full grid: 26.3 to 22.3 ms at 1080p on Vulkan, and faster on
  WebGPU too (46 to 43 ms, ReSTIR 62 to 53 ms at 540p) in spite of Dawn's indirect validation. The alpha shadow
  layers and the ReSTIR resampling still dispatch the full grid.
- `GpuProfiler`: timestamp pairs per kernel and pass (`noorray --profile`); `--timings` measures only whole frames,
  as each pair waits for the work before it.
- `Extend`: only traversal and the hit filter (stochastic opacity via `evaluateOpacity`, stochastic Gaussian
  acceptance). Writes the hit record.
- Texture filtering: ray cones (Akenine-Möller et al. 2021) give the mip level at each hit. The cone starts from the
  pixel footprint and grows with the BSDF roughness. On the web this also cuts texture cache misses.
- Sort and bin: by material id for shade, by direction octant for the next extend.
- `Shade[material]`: one compiled kernel per material with work in the queue. BSDF sampling, NEE (light alias table,
  environment CDF), MIS, Russian roulette, writes the shadow ray and the next ray.
- `Shadow`: any-hit traversal, first hit ends the search.
- ReSTIR PT is not a separate renderer: it uses the same pool, queues, bins and material kernels. The ReSTIR
  frame (`recordRestir`) runs the wavefront steps with `restirExtend` and `restirBounce` in place of `extend` and
  `shadeSurface`. `restirBounce` runs the RTXDI initial path context per bounce (`RTXDI_PathTracerContextInitial`,
  packed into a 96-byte `RestirPathRecord` per slot between the steps) and stores the initial reservoir at the end of
  the path. Then `restirResample` runs temporal and spatial reuse, and `restirShade` writes the film.
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
- Volumes: medium tracking in the path state under `hasVolumes`, in `Extend` and `Shadow`. `settings.x` holds the
  bounce limit (bits 0 to 7) and the volume bounce limit (bits 8 to 11) of Cycles; 0 volume bounces is single
  scattering. Homogeneous volumes attenuate in closed form. Only grid volumes use ratio tracking.
- Volume direct light as in Cycles: on each ray segment through a volume, `Extend` lights one point. One-sample MIS
  picks equiangular sampling (toward a light sampled from the segment middle) or distance sampling, 50/50, with the
  power heuristic. The point traces its own shadow ray. When the volume bounces are used up, the segment only
  attenuates (no delta-tracking scatter), as in Cycles. Scenes with alpha test keep delta tracking with direct light
  at the scatter in `ShadeMedium`, because the shadow ray in `Extend` has no material passes.
- Sobol patterns: the scramble seed (`settings.w`) is fixed for one accumulation, so the samples of a pixel stay
  stratified across frames. It changes every frame when the film does not accumulate. `settings.z` is the frame seed
  for hashed decisions and ReSTIR. Each depth has 4 patterns: light, BSDF and roulette, volume point and strategy,
  volume light. `raySeed` salts are `depth * 16 + salt`.
- Light linking as in Cycles: a receiver has one of up to 64 light sets (instance flags bits 24 to 31, volume kind
  bits 24 to 31, path flags bits 20 to 25 for the last scattering vertex). Each light record has a 64-bit mask of the
  sets it reaches, and `LightBuffers` writes one alias table per set. Shadow linking is not supported. ReSTIR uses the
  set of the surface for light sampling; its BSDF hits on emitters still use set 0.

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
- The resampling kernel has no material code. The hybrid shift calls `RAB_RandomReplay` (RTXDI fork hook), which
  starts a replay path in the wavefront pool instead of tracing it. Each pixel resamples in three stages: the forward
  shifts emit their replays, the inverse shifts emit theirs, then the pixel stores its reservoir. Between the stages
  the replays run through `restirExtend`, the bins, the material kernels and `restirReplay`, which writes a 48-byte
  `RestirReplayRecord`. The pixels that wait are in three pending lists. A sweep resamples as many pixels as the pool
  has slots for their replays (`poolCapacity / forward shifts`).
- Memory at 1920x1080 with one path slot per pixel: about 1.0 to 1.2 GB for the reservoirs, surfaces, closures, path
  and replay records (295 MB at 960x540).
- One reservoir for direct and indirect light (the unified reservoir of ReSTIR PT Enhanced): the light sample of the
  primary vertex is a path of length 2 whose reconnection vertex is the light (fork hook
  `RecordPrimaryNeeLightSample`), so `restirShade` adds only the light that the camera sees and the reservoir. A
  replay result keeps the BSDF pdf and the light set of its last vertex, for the NEE MIS weight of the shift. The fork
  evaluates that MIS weight with the BSDF pdf in every reconnection mode; before, the fixed threshold mode used weight
  1 there, which made the shifted target differ from the initial one.
- Analytic area lights (rect, disk, sphere, spot with a radius) get two NEE candidates per vertex in the ReSTIR
  initial path, because the path never hits them: the light sample, and a BSDF sample that hits a light and becomes a
  light sample with its uv (`sampleRestirBsdfLight`, bit 30 of the RTXDI light index). Each takes the power heuristic
  of its own strategy. Hero shot, 16 frames, 960x540, against Cycles: relMSE 2.20 to 0.50, largest pixel 2680 to 347,
  spikes (pixel > 3 where Cycles < 0.5) 172 to 113; 10.9 ms per frame (10.5 before). The rest are sub-pixel
  reflections of the small spot disks (4 cm, radiance about 4000) on the bevels; PT has as many (108). They are true
  signal; only more samples remove them without bias. RELAX anti-firefly was measured and stays off: it darkened the
  image by 5% and the highlights by 25%. Delta BSDF lobes take no BSDF candidate, so a mirror still shows no analytic
  area light; where two analytic lights overlap along a ray, only the nearest one is a BSDF candidate.
- No age-based rejection in temporal reuse: it drops the bright samples that live longest. Measured on the hero shot at
  512 spp: ReSTIR/Cycles 1.013 with it, 1.004 without it; the history cap of 8 limits the life of a sample.
- SHaRC radiance cache (`external/SHARC`, upstream v1.8.3, `Restir/RadianceCache.slang`), only in frames that are not
  accumulated (realtime; CLI `--no-cache` turns it off). An initial path ends in the cache at the first vertex of bounce
  depth >= 3 (after one indirect bounce) that has a cache entry and was not reached by a delta bounce; that vertex
  records its MIS-weighted emission plus the cached radiance as one emissive sample. The rule depends only on the path
  and the cache, so a replay that finds an entry before its path length fails, and one that ends there adds the cache.
  Update: the initial paths of one pixel per 5x5 block (offset per frame) feed the cache with the NEE and emission of
  each vertex (`SHARC_SEPARATE_EMISSIVE`), with RGB weights from the spectral throughput; a resolve kernel runs after
  shading. 2^22 entries, 64-bit keys, 160 MB. WebGPU has no buffer pointers for the atomics, so the cache is off there.
  Hero shot, 960x540: 9.6 ms against 11.2 ms per frame without denoising (10.9 against 12.5 with RELAX); 256 frames
  accumulated with the cache forced on: 0.992 of Cycles (1.002 without), relMSE at 16 frames unchanged (0.50).
  Ending at depth 2 was 8.6 ms but 0.978 of Cycles.
- Alpha-tested graph materials are opaque in the ReSTIR visibility rays; materials without a graph test their alpha.
- ReSTIR PT follows the path depth setting up to 15 bounces (the replay flags keep the bounce count in 4 bits).
- ReSTIR PT and RELAX are the defaults (CLI `--no-restir` and `--no-denoise` turn them off; Python `restir` and
  `denoise` are on). Scenes with Gaussians use the wavefront path tracer, as Gaussian vertices cannot reconnect.
- Volumes in ReSTIR PT: the resampled paths are surface paths. Every segment multiplies the throughput by the closed
  form transmittance (ratio tracking for grid volumes), so the shift is deterministic; the shift and replay segments
  and the light samples of the primary vertex do the same. On each fog segment of an initial path, the volume direct
  light of one point (5.5) goes to the film unresampled. Delta tracking on the segment offers its scatters as forks;
  each pixel keeps one by reservoir selection and continues it with probability 1/4 (weight 4 times the count) in
  the wavefront kernels after the ReSTIR frame (`restirFork`, then shade and extend). Forks add light but are not
  samples. Measured on the hero shot with fog at 512 spp: ReSTIR/PT 1.000, both 1.026 against Cycles; 17.7 ms per
  frame against 20.9 ms for PT at 960x540.
- The `temporal` setting (off: CLI `--no-temporal`, app "Temporal reuse", Python `temporal`) keeps each frame apart,
  so nothing smears. ReSTIR PT runs in the RTXDI `Spatial` mode with two neighbors per pixel, RELAX restarts its
  history every frame, and FSR is off. More neighbors cost much time and remove little noise (measured: the same
  denoised error with 2, 3 and 4). The frame cost is about the same as with temporal reuse.
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
- No own TAA: without accumulation, FSR resolves the sub-pixel jitter (its own jitter sequence). With accumulation,
  the denoised image blends into the accumulated mean with weight `samples / (samples + denoiseBlend)`.
- `Relax`: the NRD library plans the RELAX dispatches; NoorRay runs the unmodified shaders with NoorRHI pools.
  Inputs from the `denoisePrepare` kernel: diffuse and specular radiance divided by an RGB demodulation albedo
  (from the rgb wavelengths, so free of hero wavelength noise) + the primary-to-secondary hit distance, normal +
  roughness, view Z, and 2D motion in pixels with the view Z change. The diffuse prepass is off: the diffuse
  signal is never zeroed by a lobe choice, and the prepass blurs contact and cast shadows away. Unused NRD passes
  (split screen) fail only when they are dispatched.
- RELAX settings: luminance edge stopping 4 (default 2 and 1), fast history clamp 3 sigma (default 2), anti-firefly
  off. The defaults reject the rare bright paths of a few samples: on the hero shot without fog the 16-frame
  realtime image was 0.72 of the reference (metal highlights lost up to 30%); now 0.97 (PT) and 0.96 (ReSTIR).
- Fog light in front of the primary vertex (the camera segment light and the forks of the camera segment) goes to
  its own lighting bucket and a second denoiser, RELAX diffuse with the same guides and no demodulation. In the
  diffuse bucket, the surface albedo divides it (metal: 0.01) and the blur multiplies it by the neighbors' albedo,
  which made the card 5x too bright. The fog denoiser has the prepass on, luminance stopping 8 and no fast history:
  fog changes slowly and its multiple scattering comes from few bright forks. A miss through fog writes fog guides
  (albedo 1, the depth of the middle of the fog). Hero shot with fog, 16 realtime frames, ReSTIR + RELAX: ratio
  1.004, median relative error 0.11 (raw ReSTIR 0.90). The wavefront path tracer with RELAX stays noisy in fog: a
  scatter on the camera segment is its primary vertex, so the guides change from pixel to pixel.
- `Fsr`: FSR 3.1 reads the colour, depth and motion that the denoise kernels write at render resolution and writes
  the display image, which tone mapping reads. On WGSL, `Shaders/Upscale/Overrides/spd/ffx_spd.h` shadows the SPD
  include to load the SPD counter with `workgroupUniformLoad`, which WGSL uniformity analysis needs.
- `Composite` (`denoiseComposite`): denoised lighting times albedo plus denoised fog plus primary emission (a pixel
  without guides takes all its light from its raw frame), the display blend, then FSR or tone mapping. Browsers need the WebGPU `texture-formats-tier2`
  feature for the storage images: Chrome 140 does not have it, Chrome 156 does.

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
Denoise/     Denoise (prepare and composite), NrdPass and the RELAX entry wrappers
Upscale/     FsrPass, FsrCallbacks, the FSR entry wrappers, Overrides/spd
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
6. Callable shaders (NoorRay no longer uses them, see 4.2): `RayTracingPipelineDesc::callable` and
   `callShader(index, payload)`, with the index counted
   across the libraries of a pipeline. The software fallback links each callable into the kernel as a donor function
   behind a switch (`SpirvModule`, `WgslModule`), with one payload type per source. Vulkan callable modules get the
   bindless heap back in set 0 (slangc counts the payload in space 0) and lose DontInline. Test: `callShader` on
   hardware, on the fallback and on WebGPU.

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
- `RTXDI_PT_APPLICATION_PATH_TRACER`: the application runs the path loop (the wavefront). It removes
  `RTXDI_GenerateInitialSamples` and RTXDI's random replay loop; the hybrid shift calls
  `RAB_RandomReplay(neighborSample, inout surfaceForResampling, inout targetFunction, inout ptud)` instead.
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

### NoorRay.Python

- Module `noorray`, built as `noorray.abi3.so` against the stable ABI of Python 3.11 (`Py_LIMITED_API`). The C API keeps
  the module free of exceptions and RTTI. `CMAKE_POSITION_INDEPENDENT_CODE` is on for desktop builds, because the
  module links the static libraries.
- `Engine(width, height)`: one NoorRHI device and one `Session`. Methods: `sync(depsgraph, approximate=False)`,
  `resize`, `configure(samples, restir, denoise, compile_in_background)`, `set_camera(transform, orthographic,
  vertical_fov, orthographic_height, shift_x, shift_y, near, far, focus_distance, f_stop)`, `render()` (one frame,
  waits), `read(buffer)` (linear RGBA floats, top row first). Properties: `complete`, `samples`, `width`, `height`,
  `warnings` (of the last sync). `render` and `read` release the GIL; `sync` holds it.
- `Blender/SceneSync` brings the scene to the depsgraph:
  - It walks `depsgraph.object_instances` and takes MESH and LIGHT instances, which include curves, text and
    Geometry Nodes instances. An instance key is the object, or the instancer and its `persistent_id`. A mesh key is
    the evaluated mesh and the materials of its slots, so Geometry Nodes instances share one mesh.
  - The state of each object is read once per sync. Blender reuses one temporary object for every dupli, so a dupli
    is cached by its original, its data and its instancer, not by the evaluated object.
  - Meshes are read again when `depsgraph.updates` names the object or its data with a geometry update; materials
    when they or any node tree update; an image update reads the whole scene again. Lights, the camera and the world
    are read on every sync. What a sync does not see leaves the scene. A failed sync clears the scene, so the next
    one starts from nothing.
  - `Blender/MeshReader` reads the bulk arrays with `foreach_get` into C++ memory through a memoryview (`position`,
    `.corner_vert`, the normals of `normals_domain` (`vertex_normals` or `corner_normals`), UV maps, the render colour
    attribute). It splits triangles and quads itself as Blender does and takes `loop_triangles` only for n-gons.
  - The mesh has the Cycles layout: positions are the Blender vertices, there is no weld, and each stream has a
    domain. POINT normals and colours are per vertex; CORNER normals, UVs, CORNER colours and tangents are per
    triangle corner, at the index of the corner in `indices`. FACE normals give a mesh without normals, which shades
    with the geometric normal. Corner tangents sum over corners with the same vertex, normal and UV, as MikkTSpace
    does. `MeshRecord.cornerStreams` tells the shaders which streams are per corner.
  - Triangulation, the index sort and the stream output run on all cores (`Blender/Parallel.h`); scratch arrays stay
    uninitialised. When only positions and normals change, `MeshReader::fill` writes them into the mesh in place
    (`Scene::updateMesh`); a quad whose split diagonal flips patches its six corners and uploads the mesh again.
  - Light linking: `readEmitters` reads `light_linking.receiver_collection` (objects and child collections with
    INCLUDE or EXCLUDE) and `visible_volume_scatter` of the lights in `depsgraph.objects`. Receivers that the same
    linked emitters reach share a light set; set 0 holds the receivers outside every collection. The masks are set
    after the instance walk. The scene file does not store the sets yet.
  - `Blender/Reader` keeps the first error; unsupported nodes fail the sync unless `approximate` is set.
    The addon renders always with `approximate` and reports the deduplicated warnings once per sync; its .nrs export
    has a Strict Materials option (off by default) that fails on unsupported nodes.
- `export_scene(depsgraph, path, approximate=False)` runs a `SceneSync` into a new scene and writes it with
  `Format::SceneWriter` (current frame; no device). It returns the warnings. `SceneWriter` writes material programs,
  mesh topology and mesh attributes, so displacement survives the round trip.
- Later: scene edits per object (meshes from numpy arrays, lights, materials), AOVs, Gaussian training.

### NoorRay.Blender

- Blender extension `noorray_blender` (`blender_manifest.toml`), staged by CMake into
  `build-*/NoorRay.Blender/noorray_blender` with the module (`noorray.abi3.so` on Linux, `noorray.pyd` on Windows) and
  no other dependency: the Linux module links libc++ statically and exports only `PyInit_noorray`; the Windows module
  links python3.dll against the headers of the official Python 3.11 NuGet package (FetchContent).
- `cmake --workflow --preset blender-addon` writes `build-linux-release/noorray_blender-<version>.zip`, one extension
  for linux-x64 and windows-x64 with both stripped modules; it builds `windows-release` through llvm-mingw itself and
  validates the zip with Blender when Blender is in PATH. The command line exporter
  `export_scene.py` is staged next to it and calls the `noorray` module.
- Export: the operator `noorray.export_scene` (3D view sidebar, tab NoorRay, and File > Export) writes the current frame
  of the viewport depsgraph as a `.nrs` file through `noorray.export_scene`.
- `NoorRayRenderEngine` (`bl_idname = "NOORRAY"`), Cycles shader nodes (`bl_use_shading_nodes_custom = False`),
  EEVEE for Material Preview.
  - `render`: syncs the render depsgraph, renders to the sample count or the time limit of Cycles, shows the film every
    second and writes the Combined, Depth, Mist, Normal and Diffuse Color passes. A scene without an active camera is
    an error. With Persistent Data the engine stays for the next frame, whose sync reads only what changed.
  - Motion blur: `motion_steps` sub-frames at equal shares of the area of the shutter curve. Each step syncs with
    `accumulates=True`, which keeps the film across the scene change (`Renderer::setAccumulatesChanges`), and gets an
    equal share of the samples and of the time limit. All animation blurs: cameras, deformation, materials, lights.
  - `view_update`: syncs the depsgraph when an update changes more than the selection.
  - `view_draw`: reads the camera from the view and projection matrices, renders one frame while the film is not
    complete and draws it with the display space shader, through a shared OpenGL texture where the backend has one.
- A sync without a scene camera adds a default camera, which the viewport replaces with its view.
- Settings (`scene.noorray`) tag the scene when they change, so a change from Python or an animation redraws too. Light
  paths, seed, time limit and exposure are the settings of Cycles. The engine adds itself to the built-in panels that
  list `BLENDER_RENDER`.
- Memory: the path pool takes the capacity that suits the VRAM; allocator blocks grow by an eighth; one upload is at
  most 128 MB; the host copy of the texels goes after the upload; textures over the limit of Simplify shrink; unused
  textures and shaders nothing reads are pruned; the fog RELAX denoiser opens only with volumes.
- Displacement: a program that the register interpreter knows runs as before; every other graph compiles to the
  `DisplaceGraph` kernel. A displaced mesh keeps its rest positions and normals for Generated coordinates and for the
  bump of Displacement and Bump.
- Voronoi uses the PCG hash of Blender 5; colour ramps other than RGB linear, ease and constant are sampled, with 8
  pieces for each span.
- Not supported: shadow catchers, shadow linking, transparent glass film, light group, cryptomatte and index passes,
  stereo, colour grids of VDB files, noise textures in grid volume shaders, pixel-size Wireframe, material previews.
- Next steps, in order:
  1. Deforming meshes through `setMeshVertices` when only the positions change.
  2. Zero-copy display with Blender's Vulkan backend.
  3. Light passes, viewport resolution scale.
  4. macOS in the extension zip.

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
3. Orphan branch skeleton: CMake, presets, AGENTS.md, tools, hooks, external submodules.
4. `Core`, `Scene`, `Spectral`.
5. `Materials` (IR, folding, emitter, linker), `Format`, `Import`.
6. `Gpu` (GpuScene and all buffers, acceleration structures, texture table, material kernels).
7. Shaders: `Common`, `Spectral`, `Bsdf`, `Lights`, `Scene`.
8. `Render` core, `Wavefront`, adaptive sampling.
9. `Renderer`, `RendererSettings`, `FrameGraph`, `Raster` shading and raster primary visibility.
10. `Reuse` (RTXDI bridge, reservoir recording in the wavefront kernels, ReSTIR DI and PT) and `Denoise` (RELAX SH,
    composite), the presets.
11. `Training`.
12. `Session`, `NoorRay.Cli`, `NoorRay.App`, `NoorRay.Python`, `NoorRay.Blender`.
13. Tests and reference scenes.

Build order after the implementation: NoorRHI tests on both backends, then the library on Vulkan, the CLI image tests
on Vulkan, the same on Dawn, the web build in Chrome, the app, then Python and Blender.

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
| Addon | `src/hdnoorray` (node tree export only; the Hydra delegate is not ported) |
| Python | `src/pynoorray` (API ideas only; the module is new) |

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

- NRD (library and RELAX shaders) and RTXDI use the NVIDIA RTX SDK license; FidelityFX uses MIT. All 58 RTXDI headers in the fork
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
