# MaterialX materials in NoorRay

NoorRay is growing a MaterialX material system in which MaterialX and OSL
evaluate the shading *graph* and NoorRay keeps ownership of all scattering
mathematics. This document describes the state of that work, the versions it is
pinned to, and how to reproduce its diagnostics.

**Status: working end to end**, on both ingestion paths: a standalone `.mtlx`
referenced from a `.nrscene` (`NoorRay --cli`) and a *real* MaterialX node
graph arriving through Hydra from Blender (its own node graph -- Principled
BSDF, Image Texture nodes, connections and all -- not a flattened summary).
Both compile through `MaterialXCompiler` to an OptiX direct callable and
render with NoorRay's own BSDFs. Image textures are supported on both paths,
verified through Blender specifically (an Image Texture node feeding Base
Color renders the actual image, not just a solid average color). Editing a
material in Blender re-syncs and recompiles it in place, so changes are
visible in the viewport without restarting -- verified end to end with a live
edit-and-rerender test, not just a compile check. Getting here took five
compounding, entirely silent bugs -- see
[Defects fixed along the way](#defects-fixed-along-the-way-worth-knowing-about)
for what they were and how each was actually diagnosed (mostly: stop trusting
a passing build and go look at what's really being compiled and what Hydra is
really sending). See also [Current limitations](#current-limitations) for what
is intentionally still out of scope, and [Phase 0 findings](#phase-0-findings)
for the historical feasibility spike this was built on top of.

## Architecture

Both ingestion paths converge on one representation, so a `.mtlx` file on disk
and a material arriving from Hydra are compiled by the same code:

```
Hydra HdMaterialNetwork2
    └── OpenUSD hdMtlx conversion
            └── MaterialX document
                    │
Standalone .mtlx ──┘
                    │
                    ▼
          MaterialXCompiler
                    │
             OSL shader group
                    │
         OSL OptiX direct callable
                    │
                    ▼
          MaterialEvaluation
                    │
                    ▼
             NoorRay Bsdf
```

The division of labour is deliberate and load-bearing:

* MaterialX and OSL evaluate textures, noise, procedurals, math, transforms and
  parameter connections.
* NoorRay owns BSDF sampling, evaluation, PDFs, spectral upsampling and energy
  conservation. OSL never samples a BSDF, and OSL's own scattering closures are
  not used for any path-tracing decision.

The seam between the two is `MaterialEvaluation`
(`src/libnoorray/Shading/MaterialEvaluation.h`): a compact, fixed-size struct
that a material program writes and the integrator consumes. It is an ABI —
compiled programs are cached across runs, so changing its layout requires
bumping `NrMaterialEvaluationAbiVersion`.

## Supported versions

| Component | Pinned version | Why |
| --- | --- | --- |
| MaterialX | **1.39.4** | The version Blender 5.2's `libusd_ms` embeds (symbols are namespaced `MaterialX_v1_39_4`). Matching it keeps the document schema identical on both sides of the Hydra boundary. |
| OSL | **1.15.5, built with `OSL_USE_OPTIX=ON`** | NoorRay executes shader groups as OptiX direct callables. A CPU-only OSL cannot produce device code and is rejected at configure time. |
| OptiX | 9.1 | As for the rest of NoorRay. |
| CUDA (`NR_CUDA_ROOT`) | 13.x | As for the rest of NoorRay. |
| LLVM/Clang | **22** | OSL's host build, JIT, and embedded CUDA bitcode all use the same LLVM major version. |

MaterialX and NoorRay never share an ABI with Blender's embedded MaterialX:
only canonical MaterialX XML crosses the Hydra boundary. Two incompatible
MaterialX libraries must never be loaded into one Blender process, and passing
strings rather than `Document` pointers is what makes that safe.

## Build configuration

MaterialX and OSL support is always built from the pinned git submodules in
`external/`. This includes `NoorRayMaterialXCompiler`, the closure bridge,
the OSL device runtime, the standalone ingestion path, and hdNoorRay's
ingestion path when `NR_BUILD_HYDRA` is enabled. There is no MaterialX build
toggle, OSL SDK-root override, or bootstrap/download step.

### Getting the dependencies

MaterialX has no third-party dependencies and builds in a few minutes as an
ordinary `add_subdirectory` of NoorRay's own configure.

OSL is heavier: it needs OpenImageIO, LLVM and pugixml, and it must be built
with `OSL_USE_OPTIX=ON` against the CUDA toolkit so its shadeops compile to
NVPTX. Distribution packages will not do — Fedora's `openshadinglanguage`
1.15.4 ships `oslconfig.h` with `OSL_USE_OPTIX 0`. Rather than requiring a
separately pre-built OSL SDK, NoorRay drives OSL's own build itself: at
configure time it invokes a completely separate `cmake -S external/osl -B
build/<config>/osl-build ... && cmake --build ... --target install` into
`build/<config>/osl-install`, then picks up the result with an ordinary
`find_package(OSL CONFIG)`. This keeps OSL building as a genuinely standalone
CMake project (its own `CMAKE_SOURCE_DIR` resolves to `external/osl`, exactly
as it would building OSL by hand) — no submodule patching, no symlinks or
extra files anywhere under NoorRay's own `src/`. The only manual step is
installing OSL's build dependencies:

```bash
# Development packages OSL needs (Fedora).
# flex/bison are OSL's own shader-language (.osl) parser generator.
sudo dnf install llvm22-devel clang22-devel OpenImageIO-devel pugixml-devel \
                 flex bison

cmake -S . -B build/release \
      -DNR_CUDA_ROOT=/usr/local/cuda \
      ...the usual NoorRay options...
```

OSL uses `NR_CUDA_ROOT` and the automatically discovered LLVM/clang 22
toolchain for both its host JIT and embedded CUDA shadeops bitcode. See
["OSL bitcode must match its own reader's LLVM version"](#osl-bitcode-must-match-its-own-readers-llvm-version)
for why those LLVM versions must remain aligned.

`external/osl`'s own build then lands in `build/release/osl-build`, installed
to `build/release/osl-install` — both untracked, ordinary build outputs under
the already-gitignored `/build` directory, rebuilt automatically (and
incrementally) on every reconfigure.

## Supported surface profile

The initial implementation targets a bounded profile rather than arbitrary
MaterialX closure trees:

**Terminals:** `standard_surface`, `open_pbr_surface` (where the MaterialX
version provides it), and the existing manually parsed `UsdPreviewSurface` as a
fallback.

**Surface inputs:** base color, metalness, roughness, specular level / IOR,
transmission, transmission color, IOR, emission, opacity, shading normal.
`generalized_schlick_bsdf` (used by both terminals' specular/transmission
lobes) is handled the same way as `dielectric_bsdf`/`conductor_bsdf`.

**Graph nodes:** image textures, UV/primvar readers, and MaterialX math, color,
noise, procedural and transform nodes feeding those inputs. These reuse
MaterialX's own OSL implementations; NoorRay does not reimplement them.

**Correctly read but not yet shaded:** anisotropy (`roughnessU`/`roughnessV`),
thin film (`thinfilmThickness`/`thinfilmIor`), sheen (`sheenColor`/
`sheenRoughness`), subsurface (`subsurfaceRadius`/`subsurfaceAnisotropy`), and
coat (`coatWeight`/`coatRoughness`/`coatIor`, extracted from a `layer()`
whose top branch is a coat-shaped `dielectric_bsdf`). `MaterialEvaluation`
carries all of these correctly extracted from OSL — a real, verified gap
that used to mean either dropped data (subsurface, thin film) or actively
wrong results (sheen used to be added straight into `baseColor`; anisotropic
roughness was collapsed with `max()` instead of the standard isotropic
equivalent `sqrt(roughnessU * roughnessV)`; conductor treated its complex
IOR as if it were an albedo color directly instead of computing Schlick F0
from it). None of these fields are sampled as their own lobe by `Bsdf` yet
— that's real follow-up BSDF work, not a translation gap. Unsupported
inputs still produce one warning per compiled material — NoorRay does not
claim full MaterialX compliance.

### Reading the closure tree

NoorRay does not introduce a custom terminal closure. `standard_surface` and
`open_pbr_surface` already compose down to MaterialX's own standard closures
(`dielectric_bsdf`, `conductor_bsdf`, `oren_nayar_diffuse_bsdf`,
`translucent_bsdf`, `sheen_bsdf`, `subsurface_bssrdf`, `uniform_edf`,
`layer`/`add`/`mul`, ...), registered with OSL exactly as MaterialX defines
them (`src/libnoorray/MaterialX/noorray_closures.h`). The compiled program's
`Ci` therefore comes back as a real MaterialX closure tree; NoorRay walks it
(`MaterialClosureBridge.h`'s `noorrayExtractMaterialEvaluation`) and folds
each component's weighted contribution into one `MaterialEvaluation` --
scattering math itself is never delegated to OSL. Adding support for a
closure NoorRay doesn't yet read means adding a case to that walk and, if it
carries a field `MaterialEvaluation` doesn't have yet, extending that struct
(bump `NrMaterialEvaluationAbiVersion`).

Not every name that looks like a closure is one: `generalized_schlick_edf`
is not a distinct registered OSL closure at all --
`mx_generalized_schlick_edf.osl` implements it as ordinary OSL code
(`result = base * mx_fresnel_schlick(...)`), which compiles down to a plain
`ClosureMul` of whatever EDF closure `base` actually is. The generic
`ClosureColor::MUL` handling in the walk already covers it; registering it
as its own closure ID (as a previous version of this file did, reusing an
unrelated params struct "by coincidence") was dead code based on a wrong
assumption, since that closure ID is never actually emitted.

### Textures

`<image>` nodes are loaded through NoorRay's existing texture registry
(`Scene::getTextures()` / `HdNoorRayRenderParam::GetOrCreateTexture`) rather
than through OSL's own `TextureSystem`, so a MaterialX-driven material shares
the same loading, caching and CUDA texture objects as the legacy path. Color
images (`color3`/`color4` outputs) are decoded from sRGB; anything else
(roughness, normal, displacement, ...) stays linear -- this is inferred from
the `<image>` node's own declared output type, not from an authored
colorspace attribute.

The handle `MaterialXCompiler` bakes into the compiled program for each
image is a texture-registry *index*, not a raw `cudaTextureObject_t`. That
indirection matters: `Raytracer::updateTextures()` recreates every CUDA
texture object whenever the scene's texture list changes, so a raw handle
baked in at compile time would go stale the next time an unrelated texture
loads anywhere in the scene. `rs_texture` (`MaterialClosureBridge.h`) instead
looks the index up in the current texture array on every shade call, via a
pointer threaded through `ShaderGlobals::renderer` (otherwise unused in
NoorRay's bounded profile). Only compile-time-constant filenames resolve to a
texture this way; MaterialX itself declares `<image>`'s `file` input
`uniform="true"`, so this covers every image node produced by both ingestion
paths today (neither writes a graph that connects a computed value into a
filename).

### Hydra ingestion

hdNoorRay never touches Blender-USD's own embedded MaterialX (`libusd_ms.so`,
which it also links for ordinary pxr symbols) -- see
[Supported versions](#supported-versions) on why two independently-compiled
copies of the same namespaced MaterialX types must never trade live C++
objects. So instead of calling OpenUSD's `hdMtlx` (which would hand back a
`MaterialX::Document` constructed by Blender's copy), `src/hdnoorray/material.cpp`
walks the incoming network itself and rebuilds an equivalent document against
NoorRay's own vendored MaterialX, resolving each node's category and input
types from its nodedef in the same standard-library document
`MaterialXCompiler` compiles against. This is the same technique `hdMtlx`
uses internally, reimplemented against the other MaterialX instance.

The resource `HdSceneDelegate::GetMaterialResource()` hands `Sync()` can be
either `HdMaterialNetwork2` or the older, relationship-based
`HdMaterialNetworkMap` -- Blender's own Hydra integration uses the latter
(confirmed empirically; not an assumption), so `Sync()` converts unconditionally
via USD's own `HdConvertToHdMaterialNetwork2()` (a plain `Hd` utility, not
`hdMtlx` -- no MaterialX types cross the boundary here either) before the
graph walk, rather than maintaining two translators. Getting hdNoorRay's
render delegate to actually receive a real MaterialX network in that
conversion at all, rather than USD's own auto-derived `UsdPreviewSurface`
fallback, needed one more piece: see
[Defects fixed along the way](#defects-fixed-along-the-way-worth-knowing-about)'s
note on `GetMaterialRenderContexts()`.

hdNoorRay keeps one process-wide `MaterialXCompiler` (and therefore one
`OSL::ShadingSystem`), matching `MaterialXCompiler`'s own design intent,
rather than constructing one per material or per edit.

### Live editing

Editing a MaterialX-backed material in Blender re-invokes `HdNoorRayMaterial::Sync`,
which recompiles the whole document (MaterialX has no incremental
compilation NoorRay uses yet -- see [Current limitations](#current-limitations))
and swaps the new program into the *same* SBT slot the material already
owned, rebuilding the pipeline once. This is what keeps an editing session
from growing the pipeline (and leaking OptiX/CUDA resources) by one entry per
keystroke; only a brand-new material takes a fresh slot. A compile failure
mid-edit (an invalid intermediate graph state) never leaves the material
broken or stuck on a stale program: both hdNoorRay and the standalone app
fall back to the default MaterialX material -- a flat grey `open_pbr_surface`
-- and republish that instead.

A mesh with no material at all (no bound material Sprim) points its slots at
one shared native grey material slot owned by the render param, compiled once
on the render thread rather than once per mesh. `PublishFallbackMaterial()`
and material `Finalize()` use the same grey document, so an unassigned
material always renders grey instead of black.

## Defects fixed along the way (worth knowing about)

Two bugs made every MaterialX render -- standalone and Hydra alike -- fall
back to NoorRay's identical legacy default material regardless of the
document's actual content, silently. Both were invisible to the existing
e2e test (it only checked image dimensions, never pixel content) and are
worth understanding because their failure mode (consistent, plausible,
*wrong* output with no error) is exactly the kind that doesn't announce
itself:

* **The OptiX megakernel once omitted MaterialX evaluation.** The evaluation
  path used to sit behind a compile definition that was not passed to the
  hand-written `nvcc --ptx` custom command which builds the actual GPU
  megakernel. The condition was therefore false for the one compile that
  mattered, and `Material::materialxProgramIndex` was silently ignored.
  MaterialX is now unconditional, so the host and custom GPU compilation
  paths cannot diverge this way.
* **The closure-tree walk was recursive**, and OptiX's device compiler
  rejects recursive call graphs outright (a hard compile error, "Found call
  graph recursion..."). This had never been exercised before the fix above,
  since nothing had ever compiled that code path into a real OptiX module.
  Rewritten as an iterative walk over an explicit fixed-depth stack in
  `MaterialClosureBridge.h`.

Fixing both was verified with two MaterialX documents identical except for
`base_color` (`tests/e2e/MaterialXRenderTest.cpp`'s
`"MaterialX base_color actually reaches the rendered pixels"` case) --
before the fix, their renders were byte-identical; after, each is
correctly dominated by its own color. Also worth noting:
`MaterialXCompiler` now sets the `max_optix_groupdata_alloc` OSL attribute
(matching OSL's own reference CUDA renderer), which is unrelated to either
bug above but is the correct, documented way to make a compiled group
self-contained rather than relying on a renderer-supplied groupdata buffer
NoorRay never allocates.

**A fourth bug, specific to the Hydra/Blender path, was more interesting**:
every material -- MaterialX-authored or not -- arrived at
`HdNoorRayMaterial::Sync` as a flattened `UsdPreviewSurface` stub carrying
schema *defaults* (`diffuseColor (0.8, 0.8, 0.8)`, `roughness 0.4`, ...),
never the real node graph, regardless of `bl_use_materialx`. The first
hypothesis -- that Blender's own MaterialX export is broken on this
version -- turned out to be wrong: Blender's own bundled Hydra Storm engine
(`scripts/addons_core/hydra_storm/engine.py`), using the exact same
`bl_use_materialx = True` and the same minimal registration pattern hdNoorRay
already used, renders the real color correctly. The actual cause:
`HdRenderDelegate::GetMaterialRenderContexts()` -- which tells Hydra's
material-network builder which network *flavor* a delegate wants, in
descending preference -- defaults to `{""}` (universal/preview) when not
overridden. USD's own `UsdMtlxRead()` (which is what Blender's MaterialX
export actually feeds the stage through, confirmed against Blender's PR
[#111765](https://projects.blender.org/blender/blender/pulls/111765)) always
synthesizes *both* the real MaterialX network *and* an auto-derived
UsdPreviewSurface fallback for exactly this case -- a delegate that never
asks for `"mtlx"` only ever sees the fallback, no matter what Blender
exported. `HdStRenderDelegate` (Storm) overrides this; `HdNoorRayRenderDelegate`
did not. Fixed with a two-line override
(`renderDelegate.h`/`.cpp`) returning `{"mtlx", ""}` -- `"mtlx"` is the
standard token used across Hydra-consuming MaterialX renderers (Storm,
RenderMan's HdPrman, Arnold's HdArnold, ...). `bl_use_materialx` now stays
`True` and does what it was always meant to.

That fix immediately surfaced three more, previously unreachable because no
real MaterialX network had ever reached `CompileMaterialXNetwork` before:

* **The graph translator always set an explicit output string on every
  connection**, but MaterialX only accepts one on a connection to a node
  whose nodedef declares *more than one* output -- doing it unconditionally
  fails validation ("Multi-output type expected in port connection") for
  the overwhelmingly common case of connecting to an ordinary single-output
  node (Hydra names a connection's source output "out" even then, which
  is not the same thing). Fixed by looking the upstream node's own declared
  output count up before deciding whether to call `setOutputString` at all.
* **Recompiling an edited material collided with itself.** `MaterialXCompiler`
  keeps one process-wide `OSL::ShadingSystem` by design (this file's own
  "Hydra ingestion" section), which keeps every shader it is ever handed by
  name for the compiler's lifetime -- so reusing the same MaterialX element
  name on every edit (which recompiling the same material naturally does)
  got the second and every subsequent compile rejected outright ("OSL
  shading system rejected the compiled shader"), silently falling back to
  the last-good program via the same compile-failure path the "Live
  editing" section describes. Fixed by suffixing the shader name with a
  monotonic per-compile counter; the old shader's memory is not reclaimed,
  a bounded-by-edit-count leak rather than a correctness bug (see
  [Current limitations](#current-limitations)).
* **A second family of OSL texture-option setters was missing from the
  device build** (`osl_texture_set_missingcolor_alpha` first, then the rest
  of `optexture.cpp`'s `osl_texture_set_*`/`osl_texture_decode_*` family
  once checked proactively) -- for the same reason `rs_texture`'s own
  comment already documents for the four found earlier:
  `OSL_TEXTURE_SET_HOSTDEVICE` is unconditionally host-only in this OSL
  version, so the renderer is expected to supply every one of them itself.
  NoorRay now implements the complete family in `MaterialClosureBridge.h`,
  mirroring each function's host body (including optexture.cpp's own
  `__CUDA_ARCH__` fallbacks where it has them -- e.g. string-keyed wrap/interp
  setters resolve to a default rather than a real string lookup, matching
  upstream exactly).

Verified end to end against Blender itself (not just the CLI): a Principled
BSDF with an Image Texture node feeding Base Color renders the actual
checker pattern on the sphere -- Blender exports it as a real
`open_pbr_surface` graph (`ND_open_pbr_surface_surfaceshader`) with
`ND_image_color4`/`ND_convert_color4_color3`/`ND_texcoord_vector2` nodes,
which `CompileMaterialXNetwork` translates, compiles and renders
unmodified. Editing the Base Color and re-rendering correctly changes the
result (verified by literally rendering red, then blue, from the same
in-memory material and diffing the output).

**A sixth bug surfaced only under a real production scene** (Blender's
"Classroom" demo -- 83 materials, ~130k triangles), never the hand-written
test documents this was validated against up to that point: an "illegal
memory access" crash, reliably reproducible, that `compute-sanitizer
--tool memcheck` traced to a "Warp illegal address" inside OSL's own
`osl_transformc` (colorspace conversion). The cause: `rend_get_userdata`
(the renderer-service hook OSL's device code calls for anything it treats
as "ask the renderer") was a stub that always `return false`d *without
writing to its output pointer* -- but OSL's own `get_colorsystem()`
(`opcolor.cpp`) dereferences that output pointer unconditionally, ignoring
the return value entirely. Every node that touches color space conversion
(common; several stdlib color operations funnel through `transformc`) read
a garbage stack pointer as a `ColorSystem*` and crashed. Fixed by
implementing the one real request OSL's CUDA path makes of this hook --
`name == OSL::Hashes::colorsystem` -- properly:
`MaterialXCompiler::getColorSystemBlob()` pulls OSL's own ColorSystem data
out of the `OSL::ShadingSystem` (`getattribute("colorsystem", ...)`,
mirroring `external/osl/src/testrender/optixraytracer.cpp`'s own upload of
this exact data byte for byte), `Raytracer::uploadMaterialXColorSystem()`
uploads it once, and `rend_get_userdata` reads it back through
`KernelParams params` -- the same `pipelineLaunchParamsVariableName`
mechanism the main megakernel uses, re-declared in the closure-bridge
module (a separate PTX module OptiX's pipeline linker unifies with the
main one by that name, not ordinary C++ linkage).

The same production scene also caught a closure-tree bump allocator with
no bounds checking: `closurePool`, a fixed-size local array, had every
`osl_allocate_closure_component`/`osl_add_closure_closure`/... write to it
unconditionally, so a material composing enough layered BSDFs (a handful
of the classroom's materials chain multiple `mix_bsdf`/`layer` nodes deep)
overflowed it and corrupted adjacent GPU thread-local memory -- a second,
independent cause of the same class of crash. Now bounds-checked (dropping
the excess closure rather than corrupting memory) and given more headroom
(`NrClosurePoolBytes` 4096 -> 8192).

Both were found the same way: build a `.blend` more complex than anything
tried before, watch it crash, and use `compute-sanitizer --tool memcheck`
(NVIDIA's CUDA memory-checker, works transparently under `blender
--background --python ...`) to get a precise device-side stack trace
instead of guessing from the host-side "illegal memory access" `cudaStreamSynchronize`
always reports regardless of where the fault actually happened.

**A seventh bug surfaced when hdNoorRay's own build switched its host
compiler from GCC to clang: the plugin failed to load into Blender at
all**, with USD's plugin registry reporting an undefined symbol inside
`HdRenderDelegate`'s constructor and "Couldn't find plugin for id
HdNoorRayRendererPlugin". `nm -D` on both `libhdnoorray.so` and Blender's
own `libusd_ms.so` showed the *same* mangled symbol name resolving to two
different template instantiations of `TfHashMap`'s allocator
(`std::allocator<std::pair<const TfToken, VtValue>>` vs. plain
`std::allocator<VtValue>`). The cause: `pxr/base/arch/defines.h` only
enables pxr's GNU `hash_map`/`hash_set`/`SmallVector` extensions `#if
defined(ARCH_COMPILER_GCC)` -- true for whatever compiled Blender's own
`libusd_ms.so`, false under clang, so `TfHashMap` silently fell back to a
`std::unordered_map`-based implementation with a different ABI. Since
hdNoorRay is a plugin loaded *into Blender's own process*, it must
instantiate these pxr types exactly the way Blender's own binary does,
regardless of which compiler builds hdNoorRay itself. Fixed by forcing
`ARCH_HAS_GNU_STL_EXTENSIONS` as a compile definition on the `hdnoorray`
and `NoorRayHydraUsd` targets (`src/hdnoorray/CMakeLists.txt`) -- this
doesn't conflict with `arch/defines.h`'s own `#define`, which never
executes under clang in the first place. Worth checking again for any
*other* pxr header gated the same way if a future symbol-not-found error
looks like this one.

### OSL bitcode must match its own reader's LLVM version

**An eighth bug, only reachable once the seventh was fixed, crashed the
first MaterialX shader compile with a SIGSEGV inside
`llvm::DataLayout::reset`** -- called from LLVM's own `libLLVM.so.19.1`,
several frames below OSL's `BackendLLVM::run()` and
`ShadingSystemImpl::optimize_group()`. Register state at the crash
(`rdi=0x120`, a small integer rather than a pointer) pointed at a
null-plus-member-offset access: some object holding the `DataLayout` being
reset was null. The actual null pointer, traced by reading
`external/osl/src/liboslexec/llvm_instance.cpp`'s `BackendLLVM::run()`
(`OSL_LLVM_CUDA_BITCODE` branch, ~line 2182): `shadeops_module =
ll.module_from_bitcode(shadeops_cuda_llvm_compiled_ops_block, ...)`
followed immediately by `shadeops_module->setDataLayout(...)` with no null
check in between. `module_from_bitcode()` had silently failed to parse
the embedded CUDA shadeops bitcode.

The reason it failed: `cmake/OSL.cmake` generated that embedded bitcode
with **clang++-22** (LLVM 22), needed because CUDA 13's headers require a
newer clang than 19 to compile at all -- but `liboslexec` itself is linked
against **LLVM 19** (required, since OSL 1.14.7's C++ API predates LLVM
22's breaking changes), and parses that embedded bitcode with LLVM 19's
own bitcode reader at run time. LLVM's bitcode format is not reliably
readable across major-version jumps; LLVM 19's reader choked on LLVM
22-generated IR syntax (confirmed directly: feeding `llvm-as-19` a minimal
`declare void @llvm.lifetime.start.p0(ptr captures(none))` -- valid LLVM
22 syntax replacing the older `nocapture` attribute -- reproduces `error:
expected ')' at end of argument list` standalone).

The earlier workaround matched the bitcode generator to LLVM 19 and used a
separate CUDA 12.5 toolkit because clang-19 could not parse CUDA 13 headers.
Upgrading the vendored OSL to 1.15.5 removed its LLVM-22 API incompatibility,
so the host library, bitcode tools, and CUDA 13 build now all use LLVM/clang
22. The separate OSL CUDA toolkit is no longer needed.

## Current limitations

* **No incremental/interactive parameter updates.** Every edit recompiles
  MaterialX -> OSL -> PTX -> OptiX module from scratch; there is no fast path
  that only rewrites a value into an already-compiled program's parameter
  block (the `interactiveParams` argument `noorrayEvaluateMaterial` already
  threads through for this is always null today). Fine for correctness, not
  free: expect a compile's cost (milliseconds to low hundreds of
  milliseconds depending on graph size) on every slider drag.
* **A deleted material, or a material whose terminal changes to something
  NoorRay can't translate, leaks one OptiX program slot.** Nothing calls the
  equivalent of `replaceMaterialXProgram()` from `HdNoorRayMaterial::Finalize()`;
  there is no "release this slot" API yet, only "replace this slot" and
  "clear everything". Bounded by the number of such transitions in a
  session, not by editing frequency.
* **Every edit leaves its old compiled shader resident in `OSL::ShadingSystem`.**
  Each recompile uses a fresh, uniquified shader name specifically so it
  doesn't collide with the previous one (see [Defects fixed along the
  way](#defects-fixed-along-the-way-worth-knowing-about)) -- but OSL never
  reclaims a shader once loaded, so a long editing session accumulates one
  retained .oso per edit for the life of the process. Bounded by edit count
  within one session, not unbounded, but real for a long interactive session.
* **No named coordinate systems, primvar readers beyond UV, or trace/pointcloud
  queries.** `MaterialClosureBridge.h`'s renderer-service stubs for these are
  still no-ops (see the file's own comments); a graph that depends on them
  compiles but reads back identity/empty values.
* **Coat, anisotropy, thin film, sheen and subsurface are correctly read into
  `MaterialEvaluation` but not yet shaded** -- `Bsdf` has no anisotropic,
  thin-film, sheen or coat lobe, and no BSSRDF, so these fields are carried
  for future work rather than dropped or faked. See
  [Supported surface profile](#supported-surface-profile).
* **Volume shaders (`anisotropic_vdf`/`medium_vdf`) are registered so graphs
  using them compile, but the medium itself is not implemented** -- no
  participating-media/volumetric rendering exists yet; these closures are
  silently ignored by the walk (a deliberate no-op, not a bug).

## Phase 0 findings

The spike answered the questions that had to be settled before any production
material code was written.

### Proven

**MaterialX loads and validates NoorRay's custom terminal.** The pinned 1.39.4
build loads 54 standard-library documents plus NoorRay's own 2, and a document
using `noorray_principled` validates.

**MaterialX's stock OSL generator is sufficient.** A graph containing a
`position` node, a `multiply`, a `noise3d` and the NoorRay terminal generates
4490 bytes of valid OSL that calls MaterialX's own `mx_noise3d_float`
implementation. NoorRay does not need to write a shader generator — only a
terminal customization layer.

**One call evaluates the whole graph.** Generation produces a single shader
entry point whose body is a straight-line statement list, not a series of
per-node calls.

**Parameter edits are independent of graph topology.** Editing two values in
the source document changes only their default values in the generated shader
signature; the shader body is byte-identical. With OSL interactive parameters
those defaults are overridden at runtime anyway. This confirms the intended
hash split: topology hash over the graph, instance hash over the values.

(The feasibility-spike binary that produced these findings has since been
removed — production code, `NoorRayMaterialXCompiler`, fully replaced it.
Reproduce the same generation step with any real `.mtlx` scene, e.g. `./NoorRay
--cli --scene tests/assets/materialx_noise_sphere.nrscene --spp 8 --output
/tmp/out.exr` with `NR_MATERIALX_DEBUG=1` set.)

### Blocked

**OSL device-code generation could not be exercised.** No OSL build with OptiX
support exists on the development machine, and the distribution package is
compiled with `OSL_USE_OPTIX 0`. Building one requires the LLVM, clang and
OpenImageIO development packages, which need root. Until that is resolved, the
following gate items are unverified:

* MaterialX noise executing on the GPU through OSL.
* OSL's texture and runtime support being packageable with NoorRay.
* The callee-side register and stack cost of a real OSL-generated callable.

The OptiX half of the spike was measured separately — see below.

### Measurements

See [Diagnostics](#diagnostics) for how to reproduce these.

<!-- MEASUREMENTS -->

## Diagnostics

**OptiX compile feedback and stack sizes.** `NR_OPTIX_LOG_LEVEL` raises the
OptiX log callback; level 4 includes per-entry-function register counts and
pipeline statistics. NoorRay additionally logs the computed stack sizes when the
variable is set.

```bash
NR_OPTIX_LOG_LEVEL=4 ./build/<dir>/NoorRay --cli \
    --scene tests/assets/simple_sphere.nrscene --spp 8 --output /tmp/out.exr
```

**Generated OSL.** Set `NR_MATERIALX_DEBUG=1` to have hdNoorRay's material
ingestion (`src/hdnoorray/material.cpp`) log the translated MaterialX graph,
Hydra network contents, and nodedef resolution for every material Sync.

**Register pressure.** Do not respond to a regression by setting
`maxRegisterCount`. Investigate live ranges, the size of the evaluation result,
stack use and spills first; a register cap hides the problem rather than fixing
it.

## Open questions for later phases

* A MaterialX graph carries a scalar IOR, while NoorRay's dielectrics use
  Sellmeier coefficients. Mapping one onto the other is unresolved.
* `Surface::fromHit` decides whether to reconstruct a tangent from a fixed
  texture index (`normalIndex >= 0`). Phase 6 replaces that with per-program
  feature flags.
