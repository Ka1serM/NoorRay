# Vendor and specialize OpenPBR BSDF lobe math into NoorRay

## Context

NoorRay currently depends on Adobe's OpenPBR BSDF reference implementation
(`external/openpbr-bsdf/`, a git submodule) only for its precomputed
energy-compensation lookup tables (`src/Mesh/OpenPbrEnergy.h`). The actual BSDF
lobe math — diffuse, GGX microfacet specular, dielectric transmission with
dispersion — is hand-rolled in `src/Mesh/Material.h` (537 lines), loosely
inspired by OpenPBR but not a direct port.

The user previously tried a from-scratch rewrite in a separate `src/Bsdf/`
module (commit `74307b9`, "wip: custom simplified bsdfs with energy
conservation") and then reverted it (`112e135`) — they weren't satisfied with
reimplementing the BSDF math from scratch. This time, the goal is different:
**directly transcribe OpenPBR's validated lobe formulas** (not reinvent them),
but rewrite them to operate natively on NoorRay's types — `SampledSpectrum`/
`SampledWavelengths` (4-wavelength hero sampling, see `src/Raytracing/Spectrum.h`)
and `glm::vec3` — instead of OpenPBR's generic RGB-vec3 + multi-language
macro/interop layer (`interop/openpbr_interop_*.h`) and its many unused material
layers (coat, fuzz, subsurface, thin-film, thin-wall, anisotropy). That generic
machinery is the "performance overhead" to cut: NoorRay's material model only
ever needs base diffuse + specular (dielectric/metal blend) + transmission.

Net effect: same validated OpenPBR physics and energy-conservation behavior,
but as lean, direct, `NR_CPU_GPU`-annotated C++ specialized to exactly what
NoorRay uses — no virtual dispatch, no heap allocation, no macro dispatch, no
per-call RGB<->spectral conversion.

## Default decisions (flag if you want different)

- **Diffuse roughness**: not added as a new `Material` field. Hardcode
  `diffuse_roughness = 0` (pure energy-conserving Lambert + specular
  compensation) to avoid touching the material schema/serialization. Can be
  added later as an additive field if wanted.
- **Dispersion**: NoorRay's existing 3-term Sellmeier model
  (`src/Raytracing/Sellmeier.h`) fully replaces OpenPBR's simpler Abbe-number/
  Cauchy dispersion (`impl/openpbr_dispersion_utils.h`). That file is not
  ported — Sellmeier is a strict superset and is already wired into the
  transmission path.
- **Metal Fresnel**: keep NoorRay's existing Schlick-with-albedo-as-F0 model.
  OpenPBR's F82-tint metal model is out of scope (NoorRay has no separate
  metal edge-tint parameter — porting it would be new scope, not a straight
  port).
- **Anisotropy**: not ported — NoorRay has no anisotropic roughness parameter.
- **Cleanup**: delete the old hand-rolled implementations from `Material.h`
  immediately once the new code passes tests, rather than leaving dead code
  behind a flag (matches the repo's evident preference from the
  `74307b9`/`112e135` history for clean, non-speculative code).

## What gets vendored vs. what stays a submodule dependency

- **Submodule (`external/openpbr-bsdf`) stays**, but only for
  `impl/data/openpbr_*_energy_complement_data.h` and related table headers —
  raw LUT data `#include`d verbatim by the relocated energy-table wrapper. No
  lobe/orchestration/interop logic is included from the submodule at runtime
  after this change.
- **Fully vendored as new hand-written source** under a new `src/Bsdf/`
  directory: diffuse EON BRDF, isotropic GGX/VNDF/Smith math, Fresnel,
  reflection/transmission coefficient blending, and sample/eval/pdf
  orchestration — transcribed from OpenPBR's reference formulas but
  restructured for `SampledSpectrum`/`glm::vec3` and trimmed of unused layers.

CMake wiring for the `openpbr` INTERFACE target (`CMakeLists.txt` ~lines
60-64, ~265, ~318, and `tests/CMakeLists.txt`) stays unchanged — it's still
needed for the data-table include path.

## New files under `src/Bsdf/`

All header-only, `NR_CPU_GPU`/`NR_GPU` annotated (`src/CUDA/Annotations.h`
convention), no virtuals/heap/exceptions, matching the existing style in
`Material.h`/`Bsdf.h`.

1. **`src/Bsdf/OpenPbrFresnel.h`** — per-wavelength dielectric Fresnel +
   Schlick metal term, transcribed from OpenPBR's `openpbr_fresnel`/
   `openpbr_schlick` (functionally identical to Material.h's existing
   `fresnelDielectric`/`fresnelSchlick*`, formalized as free functions
   decoupled from the `Material` class).
2. **`src/Bsdf/OpenPbrMicrofacet.h`** — isotropic GGX distribution + Smith
   G1/G2 + VNDF sampling, transcribed from
   `impl/openpbr_vndf_microfacet_distribution.h` +
   `impl/openpbr_lobe_utils.h` specialized to `alpha.x==alpha.y` (NoorRay has
   no anisotropic roughness). Formalizes Material.h's existing
   `sampleGgxVndfLocal`/`distributionGgx`/`smithG1Ggx`.
3. **`src/Bsdf/OpenPbrEnergyCompensation.h`** — relocated `OpenPbrEnergy.h`
   table wrapper plus the two derived quantities diffuse/specular lobes need
   (`specularEnergyCompensation(...)`, `dielectricMultiScatterTerm(...)`),
   consolidating all OpenPBR-derived energy-table math in one place.
4. **`src/Bsdf/OpenPbrDiffuseLobe.h`** — EON diffuse BRDF, transcribed from
   `impl/openpbr_diffuse_lobe.h`, generalized from RGB `vec3` to per-wavelength
   `SampledSpectrum` (loop of `NrSpectrumSamples=4` instead of 3 RGB
   channels). Cosine-hemisphere sampling stays as Material.h's existing
   `sampleDiffuseDirection`.
5. **`src/Bsdf/OpenPbrSpecularTransmissionLobe.h`** — the comprehensive
   microfacet lobe, trimmed: GGX+Fresnel-blended reflection (dielectric/metal)
   eval/pdf/sample, and dielectric transmission eval/pdf/sample with
   per-wavelength Sellmeier IOR (replacing OpenPBR's RGB+Abbe-dispersion
   channel loop with a 4-wavelength Sellmeier loop). Subsumes
   `distributionGgx`, `smithG1Ggx`, `geometrySmith`, `fresnelDielectric`,
   `pdfGgxReflection`, `sampleDielectric` from the current `Material.h`.
6. **`src/Bsdf/OpenPbrMaterialLobes.h`** — orchestration layer combining
   diffuse + specular + transmission with the same stochastic lobe-selection
   heuristic Material.h uses today (Fresnel/luminance-weighted specular
   probability, transmission-vs-opaque split by `transmission` weight). Plays
   the role of OpenPBR's `openpbr_prepare_lobes`/`eval`/`sample`/`pdf`
   orchestration, trimmed to base+specular+transmission only.

## Material.h changes

Keep `sampleBsdfSpectral`, `evaluateDirectSpectral`, `pdfDirectSpectral`
signatures byte-for-byte identical (verify via `src/Kernels/Shade.cu`
call sites) so no caller changes. Their bodies become thin wrappers: resolve
textured scalar params exactly as today, call into `OpenPbrMaterialLobes.h`
free functions for the eval/sample/pdf math, keep
`rgbAlbedoToSpectrum`/`rgbIlluminantToSpectrum` emission/albedo upsampling
untouched (not BSDF-lobe concerns).

Delete from `Material.h` once the new path is verified: `evaluateOpaqueSpectral`,
`pdfGgxReflection`, `sampleDielectric`, `distributionGgx`, `smithG1Ggx`,
`geometrySmith`, `fresnelDielectric`, `fresnelSchlick*`, `sampleGgxVndfLocal`,
`sampleGgxHalfVector` (all moved to `src/Bsdf/`). Keep `buildBasis`,
`clampShadingNormal`, `shadingNormalCorrection` in `Material.h` unchanged
(geometry-only, unrelated to OpenPBR lobes).

## Tests

Existing suite stays wired via `tests/CMakeLists.txt` (no target/link changes
needed, just call-site updates in test bodies):

- `tests/BsdfTestFixture.h` — switch `integrateOpaque` to call the new
  `OpenPbrDiffuseLobe.h`/`OpenPbrSpecularTransmissionLobe.h` free functions
  directly instead of `Material::evaluateOpaqueSpectral`.
- `tests/GgxBsdfTest.cpp` — call `OpenPbrMicrofacet.h`'s new free functions
  instead of `Material::pdfGgxReflection`.
- `tests/DielectricBsdfTest.cpp` — call the new transmission-lobe sample
  function instead of `Material::sampleDielectric`; same invariants (finite
  pdf/weight, correct hemisphere sign, eta consistency, ~4% Fresnel
  reflectance at normal incidence for ior=1.5).
- `tests/WhiteFurnaceTest.cpp` — no source changes; full end-to-end
  render-based furnace test through `RenderTestFixture`, the strongest
  regression gate. Record baseline numbers from current HEAD before starting,
  compare after.
- Add one new **transmission furnace test**: purely transmissive dielectric
  slab (`transmission=1`, `metallic=0`) at normal incidence, verify
  reflection+refraction probability-weighted energy sums to ~1 — exercises the
  new per-wavelength Sellmeier-driven transmission coefficient blending.
- Add one new **combined-lobe PDF-normalization test**: integrate
  `pdfDirectSpectral` over the hemisphere for the full diffuse+specular
  stochastic-selection strategy, confirm it sums to ~1 (extends what
  `GgxBsdfTest.cpp` does for GGX alone).

## Migration order (buildable at each step)

1. Confirm scene/material loading code (`src/Mesh/*` deserialization) doesn't
   assume `Material`'s current field layout in a way that matters (only
   relevant if diffuseRoughness is added later).
2. Add `OpenPbrFresnel.h` + `OpenPbrMicrofacet.h` — pure math, no `Material.h`
   dependency. Verify standalone CPU and NVCC compilation.
3. Add `OpenPbrEnergyCompensation.h` (relocate `OpenPbrEnergy.h` content),
   verify table includes still resolve via the `openpbr` INTERFACE path.
4. Add `OpenPbrDiffuseLobe.h`. Cross-check numerically against old
   `Material::evaluateOpaqueSpectral` (diffuse-only: metallic=0, specular=0)
   for a grid of albedo/view-angle combos before wiring anything else in.
5. Add `OpenPbrSpecularTransmissionLobe.h`, reflection branch first
   (cross-check against old `evaluateOpaqueSpectral`'s specular term), then
   the transmission branch (cross-check against old `sampleDielectric` +
   `evaluateDirectSpectral`'s transmission half).
6. Add `OpenPbrMaterialLobes.h` orchestration (not yet wired into `Material`).
7. Swap `Material::sampleBsdfSpectral`/`evaluateDirectSpectral`/
   `pdfDirectSpectral` bodies to call the new orchestration layer; delete the
   old inline implementations from `Material.h`.
8. Update `BsdfTestFixture.h`, `GgxBsdfTest.cpp`, `DielectricBsdfTest.cpp` to
   the new free functions; add the transmission-furnace and combined-PDF
   tests.
9. Run full test suite (`opaque_bsdf_test`, `ggx_bsdf_test`,
   `dielectric_bsdf_test`, `white_furnace_test` sphere+teapot); compare
   against the pre-refactor baseline recorded before step 2.
10. Full CUDA/OptiX build verification (`NoorRayCuda` target + PTX custom
    commands for `Extend.cu`/`Connect.cu`) — new headers must be NVCC-clean.
11. Manual visual smoke check: render metallic, rough-dielectric, and
    dispersive-glass test scenes and eyeball for regressions.

## Verification

- `ctest` (or direct test binaries) for `opaque_bsdf_test`, `ggx_bsdf_test`,
  `dielectric_bsdf_test`, `white_furnace_test` — all must pass with
  furnace/energy-conservation/PDF-normalization statistics matching (within
  noise of) the pre-refactor baseline.
- Full CUDA build must succeed (`NoorRayCuda` target).
- Manual render comparison of a metallic scene, a rough dielectric scene, and
  a dispersive-glass scene against pre-refactor renders (via `/run` or
  `verify` skill once implementation is underway).

### Critical files
- `src/Mesh/Material.h`
- `src/Mesh/OpenPbrEnergy.h`
- `src/Raytracing/Bsdf.h`
- `src/Raytracing/Sellmeier.h`
- `src/Raytracing/Spectrum.h`
- `src/Kernels/Shade.cu`
- `external/openpbr-bsdf/impl/openpbr_diffuse_lobe.h`
- `external/openpbr-bsdf/impl/openpbr_comprehensive_microfacet_lobe.h`
- `external/openpbr-bsdf/impl/openpbr_reflection_transmission_coefficient.h`
- `external/openpbr-bsdf/impl/openpbr_lobe_utils.h`
- `external/openpbr-bsdf/impl/openpbr_vndf_microfacet_distribution.h`
- `tests/BsdfTestFixture.h`, `tests/CMakeLists.txt`
- `CMakeLists.txt`
