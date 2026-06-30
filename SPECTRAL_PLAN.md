# Spectral Rendering Implementation Plan — NoorRay

**Goal:** Replace per-path `glm::vec3` RGB with `SampledSpectrum` (N=4 wavelengths) using hero wavelength sampling (Wilkie et al. 2014) and Jakob–Hanika sigmoid-polynomial RGB upsampling (identical to pbrt-v4). Keep code changes minimal — shaders/kernels change, scene loading/UI does not.

**Baseline:** ~300 fps / ~2.1 ms on RTX 3080 Ti (RGB).  
**Reference:** pbrt-v4 `util/color.h`, `util/spectrum.h`, `cmd/rgb2spec_opt.cpp`.

---

## 1. Background

### Hero Wavelength Sampling (Wilkie et al. 2014)
Each path carries a package of **C = 4** wavelengths. One is the *hero* λ₀ sampled from p(λ), the others are stratified companions:

```
λᵢ = (λ₀ − λ_min + i/C · (λ_max − λ_min)) mod (λ_max − λ_min) + λ_min
```

Because shifting is deterministic the MIS balance-heuristic weight for wavelength j simplifies to:

```
wⱼ(X, λⱼ) = 1 / (C · p(λⱼ) · p(X|λⱼ))   [after cancellation in balance heuristic]
```

In practice this means: all four wavelengths ride the **same** spatial path (same ray directions — directed by the hero), each accumulated independently. At dispersive surfaces, call `terminateSecondary()` and continue with the hero only.

The hero λ₀ is drawn from `sampleVisible(u)` (already in `Spectrum.h:L144`) which importance-samples toward the photopic peak (~555 nm). This is already the state-of-the-art low-noise choice — uniform sampling wastes samples on low-sensitivity regions.

### Upsampling — Jakob & Hanika Sigmoid Polynomial
Input: sRGB triplet (albedo, light color, emission, …).  
Output: three coefficients (c₀, c₁, c₂) such that:

```
f(λ) = S(c₀λ² + c₁λ + c₂)    S(x) = 1/(1 + exp(-x))  [logistic / sigmoid]
```

This is a smooth reflectance spectrum in [0,1] that round-trips through the CIE color matching functions back to the original sRGB value. The coefficients are precomputed offline into a **64³ × 3-channel** lookup table (one entry per position in the sRGB cube), stored on GPU. At runtime: trilinear interpolation → evaluate polynomial at each λᵢ. Memory: **9 MB**.

pbrt-v4 already ships the baked table (`src/pbrt/util/spectrum.cpp` → `sRGBToSpectrumTable_Data`). We copy it verbatim.

---

## 2. What does NOT change
- Ray traversal (Extend.cu, Connect.cu structure)
- Wavefront queue logic (`appendRayWarp`, occupancy bookkeeping)
- BSDF geometry (GGX microfacet, cosine hemisphere, normal maps, opacity)
- Camera ray generation geometry
- Adaptive sampling convergence test (uses luminance Y, which we already have)
- UI, scene importer, asset loading (except adding a coefficient conversion step)
- Vulkan display path

---

## 3. Struct Changes

### 3.1 `Types.h` — PathState

**Before:**
```cpp
struct alignas(16) PathState {
    glm::vec3 throughput;   // RGB
    glm::vec3 radiance;     // RGB
    uint32_t rngState, depth, flags, packedCounters, lastBsdfPdfBits, _pad0;
};
// ~64 bytes
```

**After:**
```cpp
struct alignas(16) PathState {
    SampledSpectrum throughput;   // 4 floats = 16 bytes
    SampledSpectrum radiance;     // 4 floats = 16 bytes
    uint32_t rngState, depth, flags, packedCounters, lastBsdfPdfBits;
    uint32_t _pad0;
};
// 48 bytes (same as before — fits 16-byte alignment)
```

Wavelengths live in a **separate per-pixel buffer** (see §3.5) so PathState stays lean.

### 3.2 `Types.h` — ShadowWorkItem

**Before:**
```cpp
glm::vec3 contribution;  // 12 bytes
```
**After:**
```cpp
SampledSpectrum contribution;  // 16 bytes
```

`ShadowWorkItem` grows from 48 → 48 bytes still (contribution was padded to 16 anyway because `uint32_t sampleIndex` follows — recheck alignment). Actual change: `glm::vec3 contribution` + `uint32_t sampleIndex` → `SampledSpectrum contribution` + no extra field needed. Struct stays at 64 bytes aligned.

### 3.3 `Bsdf.h` — BsdfSample

**Before:**
```cpp
glm::vec3 weight, albedo, emission;
```
**After:**
```cpp
SampledSpectrum weight;
SampledSpectrum albedo;
SampledSpectrum emission;
```

The direction, event, metallic, roughness, specular, transmission scalars are unchanged.

### 3.4 `LightSample.h`

**Before:**
```cpp
glm::vec3 radiance;
```
**After:**
```cpp
SampledSpectrum radiance;
```

### 3.5 `WavefrontQueues` — New wavelength buffer

```cpp
struct WavefrontQueues {
    // existing fields ...
    SampledWavelengths* wavelengths{};   // [pixelCount], written once in Generate
};
```

Allocated in `Raytracer.cpp` alongside `pathStates` (same lifetime, same size class). At 8 floats × 4 bytes × (width × height) = 32 bytes/pixel — negligible vs. existing buffers.

### 3.6 `SceneData.h` — GPU upsampling table pointer

```cpp
struct GpuSceneData {
    // ... existing fields ...
    const float* rgbToSpectrumCoeffs{};   // pointer to 9 MB sRGB→spectrum table on device
};
```

---

## 4. New Files

### 4.1 `Raytracing/RgbToSpectrum.h` (CPU+GPU)

```
NR_CPU_GPU SampledSpectrum rgbToSpectrum(
    glm::vec3 rgb,
    const SampledWavelengths& wl,
    const float* coeffTable);   // device pointer to 64^3 table
```

Implementation:
1. Clamp rgb to [0,1].
2. Determine dominant channel → selects which of the three sub-tables to use.
3. Trilinear lookup in 64³ table → (c₀, c₁, c₂).
4. For each λᵢ: evaluate `sigmoid(c₀·λᵢ² + c₁·λᵢ + c₂)`.

For **unbounded** values (emission, light radiance) use `RGBUnboundedSpectrum` approach from pbrt-v4: extract luminance scale, normalize rgb, look up polynomial, multiply result by scale. Two helper functions:

```cpp
NR_CPU_GPU SampledSpectrum rgbAlbedoToSpectrum(glm::vec3 rgb, ...);    // rgb in [0,1]
NR_CPU_GPU SampledSpectrum rgbIlluminantToSpectrum(glm::vec3 rgb, ...); // rgb unbounded, D65
```

### 4.2 `Raytracing/SpectralBsdf.h`

A spectral wrapper around the existing BSDF math in `Material.h`. Instead of returning `glm::vec3`, it:
- Takes `const SampledWavelengths& wl` and `const float* coeffTable`
- Evaluates albedo/emission textures as RGB, converts to `SampledSpectrum` via `rgbAlbedoToSpectrum`
- Returns `BsdfSample` with spectral `weight`, `albedo`, `emission`

The directional sampling logic (GGX, cosine hemisphere) is **unchanged** — geometry is wavelength-independent for non-dispersive materials.

### 4.3 `src/cpu/BakeRgbToSpectrumTable.h/.cpp`

CPU-side code to:
1. Load the precomputed table data from pbrt-v4's `sRGBToSpectrumTable_Data` (copy the `float[]` array verbatim — it's 64³×3×3 floats as defined in pbrt-v4's `color.cpp`).
2. Upload to GPU via `cudaMalloc` + `cudaMemcpy`.
3. Store the device pointer in `GpuSceneData`.

The pbrt-v4 optimizer (`cmd/rgb2spec_opt.cpp`) can regenerate the table for other primaries if needed, but sRGB is sufficient for now.

### 4.4 `tests/SpectralTests.cpp` (CPU-only, GoogleTest or simple main)

See §9 for test list.

---

## 5. Kernel Changes

### 5.1 `Generate.cu` — Sample wavelengths

```cpp
// After computing jitter, before writing PathState:
const float u_lambda = randomFloat(rng);
const SampledWavelengths wl = SampledWavelengths::sampleVisible(u_lambda);
params.queues.wavelengths[pixel] = wl;

PathState state{};
state.throughput = SampledSpectrum(cameraWeight);   // hero wavelength camera response = 1
state.radiance   = SampledSpectrum(0.f);
state.rngState   = rng;
```

`cameraWeight` from the camera's `generateRay` is currently a scalar — keep it scalar and multiply into the initial throughput uniformly (no wavelength-dependent camera response for now; that can be added later via a sensor SPD table).

### 5.2 `Shade.cu` — Spectral BSDF + light sampling

```cpp
const SampledWavelengths& wl = params.queues.wavelengths[hit.sampleIndex];
PathState state = params.queues.pathStates[hit.sampleIndex];

// Environment miss:
// environmentRadiance returns SampledSpectrum (see §5.4)
state.radiance += state.throughput * environmentRadiance(params.scene, hit.rayDirection, cameraRay, wl);

// Surface hit — spectral BSDF:
const BsdfSample bsdfSample = material.sampleBsdfSpectral(
    params.scene.textures, surface.uv, viewDirection, geometricNormal,
    shadingNormal, rng, wl, params.scene.rgbToSpectrumCoeffs);

state.radiance += state.throughput * bsdfSample.emission;
state.throughput *= bsdfSample.weight;

// Direct light — spectral radiance:
// lightSample.radiance is now SampledSpectrum
shadow.contribution = directThroughput * spectralBrdf * lightSample.radiance * cosine;

// Russian roulette — use spectrumY() (luminance):
const float survival = fminf(fmaxf(spectrumY(state.throughput, wl, NrCIE_Y), 0.05f), 0.95f);
```

Note: `evaluateDirect` likewise becomes a spectral function taking `wl` and the table.

### 5.3 `Finalize.cu` — Spectrum → XYZ → sRGB → accumulate

```cpp
const SampledWavelengths wl = params.queues.wavelengths[pixel];
const SampledSpectrum L = state.radiance;

const glm::vec3 xyz = spectrumToXYZ(L, wl, NrCIE_X, NrCIE_Y, NrCIE_Z);
glm::vec3 radiance = xyzToLinearSRGB(xyz);

// clamp negative (can occur from spectral upsampling rounding)
radiance = glm::max(radiance, glm::vec3(0.f));

// rest of accumulation, adaptive state update (unchanged — uses luminance)
const float luminance = radiance.x * 0.2126f + radiance.y * 0.7152f + radiance.z * 0.0722f;
```

The `spectrumToXYZ` and `xyzToLinearSRGB` functions are already in `Spectrum.h` and used unchanged.

### 5.4 `KernelHelpers.h` — Spectral environment

```cpp
NR_GPU inline SampledSpectrum environmentRadiance(
    const GpuSceneData scene,
    const glm::vec3 direction,
    const bool cameraRay,
    const SampledWavelengths& wl)
{
    // sample texture → glm::vec4 → glm::vec3 rgb (unchanged)
    glm::vec3 rgb = environment.color;
    if (environment.textureIndex >= 0)
        rgb *= glm::vec3(scene.textures[environment.textureIndex].sample({u, v}));
    rgb *= scale;
    // convert to spectrum
    return rgbIlluminantToSpectrum(rgb, wl, scene.rgbToSpectrumCoeffs);
}
```

### 5.5 `Connect.cu` — No functional change needed

`shadow.contribution` is now `SampledSpectrum` but the addition is the same:
```cpp
params.queues.pathStates[shadow.sampleIndex].radiance += shadow.contribution;
```
Works unchanged because `SampledSpectrum::operator+=` exists in `Spectrum.h`.

---

## 6. Light Changes

Each light's `sampleLi()` currently returns `LightSample` with `glm::vec3 radiance`. Change signature to return `SampledSpectrum radiance`.

Pattern for all four light types:
```cpp
// Before (e.g. PointLight.h):
LightSample sampleLi(glm::vec3 pos, uint32_t& rng) const {
    LightSample s;
    s.radiance = color * intensity;
    ...
}

// After:
LightSample sampleLi(glm::vec3 pos, uint32_t& rng,
                     const SampledWavelengths& wl,
                     const float* coeffTable) const {
    LightSample s;
    const glm::vec3 rgb = color * intensity;
    s.radiance = rgbIlluminantToSpectrum(rgb, wl, coeffTable);
    ...
}
```

Files: `PointLight.h`, `SpotLight.h`, `RectLight.h`, `DirectionalLight.h`.

---

## 7. Material Changes (`Material.h`)

### 7.1 New method `sampleBsdfSpectral`

Add alongside the existing `sampleBsdf`. The new method:
1. Samples albedo texture → `glm::vec3 rgb` → `SampledSpectrum albedoSpec` via `rgbAlbedoToSpectrum`.
2. Samples emission texture → `SampledSpectrum emissionSpec`.
3. Calls the same geometric sampling (`sampleOpaque`, `sampleDielectric`) — directions unchanged.
4. Returns `BsdfSample` with `SampledSpectrum` fields.

The old `sampleBsdf` (returning `glm::vec3`) can be **kept** for a compile-time `#define NR_SPECTRAL` guard if backward compatibility is needed during transition, but the goal is to fully replace it.

### 7.2 Dispersion (optional but highly desirable)

For dielectric materials, replace the scalar `ior` with a Cauchy approximation:

```
n(λ) = A + B / λ²    (λ in micrometers)
```

Store two floats `iorA`, `iorB` on `Material`. For non-dispersive materials set `iorB = 0`, so `n(λ) = A = ior` (backward compatible).

At dispersive refractive events, `terminateSecondary()` on the `SampledWavelengths` is called so only the hero wavelength's refraction angle determines the ray direction — this is correct and unbiased.

---

## 8. Register Pressure & Performance Analysis

### Per-thread register cost in Shade.cu (worst case)

| Variable | Before | After |
|---|---|---|
| `PathState state` (loaded) | 6 regs (2×vec3) | 8 regs (2×SampledSpectrum=8 floats) |
| `SampledWavelengths wl` | 0 | 8 regs (4λ + 4pdf) |
| `BsdfSample` | ~9 regs (3×vec3+scalars) | ~16 regs (3×SampledSpectrum+scalars) |
| `LightSample` | 3 regs | 4 regs |
| `shadow.contribution` | 3 regs | 4 regs |

Estimated register increase: ~20–25 registers. With N=4 this is acceptable on an Ampere GPU (255 regs/thread). **Do not increase N beyond 4** without measuring occupancy.

### Memory bandwidth

| Buffer | Per-pixel size | Notes |
|---|---|---|
| `PathState` | 48 bytes (same) | `glm::vec3` → `SampledSpectrum` swap is neutral |
| `wavelengths` | 32 bytes new | Read once per kernel stage |
| `ShadowWorkItem` | +4 bytes | Marginal |

Total new bandwidth: ~32 bytes/pixel per kernel launch. At 1920×1080 = ~2M pixels: ~64 MB extra per full wavefront cycle. RTX 3080 Ti has ~912 GB/s — negligible.

### Expected FPS impact

For a typical scene (non-dispersive):
- Upsampling: 4 sigmoid evaluations per BSDF texture sample. `sigmoid(x)` ≈ 3 FFMA + 1 EXP → ~12 extra ops per wavelength.  
- `spectrumToXYZ`: 12 table lookups + 12 multiply-adds (only in Finalize).
- Estimated overhead: **15–25%** → expect ~240–255 fps baseline.

---

## 9. Tests

All in `tests/SpectralTests.cpp`, compile with CPU (`NR_CPU_GPU` = nothing).

### T1 — Round-trip: white upsampling
```cpp
// rgb(1,1,1) → sigmoid polynomial → SampledSpectrum → XYZ → sRGB ≈ (1,1,1)
```
Tolerance: 1e-3.

### T2 — Round-trip: primary colors
Red (1,0,0), Green (0,1,0), Blue (0,0,1). XYZ → sRGB should recover input within 2e-3.

### T3 — Energy conservation
For a Lambertian surface with albedo (0.5, 0.5, 0.5), integrate `sum(f(λᵢ)/pdf(λᵢ))` over N random samples → should converge to ~0.5 luminance.

### T4 — Hero wavelength stratification
`sampleVisible` with u=0..1: verify all four λᵢ are in [360, 830] nm and that the spacing is consistent with the stratification formula.

### T5 — `terminateSecondary` correctness
After `terminateSecondary()`, `secondaryTerminated()` returns true, pdf[1..3] == 0, pdf[0] == original/C.

### T6 — `spectrumY` monotonicity
For a flat unit spectrum, `spectrumY(SampledSpectrum(1.f), wl, NrCIE_Y)` ≈ 1.0 regardless of wl (Monte Carlo estimator unbiasedness).

### T7 — Regression: luminance of upsampled D65 white
sRGB white (1,1,1) upsampled and integrated against Y CMF → luminance ≈ 1.0.

---

## 10. Implementation Order

**Phase 1 — Infrastructure (no rendering change yet)**
1. Add `SampledWavelengths* wavelengths` to `WavefrontQueues` and allocate in `Raytracer.cpp`.
2. Copy pbrt-v4 `sRGBToSpectrumTable_Data` array into `BakeRgbToSpectrumTable.cpp`, upload to GPU.
3. Add `rgbToSpectrumCoeffs` pointer to `GpuSceneData`.
4. Write `RgbToSpectrum.h` with `rgbAlbedoToSpectrum` and `rgbIlluminantToSpectrum`.
5. Write and run **T1–T2** tests.

**Phase 2 — Data structure changes**
6. Change `BsdfSample.weight/albedo/emission` from `glm::vec3` → `SampledSpectrum`.
7. Change `LightSample.radiance` from `glm::vec3` → `SampledSpectrum`.
8. Change `PathState.throughput/radiance` from `glm::vec3` → `SampledSpectrum`.
9. Change `ShadowWorkItem.contribution` from `glm::vec3` → `SampledSpectrum`.
10. Fix all compile errors (light returns, BSDF returns, Connect.cu addition).
11. Write **T3–T7** tests.

**Phase 3 — Kernel wiring**
12. `Generate.cu`: call `sampleVisible`, write to `wavelengths[pixel]`.
13. `Shade.cu`: load wavelengths, convert all RGB evaluations via upsampling, spectral throughput accumulation, spectral shadow contribution, spectral Russian roulette via `spectrumY`.
14. `Finalize.cu`: `spectrumToXYZ` → `xyzToLinearSRGB` → accumulate. Update adaptive sampling luminance.
15. `KernelHelpers.h`: spectral `environmentRadiance`.
16. All light `sampleLi` methods: add `wl` + `coeffTable` params.

**Phase 4 — Dispersion (optional)**
17. Add `iorA`, `iorB` to `Material`.
18. In `sampleDielectric`: evaluate `n(λ) = iorA + iorB/λ²` per wavelength slot, call `terminateSecondary()`.

**Phase 5 — Measure**
19. Record fps, compare to RGB baseline.
20. Visual regression: render Cornell box, compare chromaticity.

---

## 11. Key Constants & Identities

```
NrLambdaMin = 360 nm,  NrLambdaMax = 830 nm  (already in Spectrum.h)
NrSpectrumSamples = 4                          (already in Spectrum.h — do not change)
NrCIE_Y_integral = 106.856895f                 (already in Spectrum.h)

sRGB table resolution: 64^3 × 3 coefficients  → 9 MB on GPU
Sigmoid: S(x) = 1/(1 + exp(-x))               (use __expf on CUDA for speed)
Cauchy glass: n(λ) = A + B/λ²  (λ in µm),     typical borosilicate: A=1.515, B=0.00420
```

---

## 12. Files Modified / Created Summary

| File | Action | Notes |
|---|---|---|
| `src/Raytracing/Types.h` | Modify | PathState, ShadowWorkItem, WavefrontQueues |
| `src/Raytracing/Bsdf.h` | Modify | BsdfSample fields |
| `src/Raytracing/KernelHelpers.h` | Modify | environmentRadiance signature |
| `src/Raytracing/SceneData.h` | Modify | Add rgbToSpectrumCoeffs ptr |
| `src/Raytracing/RgbToSpectrum.h` | **Create** | Upsampling inline functions |
| `src/Light/LightSample.h` | Modify | radiance field |
| `src/Light/PointLight.h` | Modify | sampleLi signature |
| `src/Light/SpotLight.h` | Modify | sampleLi signature |
| `src/Light/RectLight.h` | Modify | sampleLi signature |
| `src/Light/DirectionalLight.h` | Modify | sampleLi signature |
| `src/Mesh/Material.h` | Modify | Add sampleBsdfSpectral, keep sampleBsdf |
| `src/Kernels/Generate.cu` | Modify | Sample wavelengths |
| `src/Kernels/Shade.cu` | Modify | Full spectral path |
| `src/Kernels/Finalize.cu` | Modify | spectrumToXYZ conversion |
| `src/Kernels/Connect.cu` | No change | += works on SampledSpectrum already |
| `src/Raytracing/Raytracer.cpp` | Modify | Allocate wavelength buffer, upload table |
| `src/cpu/BakeRgbToSpectrumTable.cpp` | **Create** | Copy table data, upload to GPU |
| `tests/SpectralTests.cpp` | **Create** | Unit tests T1–T7 |

**Not touched:** Camera files, Scene files, Vulkan files, UI files, Samplers, TLAS/BLAS, all `.slang` shaders.
