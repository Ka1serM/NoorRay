# Spectral Sampling and Variance Reduction Plan

## Goals

- Keep every estimator unbiased.
- Remove known BSDF/PDF mismatches before optimizing variance.
- Reduce HDRI, direct-light, glossy, transmission, and wavelength noise.
- Give sampling dimensions stable ownership so conditional branches do not shift unrelated samples.

## Phase 1 — Correctness

- [x] Use the exact Smith GGX masking function in the VNDF reflection PDF.
- [x] Fix rough dielectric reflection/transmission evaluation, PDFs, Jacobians, and eta handling.
- [x] Measure adaptive-sampling variance from the current sample, not the accumulated mean.
- [x] Apply shading-normal transport correction and reject invalid geometric-normal events.
- [x] Make shadow visibility respect stochastic/cutout opacity.

## Phase 2 — Direct-light variance

- [x] Sample HDR environments from the existing luminance CDF.
- [x] Add BSDF/environment MIS for both NEE samples and BSDF misses.
- [x] Select analytic lights by estimated power instead of light count.
- [x] Include the environment in the power-weighted light distribution.
- [ ] Sample rectangular lights in solid angle where practical.
- [ ] Build an emissive-triangle distribution and add light/BSDF MIS.

## Phase 3 — Sampling architecture

- [x] Use randomized low-discrepancy wavelength samples per pixel.
- [x] Replace the 32-bit hash-chain path RNG with a high-quality counter/state generator.
- [x] Reserve fixed per-bounce dimensions for opacity, lobe, BSDF direction, light choice,
      light position, and Russian roulette.
- [x] Keep film and lens sequences decorrelated while retaining low-discrepancy convergence.
- [x] Include refractive eta scaling in Russian roulette.

## Phase 4 — Validation

- [ ] Add CPU/GPU statistical tests for GGX and dielectric PDF normalization.
- [ ] Add constant-environment and white-furnace energy tests.
- [x] Add deterministic sampler regression tests.
- [ ] Compare equal-time variance against the previous renderer and PBRT-v4.
- [x] Build the Debug test configuration.
- [ ] Run the GPU end-to-end test (current environment exposes only llvmpipe and
      lacks the required Vulkan external-semaphore extension).
