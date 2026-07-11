#pragma once

// Gaussians are truncated to this many standard deviations. GaussianProxyBlas
// scales the shared unit icosahedron proxy by this factor once, at BLAS-build
// time, so it tightly bounds a default (unit-sigma) Gaussian's support region.
// Gaussian.transform (GaussianAsset.h) then only has to carry each Gaussian's
// own true R*S — applied for free by the hardware instance transform — so
// GaussianHit.cu's density evaluation stays a plain exp(-0.5 * distanceSq)
// with no per-hit correction factor.
//
// Kept in its own header (no glm/std includes) so it's safe to pull into the
// OptiX device translation unit (GaussianHit.cu) without dragging in
// GaussianAsset.h's host-only class hierarchy.
static constexpr float GaussianCutoffSigma = 3.0f;
