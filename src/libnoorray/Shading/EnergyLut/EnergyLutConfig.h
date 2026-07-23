#pragma once

namespace nr::shading::energy_lut
{

// The tables deliberately use perceptual roughness, matching Material and
// Ggx::alphaFromPerceptualRoughness(). Cosine and IOR/F0 coordinates are
// warped in EnergyLut.h to spend resolution where the functions vary fastest.
inline constexpr int GgxRoughnessSize = 32;
inline constexpr int GgxCosineSize = 32;

inline constexpr int DielectricF0Size = 16;
inline constexpr int DielectricRoughnessSize = 16;
inline constexpr int DielectricCosineSize = 16;

inline constexpr int GlassIorSize = 32;
inline constexpr int GlassIorHalfSize = GlassIorSize / 2;
inline constexpr int GlassRoughnessSize = 16;
inline constexpr int GlassCosineSize = 16;

inline constexpr float MaximumDielectricF0 = 0.08f;
inline constexpr float MaximumGlassIor = 3.5f;
inline constexpr float MaximumGlassZ = 0.7453559924999299f;

}
