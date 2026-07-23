#pragma once

#include <cmath>
#include <cstdint>

#include <glm/geometric.hpp>
#include <glm/vec3.hpp>

#include "CUDA/Annotations.h"
#include "Shading/Spectrum.h"

inline constexpr float BsdfEpsilon = 1.0e-5f;
inline constexpr float BsdfPi = 3.14159265358979323846f;
inline constexpr float BsdfSingularPdf = 1.0e6f;

enum class BsdfEvent : uint32_t
{
    Diffuse,
    Specular,
    Transmission,
};

// All scattering closures return their value and sampling density together.
// This is the BSDF equivalent of LightSample: callers cannot accidentally use
// an evaluation from one path and a PDF from another.
struct BsdfEvaluation
{
    SampledSpectrum value{};
    float pdf{};
};

struct BsdfSample
{
    glm::vec3 direction{};
    SampledSpectrum weight{};
    float pdf{};
    float eta{1.0f};
    BsdfEvent event{BsdfEvent::Diffuse};
    bool singular{};
};
