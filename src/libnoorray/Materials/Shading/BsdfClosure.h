#pragma once

#include <cmath>
#include <cstdint>

#include <glm/geometric.hpp>
#include <glm/vec3.hpp>

#include "Backend/CUDA/Annotations.h"
#include "Materials/Shading/Spectrum.h"

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

    NR_CPU_GPU bool isFinite() const
    {
        return value.isFinite() && nr::isFinite(pdf);
    }
};

struct BsdfSample
{
    glm::vec3 direction{};
    SampledSpectrum weight{};
    float pdf{};
    float eta{1.0f};
    BsdfEvent event{BsdfEvent::Diffuse};
    bool singular{};
    // A Sellmeier transmission uses the hero wavelength for this packet's
    // direction. The caller terminates secondary wavelengths before tracing
    // the next segment, making subsequent path samples carry their own
    // wavelength-dependent refracted directions.
    bool dispersive{};

    NR_CPU_GPU bool directionIsValid() const
    {
        const float lengthSquared = glm::dot(direction, direction);
        return nr::isFinite(direction) && nr::isFinite(lengthSquared)
            && lengthSquared > 1.0e-12f && nr::isFinite(pdf) && pdf >= 0.0f
            && weight.isFinite();
    }
};
