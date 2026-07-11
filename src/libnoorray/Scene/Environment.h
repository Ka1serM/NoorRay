#pragma once

#include <cstdint>
#include <vector>

#include <cuda_runtime_api.h>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
using glm::vec3;

#include "CUDA/Annotations.h"
#include "CUDA/Unique/Texture.h"
#include "Raytracing/RgbToSpectrum.h"
#include "Samplers/RandomSampler.h"

inline constexpr float EnvironmentPi = 3.14159265358979323846f;

struct EnvironmentSample
{
    glm::vec3 direction{};
    float pdf{};
};

class Environment
{
public:
    int textureIndex{-1};
    vec3 color{1.0f};
    float rotationSin{};
    float rotationCos{1.0f};
    float visibleExposureScale{1.0f};
    float lightingExposureScale{1.0f};
    int visible{1};
    float rotation{};
    float visibleExposure{};
    float lightingExposure{1.0f};
    nr::cuda::UniqueTexture cdfTexture;
    int cdfWidth{};
    int cdfHeight{};
    float importanceWeight{};

    Environment();
    ~Environment();
    Environment(const Environment&) = delete;
    Environment& operator=(const Environment&) = delete;

    void destroyCdf() noexcept;
    void updateDerivedSettings();

    static std::vector<float> computeCdf(const float* hdr, int w, int h);

    // Equirectangular UV for a world-space direction, matching this
    // environment's rotation.
    NR_GPU glm::vec2 uv(const glm::vec3 direction) const
    {
        const float rotatedX = rotationCos * direction.x - rotationSin * direction.z;
        const float rotatedZ = rotationSin * direction.x + rotationCos * direction.z;
        return {
            0.5f + atan2f(rotatedZ, rotatedX) / (2.0f * EnvironmentPi),
            acosf(fminf(fmaxf(direction.y, -1.0f), 1.0f)) / EnvironmentPi};
    }

#if defined(NR_GPU_CODE)
    // Solid-angle PDF of sampleDirection() returning `direction`.
    NR_GPU float pdf(const glm::vec3 direction) const
    {
        if (cdfTexture.getObject() == 0 || cdfWidth <= 0 || cdfHeight <= 0)
            return 1.0f / (4.0f * EnvironmentPi);
        const glm::vec2 texCoord = uv(direction);
        const int x = min(static_cast<int>(texCoord.x * cdfWidth), cdfWidth - 1);
        const int y = min(static_cast<int>(texCoord.y * cdfHeight), cdfHeight - 1);
        const float rowCdf = cdfTexel(0, y).y;
        const float previousRowCdf = y > 0 ? cdfTexel(0, y - 1).y : 0.0f;
        const float columnCdf = cdfTexel(x, y).x;
        const float previousColumnCdf = x > 0 ? cdfTexel(x - 1, y).x : 0.0f;
        const float probability = fmaxf(rowCdf - previousRowCdf, 0.0f)
            * fmaxf(columnCdf - previousColumnCdf, 0.0f);
        const float theta0 = EnvironmentPi * static_cast<float>(y) / static_cast<float>(cdfHeight);
        const float theta1 = EnvironmentPi * static_cast<float>(y + 1) / static_cast<float>(cdfHeight);
        const float solidAngle = (2.0f * EnvironmentPi / static_cast<float>(cdfWidth))
            * fmaxf(cosf(theta0) - cosf(theta1), 1e-12f);
        return probability / solidAngle;
    }

    // Importance-samples a world-space direction from the environment's CDF
    // (or uniformly over the sphere if no HDRI CDF is available).
    NR_GPU EnvironmentSample sampleDirection(RandomState& rng) const
    {
        EnvironmentSample sample{};
        if (cdfTexture.getObject() == 0 || cdfWidth <= 0 || cdfHeight <= 0) {
            const float z = 1.0f - 2.0f * randomFloat(rng);
            const float phi = 2.0f * EnvironmentPi * randomFloat(rng);
            const float radius = sqrtf(fmaxf(1.0f - z * z, 0.0f));
            sample.direction = glm::vec3(radius * cosf(phi), z, radius * sinf(phi));
            sample.pdf = 1.0f / (4.0f * EnvironmentPi);
            return sample;
        }

        const float marginalSample = randomFloat(rng);
        int low = 0, high = cdfHeight - 1;
        while (low < high) {
            const int mid = (low + high) / 2;
            if (cdfTexel(0, mid).y < marginalSample)
                low = mid + 1;
            else
                high = mid;
        }
        const int y = low;
        const float conditionalSample = randomFloat(rng);
        low = 0; high = cdfWidth - 1;
        while (low < high) {
            const int mid = (low + high) / 2;
            if (cdfTexel(mid, y).x < conditionalSample)
                low = mid + 1;
            else
                high = mid;
        }
        const int x = low;

        const float phi = 2.0f * EnvironmentPi * ((static_cast<float>(x) + randomFloat(rng))
            / static_cast<float>(cdfWidth) - 0.5f);
        const float theta0 = EnvironmentPi * static_cast<float>(y) / static_cast<float>(cdfHeight);
        const float theta1 = EnvironmentPi * static_cast<float>(y + 1) / static_cast<float>(cdfHeight);
        const float cosTheta = cosf(theta0)
            + randomFloat(rng) * (cosf(theta1) - cosf(theta0));
        const float sinTheta = sqrtf(fmaxf(1.0f - cosTheta * cosTheta, 0.0f));
        const glm::vec3 rotated(sinTheta * cosf(phi), cosTheta, sinTheta * sinf(phi));
        sample.direction = glm::normalize(glm::vec3(
            rotationCos * rotated.x + rotationSin * rotated.z,
            rotated.y,
            -rotationSin * rotated.x + rotationCos * rotated.z));
        sample.pdf = pdf(sample.direction);
        return sample;
    }

    // Spectral radiance of the environment along `direction`. `cameraRay`
    // selects between the visible-background and lighting exposure scales.
    NR_GPU SampledSpectrum radiance(
        const nr::cuda::UniqueTexture* textures, const uint32_t textureCount,
        const glm::vec3 direction, const bool cameraRay,
        const SampledWavelengths& wl,
        const float* spectrumScale, const float* spectrumCoeffs, const float* d65Table) const
    {
        glm::vec3 rgb = color;
        if (textureIndex >= 0 && static_cast<uint32_t>(textureIndex) < textureCount)
        {
            const glm::vec2 texCoord = uv(direction);
            const glm::vec4 textureColor = textures[textureIndex].sample(texCoord);
            rgb *= glm::vec3(textureColor);
        }
        rgb *= cameraRay ? visibleExposureScale : lightingExposureScale;
        return rgbIlluminantToSpectrum(rgb, wl, spectrumScale, spectrumCoeffs, d65Table);
    }

private:
    NR_GPU float4 cdfTexel(int x, int y) const
    {
        x = max(0, min(x, cdfWidth - 1));
        y = max(0, min(y, cdfHeight - 1));
        return tex2D<float4>(cdfTexture.getObject(),
            (static_cast<float>(x) + 0.5f) / static_cast<float>(cdfWidth),
            (static_cast<float>(y) + 0.5f) / static_cast<float>(cdfHeight));
    }
#endif
};
