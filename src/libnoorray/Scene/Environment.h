#pragma once

#include <cstdint>
#include <vector>

#include <cuda_runtime_api.h>
#include <glm/mat3x3.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
using glm::vec3;

#include "CUDA/Annotations.h"
#include "CUDA/Unique/Texture.h"
#include "Shading/RgbToSpectrum.h"
#include "Samplers/RandomSampler.h"

class Texture;

inline constexpr float EnvironmentPi = 3.14159265358979323846f;

enum class EnvironmentMapping : int
{
    Equirectangular,
    EqualArea,
};

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
    int cdfDirty{1};
    EnvironmentMapping mapping{EnvironmentMapping::Equirectangular};
    glm::mat3 environmentFromWorld{1.f};
    glm::mat3 worldFromEnvironment{1.f};

    Environment();
    ~Environment();
    Environment(const Environment&) = delete;
    Environment& operator=(const Environment&) = delete;

    void destroyCdf() noexcept;
    void updateDerivedSettings();
    void setHdriTexture(const Texture& texture);
    void clearHdriTexture();
    void setEquirectangularMapping();
    void setEqualAreaMapping(const glm::mat3& environmentFromWorldTransform);

    static std::vector<float> computeCdf(const float* hdr, int w, int h,
        EnvironmentMapping mapping = EnvironmentMapping::Equirectangular);

    NR_CPU_GPU static glm::vec2 equalAreaSphereToSquare(const glm::vec3 direction)
    {
        const glm::vec3 d = glm::normalize(direction);
        const float x = fabsf(d.x), y = fabsf(d.y), z = fabsf(d.z);
        const float r = sqrtf(fmaxf(1.0f - z, 0.0f));
        const float a = fmaxf(x, y);
        const float b = a == 0.0f ? 0.0f : fminf(x, y) / a;
        float phi = 0.4067585662467884896e-5f
            + b * (0.6362265452740161349f
            + b * (0.6157201789828021349e-2f
            + b * (-0.2473337332812689442f
            + b * (0.8817706647753162947e-1f
            + b * (0.4190388180291657359e-1f
            + b * -0.2513909723434835093e-1f)))));
        if (x < y) phi = 1.0f - phi;
        float v = phi * r;
        float u = r - v;
        if (d.z < 0.0f) {
            const float oldU = u;
            u = 1.0f - v;
            v = 1.0f - oldU;
        }
        u = copysignf(u, d.x);
        v = copysignf(v, d.y);
        return glm::vec2(0.5f * (u + 1.0f), 0.5f * (v + 1.0f));
    }

    NR_CPU_GPU static glm::vec3 equalAreaSquareToSphere(const glm::vec2 point)
    {
        const float u = 2.0f * point.x - 1.0f;
        const float v = 2.0f * point.y - 1.0f;
        const float up = fabsf(u), vp = fabsf(v);
        const float signedDistance = 1.0f - (up + vp);
        const float r = 1.0f - fabsf(signedDistance);
        const float phi = (r == 0.0f ? 1.0f : (vp - up) / r + 1.0f)
            * EnvironmentPi * 0.25f;
        const float z = copysignf(1.0f - r * r, signedDistance);
        const float radial = r * sqrtf(fmaxf(2.0f - r * r, 0.0f));
        return glm::vec3(
            copysignf(cosf(phi), u) * radial,
            copysignf(sinf(phi), v) * radial,
            z);
    }

    // Equirectangular UV for a world-space direction, matching this
    // environment's rotation.
    NR_GPU glm::vec2 uv(const glm::vec3 direction) const
    {
        if (mapping == EnvironmentMapping::EqualArea) {
            glm::vec3 local = glm::normalize(environmentFromWorld * direction);
            local = glm::vec3(
                rotationCos * local.x - rotationSin * local.y,
                rotationSin * local.x + rotationCos * local.y,
                local.z);
            return equalAreaSphereToSquare(local);
        }
        const float rotatedX = rotationCos * direction.x - rotationSin * direction.z;
        const float rotatedZ = rotationSin * direction.x + rotationCos * direction.z;
        return {
            0.5f + atan2f(rotatedZ, rotatedX) / (2.0f * EnvironmentPi),
            acosf(fminf(fmaxf(direction.y, -1.0f), 1.0f)) / EnvironmentPi};
    }

#if defined(NR_GPU_CODE)
    NR_GPU glm::vec3 rgbRadiance(
        const nr::cuda::UniqueTexture* textures, const uint32_t textureCount,
        const glm::vec3 direction, const bool cameraRay) const
    {
        glm::vec3 rgb = color;
        if (textureIndex >= 0 && static_cast<uint32_t>(textureIndex) < textureCount)
        {
            const glm::vec2 texCoord = uv(direction);
            const glm::vec4 textureColor = textures[textureIndex].sample(texCoord);
            rgb *= glm::vec3(textureColor);
        }
        return rgb * (cameraRay ? visibleExposureScale : lightingExposureScale);
    }

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
        float solidAngle = 4.0f * EnvironmentPi
            / static_cast<float>(cdfWidth * cdfHeight);
        if (mapping == EnvironmentMapping::Equirectangular) {
            const float theta0 = EnvironmentPi * static_cast<float>(y) / static_cast<float>(cdfHeight);
            const float theta1 = EnvironmentPi * static_cast<float>(y + 1) / static_cast<float>(cdfHeight);
            solidAngle = (2.0f * EnvironmentPi / static_cast<float>(cdfWidth))
                * fmaxf(cosf(theta0) - cosf(theta1), 1e-12f);
        }
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

        if (mapping == EnvironmentMapping::EqualArea) {
            glm::vec3 local = equalAreaSquareToSphere(glm::vec2(
                (static_cast<float>(x) + randomFloat(rng)) / static_cast<float>(cdfWidth),
                (static_cast<float>(y) + randomFloat(rng)) / static_cast<float>(cdfHeight)));
            local = glm::vec3(
                rotationCos * local.x + rotationSin * local.y,
                -rotationSin * local.x + rotationCos * local.y,
                local.z);
            sample.direction = glm::normalize(worldFromEnvironment * local);
        } else {
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
        }
        sample.pdf = pdf(sample.direction);
        return sample;
    }

    // Spectral radiance of the environment along `direction`. Intensity applies
    // to every ray; camera rays additionally receive the visible-exposure offset.
    NR_GPU SampledSpectrum radiance(
        const nr::cuda::UniqueTexture* textures, const uint32_t textureCount,
        const glm::vec3 direction, const bool cameraRay,
        const SampledWavelengths& wl,
        const float* spectrumScale, const float* spectrumCoeffs, const float* d65Table) const
    {
        const glm::vec3 rgb = rgbRadiance(textures, textureCount, direction, cameraRay);
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
