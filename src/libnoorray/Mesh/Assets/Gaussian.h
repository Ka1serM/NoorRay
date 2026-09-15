#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <glm/ext/matrix_float4x3.hpp>
#include <glm/vec3.hpp>

#include "Shared/Math.h"
#include "Shared/RenderSettings.h"

inline constexpr uint32_t MaxSphericalHarmonicsCoefficientCount = 16;
inline constexpr uint32_t SphericalHarmonicsChannelCount = 3;
// IEEE-754 binary16 conversion used by the graphics API upload ABI. Keeping the
// compact representation as plain bits avoids pulling a device SDK into the
// host scene model.
inline float halfToFloat(const half bits)
{
    const uint32_t sign = (bits & 0x8000u) << 16u;
    const uint32_t exponent = (bits >> 10u) & 0x1fu;
    const uint32_t mantissa = bits & 0x3ffu;
    uint32_t value{};
    if (exponent == 0u)
    {
        if (mantissa == 0u)
            value = sign;
        else
        {
            uint32_t normalized = mantissa;
            uint32_t shift = 0u;
            while ((normalized & 0x400u) == 0u)
            {
                normalized <<= 1u;
                ++shift;
            }
            value = sign | ((127u - 14u - shift) << 23u)
                | ((normalized & 0x3ffu) << 13u);
        }
    }
    else if (exponent == 0x1fu)
        value = sign | 0x7f800000u | (mantissa << 13u);
    else
        value = sign | ((exponent + 112u) << 23u) | (mantissa << 13u);
    float result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

inline half floatToHalf(const float input)
{
    uint32_t bits;
    std::memcpy(&bits, &input, sizeof(bits));
    const uint32_t sign = (bits >> 16u) & 0x8000u;
    const uint32_t exponent = (bits >> 23u) & 0xffu;
    const uint32_t mantissa = bits & 0x7fffffu;
    if (exponent == 0xffu)
        return static_cast<half>(sign | 0x7c00u | (mantissa ? 0x200u : 0u));
    const int32_t adjusted = static_cast<int32_t>(exponent) - 127 + 15;
    if (adjusted <= 0)
    {
        if (adjusted < -10)
            return static_cast<half>(sign);
        const uint32_t rounded = (mantissa | 0x800000u)
            >> static_cast<uint32_t>(1 - adjusted);
        return static_cast<half>(sign | ((rounded + 0x1000u) >> 13u));
    }
    if (adjusted >= 31)
        return static_cast<half>(sign | 0x7c00u);
    return static_cast<half>(sign | (static_cast<uint32_t>(adjusted) << 10u)
        | ((mantissa + 0x1000u) >> 13u));
}

struct SphericalHarmonicsCoefficients
{
    std::array<half, MaxSphericalHarmonicsCoefficientCount
        * SphericalHarmonicsChannelCount> values{};
    uint32_t count{};

    glm::vec3 get(const uint32_t index) const
    {
        return {
            halfToFloat(values[index * SphericalHarmonicsChannelCount + 0]),
            halfToFloat(values[index * SphericalHarmonicsChannelCount + 1]),
            halfToFloat(values[index * SphericalHarmonicsChannelCount + 2]),
        };
    }

    void set(const uint32_t index, const glm::vec3 value)
    {
        values[index * SphericalHarmonicsChannelCount + 0] = floatToHalf(value.x);
        values[index * SphericalHarmonicsChannelCount + 1] = floatToHalf(value.y);
        values[index * SphericalHarmonicsChannelCount + 2] = floatToHalf(value.z);
    }
};

static_assert(sizeof(SphericalHarmonicsCoefficients) == 100);


class Scene;

struct Gaussian
{
    glm::mat4x3 transform;
    float opacity;
    SphericalHarmonicsCoefficients sphericalHarmonics;

    glm::vec3 getShCoefficient(const uint32_t index) const
    {
        return sphericalHarmonics.get(index);
    }

    void setShCoefficient(const uint32_t index, const glm::vec3 value)
    {
        sphericalHarmonics.set(index, value);
    }
};

class GaussianAsset
{
public:
    static GaussianAsset CreateFromFile(
        Scene& scene, const std::string& name, const std::string& path);

    GaussianAsset(Scene& scene, std::string name, std::vector<Gaussian> gaussians);
    GaussianAsset(GaussianAsset&& other) noexcept = default;
    GaussianAsset(const GaussianAsset&) = delete;
    GaussianAsset& operator=(const GaussianAsset&) = delete;
    GaussianAsset& operator=(GaussianAsset&& other) noexcept = delete;
    ~GaussianAsset() = default;

    // Frees the splat data, which is by far the largest resource in a scene.
    void releaseResources();

    const std::string& getName() const { return name; }
    std::string getType() const { return "Gaussian Asset"; }

    uint32_t getGaussianCount() const { return static_cast<uint32_t>(gaussians.size()); }
    Gaussian getGaussian(uint32_t index) const { return gaussians[index]; }
    const std::vector<Gaussian>& getGaussians() const { return gaussians; }
    std::vector<Gaussian>& getGaussians() { return gaussians; }

    // Direct mutations own their synchronization and invalidation. A bulk
    // replacement performs each exactly once, which is the trainer path.
    void setGaussian(uint32_t index, const Gaussian& gaussian);
    void setGaussians(const std::vector<Gaussian>& gaussians);
    void notifyGaussiansChanged();

    const std::string& getPath() const { return path; }

    bool isDirty() const { return dirty; }
    void clearDirtyFlag() { dirty = false; }

private:
    Scene& scene;
    std::string name;
    std::string path;
    bool dirty = false;
    std::vector<Gaussian> gaussians;
};
