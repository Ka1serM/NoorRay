#pragma once

#include <cstdint>
#include <vector>

#include <glm/mat3x3.hpp>
#include <glm/vec3.hpp>

#include <noorrhi/noorrhi.hpp>

#include "Shared/Environment.h"

class Texture;
class Scene;

enum class EnvironmentMapping : int
{
    Equirectangular,
    EqualArea,
};

// Host-side environment light. The shader record in Shared/Environment.h is
// reached through the inherited `data` member and is the single source of
// truth for everything the GPU reads - there is no CPU mirror to keep in step.
// Members below are either authored inputs the record stores in derived form,
// or state the shader never sees.
class Environment : public noorrhi::Shared<nr::graphics::Environment>
{
public:
    explicit Environment(Scene* owner = nullptr);
    ~Environment();
    Environment(const Environment&) = delete;
    Environment& operator=(const Environment&) = delete;

    EnvironmentMapping mapping() const
    {
        return static_cast<EnvironmentMapping>(data.mapping);
    }
    float getRotation() const;
    void setRotation(float value)
    {
        rotation = value;
        constexpr float kPi = 3.14159265f;
        const float radians = rotation * (kPi / 180.0f);
        data.rotationSin = std::sin(radians);
        data.rotationCos = std::cos(radians);
        markChanged();
    }
    float getVisibleExposure() const;
    void setVisibleExposure(float value)
    {
        visibleExposure = value;
        data.visibleExposureScale = lightingExposure * std::pow(2.0f, visibleExposure);
        markChanged();
    }
    float getLightingExposure() const;
    void setLightingExposure(float value)
    {
        lightingExposure = value;
        data.lightingExposureScale = lightingExposure;
        data.visibleExposureScale = lightingExposure * std::pow(2.0f, visibleExposure);
        const float luminance = std::max(0.2126f * data.color.r
            + 0.7152f * data.color.g + 0.0722f * data.color.b, 0.0f);
        data.importanceWeight = 4.0f * 3.14159265f * luminance
            * std::max(data.lightingExposureScale, 0.0f);
        markChanged();
    }
    int getTextureIndex() const;
    void setTextureIndex(int value);
    glm::vec3 getColor() const;
    void setColor(const glm::vec3& value)
    {
        data.color = value;
        const float luminance = std::max(0.2126f * data.color.r
            + 0.7152f * data.color.g + 0.0722f * data.color.b, 0.0f);
        data.importanceWeight = 4.0f * 3.14159265f * luminance
            * std::max(data.lightingExposureScale, 0.0f);
        markChanged();
    }
    glm::mat3 environmentFromWorld() const;

    void destroyCdf() noexcept;
    void setHdriTexture(const Texture& texture);
    void clearHdriTexture();
    void releaseGpu() noexcept;

    // Rebuilds the HDRI and importance-CDF images from `hdri` (null clears
    // them), then republishes the shared record. Called by the renderer at
    // the scene publication boundary.
    void uploadImages(noorrhi::Device& device, const Texture* hdri);
    // Republishes the small scalar record only, leaving the images resident.
    void uploadRecord(noorrhi::Device& device);

    noorrhi::Image<std::byte> hdriImage;
    noorrhi::Image<std::byte> cdfImage;

    void setEquirectangularMapping(
        const glm::mat3& environmentFromWorldTransform = glm::mat3(1.f));
    void setEqualAreaMapping(const glm::mat3& environmentFromWorldTransform);
    void setMapping(EnvironmentMapping mapping, const glm::mat3& environmentFromWorldTransform);

    static std::vector<float> computeCdf(const float* hdr, int w, int h,
        EnvironmentMapping mapping = EnvironmentMapping::Equirectangular);

private:
    Scene* owner_{};
    void markChanged();
    // Authored inputs. `data` stores their shader-ready derived values.
    float rotation{};
    float visibleExposure{};
    float lightingExposure{1.0f};
    int textureIndex{-1};
    int cdfDirty{1};

    // The record carries environmentFromWorld in its padded shader layout;
    // this is the inverse, which only the host needs.
    glm::mat3 worldFromEnvironment{1.f};
};
