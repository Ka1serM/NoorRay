#pragma once

#include <cstdint>
#include <vector>

#include <glm/mat3x3.hpp>
#include <glm/vec3.hpp>

#include <noorrhi/noorrhi.hpp>

#include "Shared/Environment.h"

class Texture;

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
    // Authored inputs. The record holds the lowered forms: sin/cos of the
    // rotation and the pre-multiplied exposure scales.
    float rotation{};
    float visibleExposure{};
    float lightingExposure{1.0f};

    // Index into Scene's texture table, resolved to an image at upload time.
    int textureIndex{-1};
    int cdfDirty{1};

    // The record carries environmentFromWorld in its padded shader layout;
    // this is the inverse, which only the host needs.
    glm::mat3 worldFromEnvironment{1.f};

    Environment();
    ~Environment();
    Environment(const Environment&) = delete;
    Environment& operator=(const Environment&) = delete;

    EnvironmentMapping mapping() const
    {
        return static_cast<EnvironmentMapping>(data.mapping);
    }
    glm::mat3 environmentFromWorld() const;

    void destroyCdf() noexcept;
    void updateDerivedSettings();
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
};
