#include "EnvironmentSnapshot.h"

#include "Scene/Resources/Environment.h"

VulkanEnvironmentSnapshot makeEnvironmentSnapshot(const Environment& environment,
    const std::uint32_t texture, const std::uint32_t cdfTexture)
{
    VulkanEnvironmentSnapshot snapshot{};
    snapshot.color = environment.color;
    snapshot.rotationSin = environment.rotationSin;
    snapshot.rotationCos = environment.rotationCos;
    snapshot.visibleExposureScale = environment.visibleExposureScale;
    snapshot.lightingExposureScale = environment.lightingExposureScale;
    snapshot.importanceWeight = environment.importanceWeight;
    snapshot.texture = texture;
    snapshot.cdfTexture = cdfTexture;
    snapshot.cdfWidth = environment.cdfWidth;
    snapshot.cdfHeight = environment.cdfHeight;
    snapshot.mapping = static_cast<std::int32_t>(environment.mapping);
    for (int column = 0; column < 3; ++column)
        snapshot.environmentFromWorld[column] =
            glm::vec4(environment.environmentFromWorld[column], 0.0f);
    return snapshot;
}
