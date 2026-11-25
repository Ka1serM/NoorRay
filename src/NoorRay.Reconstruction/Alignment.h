#pragma once

#include <cstdint>
#include <optional>

#include "DiffRenderer/Scenes/Dataset.h"
#include "NoorRay/Core/Math.h"

namespace NoorRay::Reconstruction {

//! Places the COLMAP world in the scene: the mean up of the photographs becomes +y, the centre of the cameras the origin, and their radius four units.
class Alignment {
public:
    Alignment() = default;
    explicit Alignment(const DiffRenderer::Scenes::Dataset& dataset);

    //! The method's camera for a scene camera, at this image size.
    DiffRenderer::Scenes::Camera toMethod(const Core::Transform& cameraToWorld, float verticalFov, std::uint32_t width, std::uint32_t height) const;
    //! The scene transform of a camera of the dataset, which looks along -z with y up.
    Core::Transform toScene(const DiffRenderer::Scenes::Camera& camera) const;
    //! The scene transform of the whole dataset space.
    Core::Transform getDatasetToScene() const;
    static float getVerticalFov(const DiffRenderer::Scenes::Camera& camera);
    Core::Bounds getCameraBounds(const DiffRenderer::Scenes::Dataset& dataset) const;

private:
    Core::float3 toSceneVector(Core::float3 vector) const;
    Core::float3 toMethodVector(Core::float3 vector) const;

    //! The rows of the rotation from the COLMAP world to the scene.
    Core::float3 rows[3]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    Core::float3 center{};
    float scale = 1.0f;
};

}
