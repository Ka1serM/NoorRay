#pragma once

#include <cstdint>
#include <expected>
#include <vector>

#include "NoorRHI/Resources/Buffer.h"
#include "NoorRay/Core/Error.h"
#include "NoorRay/Scene/Camera.h"
#include "NoorRay/Shaders/Common/GpuRecords.h"

namespace NoorRay::Gpu {

//! The distance from the rear lens vertex to the film that focuses an axial point at focusDistance, or the nominal distance when no ray gets through.
float focusLens(const Scene::Lens& lens, float focusDistance);

class CameraBuffers {
public:
    [[nodiscard]] std::expected<void, Core::Error> update(NoorRHI::Device& device, const Scene::Camera& camera, std::uint32_t width, std::uint32_t height);

    const Shaders::CameraRecord& getRecord() const { return record; }

private:
    NoorRHI::Buffer<Shaders::LensElementRecord> lensElements;
    std::vector<Shaders::LensElementRecord> uploaded;
    Shaders::CameraRecord record{};
};

}
