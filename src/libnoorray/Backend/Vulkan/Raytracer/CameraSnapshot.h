#pragma once

#include <cstdint>

// Compact camera record consumed by the raytracer Slang kernel through a
// descriptor-heap storage-buffer entry.  The matrix is row-major because the
// shader reads it as four-byte scalar words; this avoids GLM/Slang packing
// differences while preserving Camera::cameraToWorld semantics.
struct VulkanCameraSnapshot
{
    float cameraToWorld[16]{
        1.f, 0.f, 0.f, 0.f,
        0.f, 1.f, 0.f, 0.f,
        0.f, 0.f, 1.f, 0.f,
        0.f, 0.f, 0.f, 1.f};
    uint32_t projection{}; // CameraProjectionType, kept as a plain ABI value.
    float sensorWidthMm{36.f};
    float sensorHeightMm{24.f};
    float focalLengthMm{50.f};
    float focusDistanceCm{500.f};
    float apertureDiameterMm{};
    uint32_t sensorOrigin{};
    float exposure{};
    uint32_t reserved[2]{};
};

static_assert(sizeof(VulkanCameraSnapshot) == 104);
