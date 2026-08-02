#pragma once

#include "api.h"

#include <pxr/pxr.h>

#include <cstdint>
#include <string>
#include <vector>

PXR_NAMESPACE_OPEN_SCOPE

enum class HdNoorRayImagePixelType
{
    Rgba8,
    Rgba16Float,
    Rgba32Float,
};

// Pixels decoded by Hio and expanded to the four-channel layouts accepted by
// Texture's in-memory constructor. Color conversion is intentionally left to
// TextureEncoding at the call site.
struct HDNOORRAY_API HdNoorRayDecodedImage
{
    int width{};
    int height{};
    HdNoorRayImagePixelType pixelType{HdNoorRayImagePixelType::Rgba8};
    std::vector<uint8_t> rgba8;
    // IEEE-754 binary16 bit patterns. Keeping Hio's native representation
    // avoids doubling EXR memory and lets CUDA upload a half4 array directly.
    std::vector<uint16_t> rgba16Float;
    std::vector<float> rgba32Float;

    [[nodiscard]] const void* Data() const noexcept
    {
        switch (pixelType) {
        case HdNoorRayImagePixelType::Rgba16Float:
            return rgba16Float.data();
        case HdNoorRayImagePixelType::Rgba32Float:
            return rgba32Float.data();
        case HdNoorRayImagePixelType::Rgba8:
            return rgba8.data();
        }
        return nullptr;
    }
};

// Opens an Ar-resolvable asset through Hio, reads its native storage format,
// and expands one to four source channels to RGBA. Integer sources become
// RGBA8; half sources stay RGBA16F; float and double sources become RGBA32F.
// Native four-channel byte, half, and float layouts are read directly into
// their final vectors without an intermediate image-sized allocation.
//
// On failure, image is reset and error receives a user-facing diagnostic when
// it is non-null.
HDNOORRAY_API bool HdNoorRayLoadImage(
    const std::string& assetPath, HdNoorRayDecodedImage* image,
    std::string* error = nullptr, bool flipY = true);

PXR_NAMESPACE_CLOSE_SCOPE
