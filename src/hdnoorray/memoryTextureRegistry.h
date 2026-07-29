#pragma once

#include "api.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

// Deliberately narrow ABI used by the Blender Python addon. The module is
// already loaded by Hydra; ctypes opens that same DSO and copies Blender's
// scene-linear image buffer directly into this registry. MaterialX carries
// only the corresponding noorray-memory URI.
extern "C" {
HDNOORRAY_API int HdNoorRayRegisterMemoryTextureF32(const char* uri,
    const float* rgba, size_t floatCount, int width, int height);
HDNOORRAY_API void HdNoorRayUnregisterMemoryTexture(const char* uri);
}

namespace hdnoorray
{
struct MemoryTexturePixels
{
    std::shared_ptr<const std::vector<float>> rgba;
    int width{};
    int height{};
};

std::shared_ptr<const MemoryTexturePixels> findMemoryTexture(
    const std::string& uri);
} // namespace hdnoorray
