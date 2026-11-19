#pragma once

#include <array>
#include <cstdint>

#include "NoorRHI/Pipelines/Pipeline.h"
#include "NoorRay/Gpu/GpuScene.h"

namespace NoorRay::Render {

//! The bit order matches NOORRAY_LINK_CONSTANTS in cmake/NoorRayShaders.cmake.
enum class LinkConstant : std::uint32_t {
    HasGaussians = 1u << 0,
    HasVolumes = 1u << 1,
    HasDispersion = 1u << 2,
    HasAlphaTest = 1u << 3,
    LitGaussians = 1u << 4,
};

inline constexpr std::size_t variantCount = 32;
using KernelVariants = std::array<const NoorRHI::Shader*, variantCount>;

class VariantKey {
public:
    VariantKey() = default;
    VariantKey(const Gpu::SceneContent& content, const bool litGaussians) {
        set(LinkConstant::HasGaussians, content.hasGaussians);
        set(LinkConstant::LitGaussians, content.hasGaussians && litGaussians);
        set(LinkConstant::HasVolumes, content.hasVolumes);
        set(LinkConstant::HasDispersion, content.hasDispersion);
        set(LinkConstant::HasAlphaTest, content.hasAlphaTest);
    }

    bool has(const LinkConstant constant) const { return (mask & static_cast<std::uint32_t>(constant)) != 0; }
    std::uint32_t getMask() const { return mask; }
    const NoorRHI::Shader& select(const KernelVariants& variants) const { return *variants[mask]; }
    bool operator==(const VariantKey&) const = default;

private:
    void set(const LinkConstant constant, const bool value) {
        if (value)
            mask |= static_cast<std::uint32_t>(constant);
    }

    std::uint32_t mask = 0;
};

}
