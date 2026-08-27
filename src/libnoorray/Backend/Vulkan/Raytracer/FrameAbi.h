#pragma once

#include <cstdint>
#include <cstddef>

// The raytracer deliberately keeps this ABI independent from Vulkan-Hpp
// and from the host scene classes. It is copied into push data once per
// dispatch; large scene records are reached through descriptor-heap indices
// and images/TLAS are referenced by VK_EXT_descriptor_heap indices. In the
// native Vulkan path the legacy sceneAddress slot carries the packed
// analytic-light descriptor index; mesh data is sceneBuffer.
namespace nr::vulkan {

struct FrameParams
{
    uint64_t sceneAddress{};
    uint64_t cameraAddress{};
    uint64_t lensAddress{};
    uint64_t materialsAddress{};
    uint32_t colorImage{};
    uint32_t albedoImage{};
    uint32_t normalImage{};
    uint32_t positionImage{};
    uint32_t cryptomatteImage{};
    uint32_t lensBuffer{};
    // This legacy slot is the descriptor-heap index of the SVM word stream.
    uint32_t svmWords{};
    uint32_t topLevelAS{};
    uint32_t width{};
    uint32_t height{};
    uint32_t frameIndex{};
    uint32_t sampleIndex{};
    float exposure{};
    float shutterOpen{};
    float shutterClose{};
    // Descriptor-heap index for the packed Vulkan scene record. The record
    // contains immutable mesh/instance metadata and host-mirrored geometry.
    uint32_t sceneBuffer{0xFFFFFFFFu};
    uint32_t gaussianRecords{0xFFFFFFFFu};
    uint32_t gaussianOpacities{0xFFFFFFFFu};
    uint32_t gaussianShCoefficients{0xFFFFFFFFu};
    uint32_t gaussianInstanceOffsets{0xFFFFFFFFu};
    uint32_t gaussianCount{};
    // Low 8 bits cap rendered SH coefficients. Bit 31 selects terminal
    // direct-colour shading; clear means isotropic GI transport.
    uint32_t gaussianShCoefficientCount{};
    // Squared normalized-space cutoff used by the Gaussian acceptance test.
    // Keeping this in the push ABI makes RenderSettings::gaussianCutoffSigma
    // observable by the native renderer instead of silently assuming 3 sigma.
    float gaussianCutoffDistanceSq{9.0f};
    // Running RGB sum and sample count, indexed by pixel.
    uint32_t accumulationBuffer{0xFFFFFFFFu};
    // Descriptor index of the uploaded VulkanEnvironmentSnapshot.
    uint32_t environmentBuffer{0xFFFFFFFFu};
    // Packed unorm16 energy-compensation LUTs (see ShadingTables.h).
    uint32_t energyLutBuffer{0xFFFFFFFFu};
    // CIE X/Y/Z colour matching functions followed by CIE D65.
    uint32_t spectralTables{0xFFFFFFFFu};
    // Iterative path depth. Keeping transport iterative avoids recursive
    // ray-pipeline stack growth while closest-hit owns material sampling.
    uint32_t maxBounces{10u};
    float indirectLightClamp{10.0f};
    uint32_t transparentBackground{};
    uint32_t gaussianProxyTriangleCount{8u};
    uint32_t gaussianOverdrawImage{0xFFFFFFFFu};
    uint32_t gaussianOverdrawEnabled{};
    uint32_t gaussianOverdrawMax{1024u};
    // AOVs use a deterministic center-of-pixel primary sample.
    uint32_t aovEnabled{};
    // TLAS index of the first Gaussian instance. The hit shaders derive the
    // Gaussian ID from InstanceIndex() rather than the instance record's
    // 24-bit custom index, so they need the offset past the mesh instances.
    uint32_t gaussianInstanceBase{};
};

static_assert(offsetof(FrameParams, sceneAddress) == 0);
static_assert(offsetof(FrameParams, colorImage) == 32);
static_assert(offsetof(FrameParams, width) == 64);
static_assert(offsetof(FrameParams, sceneBuffer) == 92);
static_assert(offsetof(FrameParams, gaussianRecords) == 96);
static_assert(offsetof(FrameParams, environmentBuffer) == 128);
static_assert(offsetof(FrameParams, energyLutBuffer) == 132);
static_assert(offsetof(FrameParams, spectralTables) == 136);
static_assert(offsetof(FrameParams, gaussianInstanceBase) == 172);
static_assert(sizeof(FrameParams) == 176);

} // namespace nr::vulkan
