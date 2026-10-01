#include "Realtime/RenderTargets.h"

namespace
{
using noorrhi::ImageFormat;

// NRD_NORMAL_ENCODING 2 (CMakeLists.txt) packs normal and roughness for this
// format; the composite unpacks them the same way.
constexpr ImageFormat NormalRoughnessFormat = ImageFormat::A2b10g10r10Unorm;
// Non-negative colours without alpha: the material factors.
constexpr ImageFormat FactorFormat = ImageFormat::B10g11r11Float;

noorrhi::Image<std::byte> target(noorrhi::Device& device, const Extent extent,
    const ImageFormat format, const bool allocated = true)
{
    if (!allocated)
        return {};
    return device.image<std::byte>(extent.width, extent.height,
        noorrhi::ImageUsage::Storage | noorrhi::ImageUsage::Sampled, format);
}
}

RenderTargets::RenderTargets(noorrhi::Device& device, const RenderTargetLayout& layout)
    : layout_(layout)
    , diffuse_(target(device, layout.lighting, ImageFormat::Rgba16Float))
    , specular_(target(device, layout.lighting, ImageFormat::Rgba16Float))
    , diffuseSh1_(target(device, layout.lighting, ImageFormat::Rgba16Float,
        layout.sphericalHarmonics))
    , specularSh1_(target(device, layout.lighting, ImageFormat::Rgba16Float,
        layout.sphericalHarmonics))
    , lightingNormalRoughness_(target(device, layout.lighting, NormalRoughnessFormat,
        !layout.sharedGuides))
    // View Z needs full precision at distance.
    , lightingViewZ_(target(device, layout.lighting, ImageFormat::R32Float, !layout.sharedGuides))
    , lightingMotion_(target(device, layout.lighting, ImageFormat::Rg16Float, !layout.sharedGuides))
    , layerDiffuse_(target(device, layout.lighting, ImageFormat::Rgba16Float, layout.layers))
    , layerSpecular_(target(device, layout.lighting, ImageFormat::Rgba16Float, layout.layers))
    , layerLightingNormalRoughness_(target(device, layout.lighting, NormalRoughnessFormat,
        layout.layers && !layout.sharedGuides))
    , layerLightingViewZ_(target(device, layout.lighting, ImageFormat::R32Float,
        layout.layers && !layout.sharedGuides))
    , layerLightingMotion_(target(device, layout.lighting, ImageFormat::Rg16Float, layout.layers))
    , normalRoughness_(target(device, layout.render, NormalRoughnessFormat))
    , viewZ_(target(device, layout.render, ImageFormat::R32Float))
    , motion_(target(device, layout.render, ImageFormat::Rg16Float))
    , depth_(target(device, layout.render, ImageFormat::R32Float))
    , emission_(target(device, layout.render, ImageFormat::Rgba16Float))
    , diffuseFactor_(target(device, layout.render, FactorFormat))
    , specularFactor_(target(device, layout.render, FactorFormat))
    , layerDiffuseFactor_(target(device, layout.render, FactorFormat, layout.layers))
    , layerSpecularFactor_(target(device, layout.render, FactorFormat, layout.layers))
    , layerNormalRoughness_(target(device, layout.render, NormalRoughnessFormat, layout.layers))
    , layerViewZ_(target(device, layout.render, ImageFormat::R32Float, layout.layers))
    , transparencyMask_(target(device, layout.render, ImageFormat::R8Unorm, layout.layers))
    , color_(target(device, layout.render, ImageFormat::Rgba16Float, layout.upscaled))
    , cryptomatte_(target(device, layout.render, ImageFormat::R32Uint))
{
}

nr::graphics::RenderTargetHandles RenderTargets::handles() const
{
    nr::graphics::RenderTargetHandles handles{};
    handles.diffuse = diffuse_.storage_handle().value;
    handles.specular = specular_.storage_handle().value;
    handles.diffuseSh1 = diffuseSh1_.storage_handle().value;
    handles.specularSh1 = specularSh1_.storage_handle().value;
    handles.lightingNormalRoughness = lightingNormalRoughnessImage().storage_handle().value;
    handles.lightingViewZ = lightingViewZImage().storage_handle().value;
    handles.lightingMotion = lightingMotionImage().storage_handle().value;
    handles.layerDiffuse = layerDiffuse_.storage_handle().value;
    handles.layerSpecular = layerSpecular_.storage_handle().value;
    handles.layerLightingNormalRoughness = layerLightingNormalRoughnessImage().storage_handle().value;
    handles.layerLightingViewZ = layerLightingViewZImage().storage_handle().value;
    handles.layerLightingMotion = layerLightingMotion_.storage_handle().value;
    handles.normalRoughness = normalRoughness_.storage_handle().value;
    handles.viewZ = viewZ_.storage_handle().value;
    handles.motion = motion_.storage_handle().value;
    handles.depth = depth_.storage_handle().value;
    handles.emission = emission_.storage_handle().value;
    handles.diffuseFactor = diffuseFactor_.storage_handle().value;
    handles.specularFactor = specularFactor_.storage_handle().value;
    handles.layerDiffuseFactor = layerDiffuseFactor_.storage_handle().value;
    handles.layerSpecularFactor = layerSpecularFactor_.storage_handle().value;
    handles.layerNormalRoughness = layerNormalRoughness_.storage_handle().value;
    handles.layerViewZ = layerViewZ_.storage_handle().value;
    handles.transparencyMask = transparencyMask_.storage_handle().value;
    handles.color = color_.storage_handle().value;
    handles.cryptomatte = cryptomatte_.storage_handle().value;
    return handles;
}
