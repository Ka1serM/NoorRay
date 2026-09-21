#include "Realtime/RenderTargets.h"

namespace
{
noorrhi::Image<std::byte> target(noorrhi::Device& device, const Extent extent,
    const noorrhi::ImageFormat format)
{
    return device.image<std::byte>(extent.width, extent.height,
        noorrhi::ImageUsage::Storage | noorrhi::ImageUsage::Sampled, format);
}
}

RenderTargets::RenderTargets(noorrhi::Device& device, const Extent extent)
    : extent_(extent)
    , diffuse_(target(device, extent, noorrhi::ImageFormat::Rgba16Float))
    , specular_(target(device, extent, noorrhi::ImageFormat::Rgba16Float))
    , normalRoughness_(target(device, extent, noorrhi::ImageFormat::Rgba16Float))
    // View Z needs full precision at distance.
    , viewZ_(target(device, extent, noorrhi::ImageFormat::Rgba32Float))
    , motion_(target(device, extent, noorrhi::ImageFormat::Rgba16Float))
    , depth_(target(device, extent, noorrhi::ImageFormat::Rgba32Float))
    , emission_(target(device, extent, noorrhi::ImageFormat::Rgba32Float))
    , diffuseFactor_(target(device, extent, noorrhi::ImageFormat::Rgba16Float))
    , specularFactor_(target(device, extent, noorrhi::ImageFormat::Rgba16Float))
    , color_(target(device, extent, noorrhi::ImageFormat::Rgba32Float))
    , albedo_(target(device, extent, noorrhi::ImageFormat::Rgba32Float))
    , normal_(target(device, extent, noorrhi::ImageFormat::Rgba32Float))
    , position_(target(device, extent, noorrhi::ImageFormat::Rgba32Float))
    , cryptomatte_(target(device, extent, noorrhi::ImageFormat::R32Uint))
{
}

nr::graphics::RenderTargetHandles RenderTargets::handles() const
{
    nr::graphics::RenderTargetHandles handles{};
    handles.diffuse = diffuse_.storage_handle().value;
    handles.specular = specular_.storage_handle().value;
    handles.normalRoughness = normalRoughness_.storage_handle().value;
    handles.viewZ = viewZ_.storage_handle().value;
    handles.motion = motion_.storage_handle().value;
    handles.depth = depth_.storage_handle().value;
    handles.emission = emission_.storage_handle().value;
    handles.diffuseFactor = diffuseFactor_.storage_handle().value;
    handles.specularFactor = specularFactor_.storage_handle().value;
    handles.color = color_.storage_handle().value;
    handles.albedo = albedo_.storage_handle().value;
    handles.normal = normal_.storage_handle().value;
    handles.position = position_.storage_handle().value;
    handles.cryptomatte = cryptomatte_.storage_handle().value;
    return handles;
}
