#include "NoorRay/Render/Upscale/DlssRr.h"

#include <algorithm>
#include <cmath>

#include "NoorRay/Render/Denoise/DenoiserPlugin.h"

namespace NoorRay::Render {

namespace {

float halton(std::uint32_t index, const std::uint32_t base) {
    float result = 0.0f;
    float fraction = 1.0f / static_cast<float>(base);
    for (++index; index > 0; index /= base, fraction /= static_cast<float>(base))
        result += fraction * static_cast<float>(index % base);
    return result;
}

}

std::vector<std::string> DlssRrExtensions::getDeviceExtensions(const std::uintptr_t instance, const std::uintptr_t physicalDevice) const {
    std::vector<std::string> names;
    for (const DenoiserPlugin* plugin : getDenoiserPlugins())
        for (std::string& name : plugin->getDeviceExtensions(instance, physicalDevice))
            if (std::ranges::find(names, name) == names.end())
                names.push_back(std::move(name));
    return names;
}

DlssRr::DlssRr() = default;
DlssRr::~DlssRr() = default;
DlssRr::DlssRr(DlssRr&&) noexcept = default;
DlssRr& DlssRr::operator=(DlssRr&&) noexcept = default;

bool DlssRr::isBuilt() {
    return !getDenoiserPlugins().empty();
}

std::expected<void, Core::Error> DlssRr::resize(NoorRHI::Device& device, const std::uint32_t renderWidth, const std::uint32_t renderHeight,
                                                const std::uint32_t displayWidth, const std::uint32_t displayHeight) {
    if (!isBuilt())
        return Core::fail(Core::ErrorCode::Unsupported, "this build has no DLSS-RR denoiser plugin");
    if (output && creation.renderWidth == renderWidth && creation.renderHeight == renderHeight && creation.outputWidth == displayWidth
        && creation.outputHeight == displayHeight)
        return {};
    constexpr NoorRHI::ImageUsage usage = NoorRHI::ImageUsage::Storage | NoorRHI::ImageUsage::Sampled;
    const auto open = [&](NoorRHI::Image& image, const std::uint32_t width, const std::uint32_t height, const NoorRHI::ImageFormat format,
                          const std::string_view name) {
        return Core::check(image.open(device, {.width = width, .height = height, .format = format, .usage = usage, .name = name}));
    };
    if (auto opened = open(color, renderWidth, renderHeight, NoorRHI::ImageFormat::Rgba16Float, "NoorRay DLSS-RR colour"); !opened)
        return opened;
    if (auto opened = open(diffuseAlbedo, renderWidth, renderHeight, NoorRHI::ImageFormat::Rgba16Float, "NoorRay DLSS-RR diffuse albedo"); !opened)
        return opened;
    if (auto opened = open(specularAlbedo, renderWidth, renderHeight, NoorRHI::ImageFormat::Rgba16Float, "NoorRay DLSS-RR specular albedo"); !opened)
        return opened;
    if (auto opened = open(normals, renderWidth, renderHeight, NoorRHI::ImageFormat::Rgba16Float, "NoorRay DLSS-RR normals"); !opened)
        return opened;
    if (auto opened = open(roughness, renderWidth, renderHeight, NoorRHI::ImageFormat::R16Float, "NoorRay DLSS-RR roughness"); !opened)
        return opened;
    if (auto opened = open(depth, renderWidth, renderHeight, NoorRHI::ImageFormat::R32Float, "NoorRay DLSS-RR depth"); !opened)
        return opened;
    if (auto opened = open(motion, renderWidth, renderHeight, NoorRHI::ImageFormat::Rg16Float, "NoorRay DLSS-RR motion"); !opened)
        return opened;
    if (auto opened = open(specularHitDistance, renderWidth, renderHeight, NoorRHI::ImageFormat::R16Float, "NoorRay DLSS-RR specular hit distance"); !opened)
        return opened;
    if (auto opened = open(output, displayWidth, displayHeight, NoorRHI::ImageFormat::Rgba16Float, "NoorRay DLSS-RR output"); !opened)
        return opened;
    creation.renderWidth = renderWidth;
    creation.renderHeight = renderHeight;
    creation.outputWidth = displayWidth;
    creation.outputHeight = displayHeight;
    return {};
}

std::expected<void, Core::Error> DlssRr::record(NoorRHI::Device& device, NoorRHI::CommandList& list, const DlssRrFrame& frame) {
    if (!output)
        return Core::fail(Core::ErrorCode::InvalidState, "DLSS-RR has no size");
    if (!backend) {
        library.clear();
        if (!isBuilt())
            return Core::fail(Core::ErrorCode::Unsupported, "this build has no DLSS-RR denoiser plugin");
        std::expected<std::unique_ptr<Backend>, Core::Error> opened = getDenoiserPlugins().front()->open(device);
        if (!opened)
            return std::unexpected(std::move(opened.error()));
        backend = std::move(*opened);
    }
    std::expected<bool, Core::Error> creates = backend->create(list, creation);
    if (!creates)
        return std::unexpected(std::move(creates.error()));
    if (*creates)
        library = backend->getLibrary();

    //! NGX takes the offset of the sample from the pixel centre with the opposite sign.
    evaluation.jitter = {0.5f - frame.jitter.x, 0.5f - frame.jitter.y};
    evaluation.renderSubrect = {creation.renderWidth, creation.renderHeight};
    evaluation.mvScale = {1.0f, 1.0f};
    evaluation.reset = frame.reset || *creates;
    evaluation.preExposure = 1.0f;
    evaluation.exposureScale = 1.0f;
    evaluation.frameTimeDeltaMs = frame.frameTime;
    evaluation.worldToView = frame.view.worldToView;
    evaluation.viewToClip = frame.view.viewToClip;
    return backend->evaluate(list, evaluation, *this);
}

NoorRHI::float2 DlssRr::getJitter(const std::uint32_t frameIndex) const {
    const float ratio = static_cast<float>(creation.outputWidth) / static_cast<float>(std::max(creation.renderWidth, 1u));
    const auto phases = std::max(static_cast<std::uint32_t>(std::ceil(8.0f * ratio * ratio)), 16u);
    const std::uint32_t phase = frameIndex % phases;
    return {halton(phase, 2), halton(phase, 3)};
}

std::array<DlssRrInput, 8> DlssRr::getInputs() const {
    return {{
        {"color", &color},
        {"diffuseAlbedo", &diffuseAlbedo},
        {"specularAlbedo", &specularAlbedo},
        {"normals", &normals},
        {"roughness", &roughness},
        {"depth", &depth},
        {"motionVectors", &motion},
        {"specularHitDistance", &specularHitDistance},
    }};
}

}
