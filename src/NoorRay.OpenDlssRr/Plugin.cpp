#include <cstdio>
#include <filesystem>

#include <OpenDlssRr/EmbeddedShaders.h>
#include <OpenDlssRr/RayReconstruction.h>
#include <vulkan/vulkan.h>

#include "NoorRHI/Device/Interop.h"
#include "NoorRay.OpenDlssRr/Weights.h"
#include "NoorRay/Render/Denoise/DenoiserPlugin.h"

namespace NoorRay::OpenDlssRr {

using Render::DlssRr;
using Render::DlssRrCreation;
using Render::DlssRrEvaluation;

namespace {

Core::Error openError(const ::OpenDlssRr::Error& error) {
    return {Core::ErrorCode::Gpu, "OpenDLSS-RR: " + error.message};
}

std::expected<::OpenDlssRr::Texture, Core::Error> getTexture(const NoorRHI::Image& image) {
    std::expected<std::uintptr_t, Core::Error> handle = Core::check(NoorRHI::Interop::getImage(image));
    if (!handle)
        return std::unexpected(std::move(handle.error()));
    std::expected<std::uintptr_t, Core::Error> view = Core::check(NoorRHI::Interop::getImageView(image));
    if (!view)
        return std::unexpected(std::move(view.error()));
    return ::OpenDlssRr::Texture{reinterpret_cast<VkImage>(*handle), reinterpret_cast<VkImageView>(*view)};
}

//! OpenDLSS-RR: the kernels and weights of NGX as plain Vulkan.
class Backend final : public DlssRr::Backend {
public:
    Backend(NoorRHI::Device& device, const NoorRHI::Interop::DeviceHandles& handles) : device(device), handles(handles) {}
    ~Backend() override { release(); }

    std::expected<bool, Core::Error> create(NoorRHI::CommandList& list, const DlssRrCreation& creation) override;
    std::expected<void, Core::Error> evaluate(NoorRHI::CommandList& list, const DlssRrEvaluation& evaluation, const DlssRr& images) override;
    std::string getLibrary() const override { return library; }

private:
    //! The runtime frees its resources at once, so the GPU must finish the frames that use them first.
    void release() {
        if (!reconstruction)
            return;
        static_cast<void>(device.waitIdle());
        reconstruction = {};
    }

    NoorRHI::Device& device;
    NoorRHI::Interop::DeviceHandles handles;
    ::OpenDlssRr::RayReconstruction reconstruction;
    DlssRrCreation created;
    std::string library;
};

std::expected<bool, Core::Error> Backend::create(NoorRHI::CommandList&, const DlssRrCreation& creation) {
    if (reconstruction && created == creation)
        return false;
    release();
    const std::expected<void, Core::Error> downloaded = Weights::wait();
    std::expected<std::filesystem::path, ::OpenDlssRr::Error> found = ::OpenDlssRr::findLibrary();
    if (!found && !downloaded)
        return Core::fail(Core::ErrorCode::Io, "OpenDLSS-RR: " + downloaded.error().message + "; " + found.error().message);
    if (!found)
        return std::unexpected(openError(found.error()));
    std::fprintf(stderr, "NoorRay: OpenDLSS-RR %ux%u -> %ux%u, weights of %s\n", creation.renderWidth, creation.renderHeight, creation.outputWidth,
                 creation.outputHeight, found->string().c_str());
    const ::OpenDlssRr::Settings settings{
        .instance = reinterpret_cast<VkInstance>(handles.instance),
        .physicalDevice = reinterpret_cast<VkPhysicalDevice>(handles.physicalDevice),
        .device = reinterpret_cast<VkDevice>(handles.device),
        .pipelineCache = reinterpret_cast<VkPipelineCache>(handles.pipelineCache),
        .library = *found,
        .shaders = ::OpenDlssRr::getEmbeddedShaders(),
        .renderWidth = creation.renderWidth,
        .renderHeight = creation.renderHeight,
        .outputWidth = creation.outputWidth,
        .outputHeight = creation.outputHeight,
    };
    if (auto opened = reconstruction.open(settings); !opened) {
        reconstruction = {};
        return std::unexpected(openError(opened.error()));
    }
    //! The lifted kernels take seconds to compile, so the cache keeps them at once; Blender can exit without closing the device.
    if (auto saved = Core::check(NoorRHI::Interop::savePipelineCache(device)); !saved)
        std::fprintf(stderr, "NoorRay: OpenDLSS-RR cannot save the pipeline cache: %s\n", saved.error().message.c_str());
    created = creation;
    library = found->filename().string();
    return true;
}

std::expected<void, Core::Error> Backend::evaluate(NoorRHI::CommandList& list, const DlssRrEvaluation& evaluation, const DlssRr& images) {
    ::OpenDlssRr::Textures textures;
    for (auto [image, texture] :
         {std::pair{&images.getColor(), &textures.color}, std::pair{&images.getDiffuseAlbedo(), &textures.diffuseAlbedo},
          std::pair{&images.getSpecularAlbedo(), &textures.specularAlbedo}, std::pair{&images.getNormals(), &textures.normals},
          std::pair{&images.getRoughness(), &textures.roughness}, std::pair{&images.getDepth(), &textures.depth},
          std::pair{&images.getMotion(), &textures.motionVectors}, std::pair{&images.getSpecularHitDistance(), &textures.specularHitDistance},
          std::pair{&images.getOutput(), &textures.output}}) {
        std::expected<::OpenDlssRr::Texture, Core::Error> got = getTexture(*image);
        if (!got)
            return std::unexpected(std::move(got.error()));
        *texture = *got;
    }
    //! NoorRHI waits for all earlier commands here, and its later commands wait for these.
    std::expected<std::uintptr_t, Core::Error> commands = Core::check(NoorRHI::Interop::getCommandBuffer(list));
    if (!commands)
        return std::unexpected(std::move(commands.error()));
    const ::OpenDlssRr::Evaluation evaluate{
        .jitter = {evaluation.jitter.x, evaluation.jitter.y},
        .reset = evaluation.reset,
        .worldToView = evaluation.worldToView,
        .viewToClip = evaluation.viewToClip,
    };
    if (auto recorded = reconstruction.record(reinterpret_cast<VkCommandBuffer>(*commands), textures, evaluate); !recorded)
        return std::unexpected(openError(recorded.error()));
    return {};
}

class Plugin final : public Render::DenoiserPlugin {
public:
    std::string_view getName() const override { return "OpenDLSS-RR"; }

    std::vector<std::string> getDeviceExtensions(const std::uintptr_t instance, const std::uintptr_t physicalDevice) const override {
        std::expected<std::vector<const char*>, ::OpenDlssRr::Error> supported =
            ::OpenDlssRr::getSupportedDeviceExtensions(reinterpret_cast<VkInstance>(instance), reinterpret_cast<VkPhysicalDevice>(physicalDevice));
        if (!supported) {
            std::fprintf(stderr, "NoorRay: OpenDLSS-RR has no extension list: %s\n", supported.error().message.c_str());
            return {};
        }
        return {supported->begin(), supported->end()};
    }

    void prepare() override { Weights::prefetch(); }

    std::expected<std::unique_ptr<DlssRr::Backend>, Core::Error> open(NoorRHI::Device& device) override {
        std::expected<NoorRHI::Interop::DeviceHandles, Core::Error> handles = Core::check(NoorRHI::Interop::getDeviceHandles(device));
        if (!handles)
            return std::unexpected(std::move(handles.error()));
        if (auto checked =
                ::OpenDlssRr::checkDevice(reinterpret_cast<VkInstance>(handles->instance), reinterpret_cast<VkPhysicalDevice>(handles->physicalDevice));
            !checked)
            return std::unexpected(openError(checked.error()));
        return std::make_unique<Backend>(device, *handles);
    }
};

Plugin plugin;
const Render::DenoiserPluginRegistration registration{plugin};

}

}
