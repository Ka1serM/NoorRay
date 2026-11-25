#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "NoorRHI/Commands/CommandList.h"
#include "NoorRHI/Device/Device.h"
#include "NoorRHI/Resources/Image.h"
#include "NoorRay/Core/Error.h"
#include "NoorRay/Render/Denoise/Relax.h"
#include "NoorRay/Render/RendererSettings.h"

namespace NoorRay::Render {

//! The Vulkan extensions of the denoiser plugins, which the device must enable when it opens; only the ones that the device has.
class DlssRrExtensions : public NoorRHI::ExtensionProvider {
public:
    std::vector<std::string> getInstanceExtensions() const override { return {}; }
    std::vector<std::string> getDeviceExtensions(std::uintptr_t instance, std::uintptr_t physicalDevice) const override;
};

//! The camera of one DLSS-RR frame; the inputs are the images of DlssRr.
struct DlssRrFrame {
    //! The sub-pixel sample position of the primary rays in [0, 1).
    NoorRHI::float2 jitter{0.5f, 0.5f};
    DenoiseView view;
    float frameTime = 16.6f;
    bool reset = false;
};

//! The sizes of the last feature.
struct DlssRrCreation {
    std::uint32_t renderWidth = 0;
    std::uint32_t renderHeight = 0;
    std::uint32_t outputWidth = 0;
    std::uint32_t outputHeight = 0;

    bool operator==(const DlssRrCreation&) const = default;
};

//! The values of the last evaluation, as NGX names them.
struct DlssRrEvaluation {
    NoorRHI::float2 jitter{};
    NoorRHI::uint2 renderSubrect{};
    NoorRHI::float2 mvScale{1.0f, 1.0f};
    bool reset = false;
    float preExposure = 1.0f;
    float exposureScale = 1.0f;
    float frameTimeDeltaMs = 0.0f;
    std::array<float, 16> worldToView{};
    std::array<float, 16> viewToClip{};
};

//! One bound NGX input: its name in NVSDK_NGX_VK_DLSSD_Eval_Params without pIn, first letter lower case.
struct DlssRrInput {
    std::string_view name;
    const NoorRHI::Image* image = nullptr;
};

//! DLSS Ray Reconstruction through a denoiser plugin: it denoises the path traced colour and upscales it to the display size in one pass.
class DlssRr {
public:
    DlssRr();
    ~DlssRr();
    DlssRr(DlssRr&&) noexcept;
    DlssRr& operator=(DlssRr&&) noexcept;

    //! False in a build without a denoiser plugin, such as the web build.
    static bool isBuilt();

    [[nodiscard]] std::expected<void, Core::Error> resize(NoorRHI::Device& device, std::uint32_t renderWidth, std::uint32_t renderHeight,
                                                          std::uint32_t displayWidth, std::uint32_t displayHeight);
    [[nodiscard]] std::expected<void, Core::Error> record(NoorRHI::Device& device, NoorRHI::CommandList& list, const DlssRrFrame& frame);
    //! The sample position in [0, 1) of a frame in the Halton (2, 3) sequence, with as many phases as DLSS needs for the current sizes.
    NoorRHI::float2 getJitter(std::uint32_t frameIndex) const;

    //! The inputs from the denoise kernels: the noisy colour, the guides, linear view depth, and the motion to the previous frame in render pixels.
    const NoorRHI::Image& getColor() const { return color; }
    const NoorRHI::Image& getDiffuseAlbedo() const { return diffuseAlbedo; }
    const NoorRHI::Image& getSpecularAlbedo() const { return specularAlbedo; }
    const NoorRHI::Image& getNormals() const { return normals; }
    const NoorRHI::Image& getRoughness() const { return roughness; }
    const NoorRHI::Image& getDepth() const { return depth; }
    const NoorRHI::Image& getMotion() const { return motion; }
    const NoorRHI::Image& getSpecularHitDistance() const { return specularHitDistance; }
    const NoorRHI::Image& getOutput() const { return output; }
    std::array<DlssRrInput, 8> getInputs() const;

    //! The file name of the weights that the plugin loaded; empty before the first record.
    const std::string& getLibrary() const { return library; }
    const DlssRrCreation& getCreation() const { return creation; }
    const DlssRrEvaluation& getEvaluation() const { return evaluation; }

    struct Backend;

private:
    std::unique_ptr<Backend> backend;
    NoorRHI::Image color;
    NoorRHI::Image diffuseAlbedo;
    NoorRHI::Image specularAlbedo;
    NoorRHI::Image normals;
    NoorRHI::Image roughness;
    NoorRHI::Image depth;
    NoorRHI::Image motion;
    NoorRHI::Image specularHitDistance;
    NoorRHI::Image output;
    std::string library;
    DlssRrCreation creation;
    DlssRrEvaluation evaluation;
};

}
