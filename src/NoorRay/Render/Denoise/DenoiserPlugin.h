#pragma once

#include <cstdint>
#include <expected>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "NoorRHI/Commands/CommandList.h"
#include "NoorRHI/Device/Device.h"
#include "NoorRay/Core/Error.h"
#include "NoorRay/Render/Upscale/DlssRr.h"

namespace NoorRay::Render {

//! One implementation of DLSS-RR. It records into the command list of the renderer and keeps the feature of the last creation.
struct DlssRr::Backend {
    Backend() = default;
    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;
    virtual ~Backend() = default;

    //! Makes the feature again if the creation changed; true if it did, which starts the history again.
    [[nodiscard]] virtual std::expected<bool, Core::Error> create(NoorRHI::CommandList& list, const DlssRrCreation& creation) = 0;
    [[nodiscard]] virtual std::expected<void, Core::Error> evaluate(NoorRHI::CommandList& list, const DlssRrEvaluation& evaluation, const DlssRr& images) = 0;
    //! The file name of the library or the model that the feature runs.
    virtual std::string getLibrary() const = 0;
};

//! A denoiser that a build can leave out, which registers itself with a static DenoiserPluginRegistration; DLSS-RR runs the first one.
class DenoiserPlugin {
public:
    virtual ~DenoiserPlugin() = default;

    virtual std::string_view getName() const = 0;
    //! The Vulkan device extensions that the plugin needs and the physical device has; the device enables them when it opens.
    virtual std::vector<std::string> getDeviceExtensions(std::uintptr_t instance, std::uintptr_t physicalDevice) const = 0;
    //! Runs at program start, before the first device opens.
    virtual void prepare() {}
    [[nodiscard]] virtual std::expected<std::unique_ptr<DlssRr::Backend>, Core::Error> open(NoorRHI::Device& device) = 0;
};

inline std::vector<DenoiserPlugin*>& getDenoiserPlugins() {
    static std::vector<DenoiserPlugin*> plugins;
    return plugins;
}

inline void prepareDenoiserPlugins() {
    for (DenoiserPlugin* plugin : getDenoiserPlugins())
        plugin->prepare();
}

struct DenoiserPluginRegistration {
    explicit DenoiserPluginRegistration(DenoiserPlugin& plugin) { getDenoiserPlugins().push_back(&plugin); }
};

}
