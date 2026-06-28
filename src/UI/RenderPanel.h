#pragma once

#include "UI/ImGuiComponent.h"
#include "portable-file-dialogs.h"
#include <string>
#include <utility>
#include <future>
#include <vector>
#include <functional>

#include "Vulkan/Tonemapper.h"

namespace vk { class Image; enum class Format; }
class Context;
class Raytracer;
class Renderer;
class Image;
class Scene;

class RenderPanel : public ImGuiComponent {
public:
    RenderPanel(std::string name, Context& context, Scene& scene, Raytracer& raytracer, Renderer& renderer, Tonemapper& tonemapper);

    void renderUi() override;

    bool isSaveRequested() const { return saveRequested; }
    void executeSave();

    int getPixelSizePercent() const { return pixelSizePercent; }

private:
    // Render Settings
    int pixelSizePercent{100};
    bool changed = false;

    // State machine for the save process
    enum class SaveState {
        IDLE,
        COPY_PENDING,
        WRITING_TO_DISK
    };

    // Struct to hold information for a single save job
    struct SaveJob {
        std::function<const Image&()> imageProvider;
        std::string filename;
        vk::Format format;
    };

    // Helper Functions
    std::vector<uint8_t> copyImageToHostMemory(vk::Image srcImage, vk::Format format, uint32_t width, uint32_t height) const;
    static void writeDataToFile(std::vector<uint8_t>& imageData, vk::Format format, const std::string& filename, uint32_t width, uint32_t height);

    // Core System References
    Context& context;
    Scene& scene;
    Raytracer& raytracer;
    Renderer& renderer;
    Tonemapper& tonemapper;

    // UI State
    std::string saveLocation = ".";
    char beautyFilenameBuffer[256] = "render_beauty.png";
    char rawFilenameBuffer[256] = "render_hdr.hdr";
    char albedoFilenameBuffer[256] = "render_albedo.png";
    char normalFilenameBuffer[256] = "render_normals.hdr";
    char cryptoFilenameBuffer[256] = "render_crypto.bin";

    // Save Process Management
    bool saveRequested = false;
    std::vector<std::future<void>> m_saveFutures;
    std::unique_ptr<pfd::select_folder> m_folderDialog;

    SaveState m_saveState = SaveState::IDLE;
    std::vector<SaveJob> m_saveJobs;
};
