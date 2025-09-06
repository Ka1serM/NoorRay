#pragma once
#include "UI/ImGuiComponent.h"
#include "Vulkan/Buffer.h"
#include <string>
#include <future>
#include <vector>
#include <functional>

#include "Vulkan/Tonemapper.h"

namespace vk { class Image; }
class Context;
class Raytracer;
class Renderer;
class Image;

class RenderPanel : public ImGuiComponent {
public:
    RenderPanel(std::string name, Context& context, Raytracer& raytracer, Renderer& renderer, Tonemapper& tonemapper);

    void renderUi() override;

    bool isSaveRequested() const { return saveRequested; }
    void executeSave();
    
    int getSamples() const { return samples; }
    int getDiffuseBounces() const { return diffuseBounces; }
    int getSpecularBounces() const { return specularBounces; }
    int getTransmissionBounces() const { return transmissionBounces; }
    float getExposure() const { return exposure; }
    
private:
    int samples, diffuseBounces, specularBounces, transmissionBounces;
    float exposure;
    
    // State machine for the save process
    enum class SaveState {
        IDLE,
        COPY_PENDING,
        WRITING_TO_DISK
    };
    
    // Struct to hold information for a single save job
    struct SaveJob {
        std::function<const Image&()> imageProvider; // Use a lambda to get the image at the time of copy
        std::string filename;
        vk::Format format;
    };

    std::vector<uint8_t> copyImageToHostMemory(vk::Image srcImage, vk::Format format, uint32_t width, uint32_t height) const;
    static void writeDataToFile(const std::vector<uint8_t>& imageData, vk::Format format, const std::string& filename, uint32_t width, uint32_t height);
    Context& context;
    Raytracer& raytracer;
    Renderer& renderer;
    Tonemapper& tonemapper;

    std::string saveLocation = ".";
    char beautyFilenameBuffer[256] = "render_beauty.png";
    char rawFilenameBuffer[256] = "render_hdr.hdr";
    char albedoFilenameBuffer[256] = "render_albedo.png";
    char normalFilenameBuffer[256] = "render_normals.hdr";
    char cryptoFilenameBuffer[256] = "render_crypto.bin";

    bool saveRequested = false;
    std::vector<std::future<void>> m_saveFutures;
    std::future<std::string> m_pendingFolderSelection;
    
    // New state management members
    SaveState m_saveState = SaveState::IDLE;
    std::vector<SaveJob> m_saveJobs;
};