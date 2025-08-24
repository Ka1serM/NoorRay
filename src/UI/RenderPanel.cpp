#include "RenderPanel.h"
#include "imgui.h"
#include "Vulkan/Context.h"
#include "Raytracing/Raytracer.h"
#include "portable-file-dialogs.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <iostream>
#include <filesystem>
#include <vector>
#include <fstream>
#include <future>
#include <algorithm>
#include <stdexcept>

// Helper function to convert 16-bit half float to 32-bit float.
static float halfToFloat(const uint16_t half) {
    const uint32_t sign = (half >> 15) & 0x0001;
    uint32_t exponent = (half >> 10) & 0x001f;
    uint32_t mantissa = half & 0x03ff;

    if (exponent == 0) {
        if (mantissa == 0)
            return sign ? -0.0f : 0.0f;
        while (!(mantissa & 0x0400)) {
            mantissa <<= 1;
            exponent--;
        }
        exponent++;
        mantissa &= ~0x0400;
    } else if (exponent == 31) {
        return sign ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
    }

    exponent = exponent + (127 - 15);
    mantissa = mantissa << 13;

    const uint32_t result = (sign << 31) | (exponent << 23) | mantissa;
    float f;
    memcpy(&f, &result, sizeof(float));
    return f;
}


RenderPanel::RenderPanel(
    std::string name,
    Context& context,
    Raytracer& raytracer,
    Renderer& renderer
)
    : ImGuiComponent(std::move(name)), samples(1), diffuseBounces(4), specularBounces(12), transmissionBounces(24),
      context(context),
      raytracer(raytracer),
      renderer(renderer)
{
}

void RenderPanel::renderUi() {
    // 1. Clean up completed file-writing futures
    if (m_saveState == SaveState::WRITING_TO_DISK) {
        std::erase_if(m_saveFutures,
            [](const std::future<void>& f) {
                return f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
            });
        if (m_saveFutures.empty()) m_saveState = SaveState::IDLE;
    }

    // 2. Perform GPU copies if pending
    if (m_saveState == SaveState::COPY_PENDING) {
        const uint32_t width = raytracer.getWidth();
        const uint32_t height = raytracer.getHeight();
        for (const auto& job : m_saveJobs) {
            const Image& imageToSave = job.imageProvider();
            std::vector<uint8_t> imageData = copyImageToHostMemory(imageToSave.getImage(), job.format, width, height);
            std::future<void> saveFuture = std::async(std::launch::async,
                [this, imgData = std::move(imageData), format = job.format, filename = job.filename, w = width, h = height]() {
                    this->writeDataToFile(imgData, format, filename, w, h);
                });
            m_saveFutures.push_back(std::move(saveFuture));
        }
        m_saveJobs.clear();
        m_saveState = SaveState::WRITING_TO_DISK;
    }

    const bool isSaving = (m_saveState != SaveState::IDLE);

    ImGui::Begin(getType().c_str());

    // --- Render Settings (top) ---
    if (ImGui::BeginTable("RenderSettingsTable", 2, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_NoBordersInBody)) {
        // Label column auto-sizes by default
        ImGui::TableSetupColumn("Label"); // no width flags
        ImGui::TableSetupColumn("Control", ImGuiTableColumnFlags_WidthStretch);

        // Samples Per Pixel
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Samples Per Pixel");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-FLT_MIN);
        ImGui::DragInt("##SamplesDrag", &samples, 0.1f, 1, 64, "%d");

        // Diffuse Bounces
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Diffuse Bounces");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-FLT_MIN);
        ImGui::DragInt("##DiffuseBounces", &diffuseBounces, 0.1f, 1, 64, "%d");

        // Specular Bounces
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Specular Bounces");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-FLT_MIN);
        ImGui::DragInt("##SpecularBounces", &specularBounces, 0.1f, 1, 64, "%d");

        // Transmission Bounces
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Transmission Bounces");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-FLT_MIN);
        ImGui::DragInt("##TransmissionBounces", &transmissionBounces, 0.1f, 1, 64, "%d");

        ImGui::EndTable();
    }
    
    // --- Save Location ---
    ImGui::SeparatorText("Save Location");
    if (ImGui::BeginTable("SaveLocationTable", 2, ImGuiTableFlags_SizingFixedFit)) {
        ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed);
        ImGui::TableSetupColumn("Widget", ImGuiTableColumnFlags_WidthStretch);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted("Output Folder");
        ImGui::TableSetColumnIndex(1);

        float buttonWidth = ImGui::CalcTextSize("Select").x + ImGui::GetStyle().FramePadding.x * 2.0f;
        ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - buttonWidth - ImGui::GetStyle().ItemSpacing.x);
        ImGui::InputText("##savePath", const_cast<char*>(saveLocation.c_str()), saveLocation.size() + 1, ImGuiInputTextFlags_ReadOnly);
        ImGui::PopItemWidth();
        ImGui::SameLine();

        // Launch async folder selection
        if (ImGui::Button("Select", ImVec2(buttonWidth, 0)))
            if (!m_pendingFolderSelection.valid())
                m_pendingFolderSelection = std::async(std::launch::async, []() {
                    return pfd::select_folder("Select Render Output Folder", "").result();
                });

        // Poll async result
        if (m_pendingFolderSelection.valid())
            if (m_pendingFolderSelection.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                std::string selection = m_pendingFolderSelection.get();
                if (!selection.empty()) saveLocation = selection;
            }

        ImGui::EndTable();
    }

    // --- Output Files ---
    ImGui::Spacing();
    ImGui::SeparatorText("Output Files");
    if (ImGui::BeginTable("FilenamesTable", 2, ImGuiTableFlags_SizingFixedFit)) {
        ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed);
        ImGui::TableSetupColumn("Widget", ImGuiTableColumnFlags_WidthStretch);

        ImGui::TableNextRow(); ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted("Beauty (.hdr)");
        ImGui::TableSetColumnIndex(1); ImGui::PushItemWidth(-FLT_MIN);
        ImGui::InputTextWithHint("##beauty", "e.g., final_render.hdr", beautyFilenameBuffer, sizeof(beautyFilenameBuffer));
        ImGui::PopItemWidth();

        ImGui::TableNextRow(); ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted("Albedo (.png)");
        ImGui::TableSetColumnIndex(1); ImGui::PushItemWidth(-FLT_MIN);
        ImGui::InputTextWithHint("##albedo", "e.g., albedo_pass.png", albedoFilenameBuffer, sizeof(albedoFilenameBuffer));
        ImGui::PopItemWidth();

        ImGui::TableNextRow(); ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted("Normals (.hdr)");
        ImGui::TableSetColumnIndex(1); ImGui::PushItemWidth(-FLT_MIN);
        ImGui::InputTextWithHint("##normal", "e.g., world_normals.hdr", normalFilenameBuffer, sizeof(normalFilenameBuffer));
        ImGui::PopItemWidth();

        ImGui::TableNextRow(); ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted("Crypto (.bin)");
        ImGui::TableSetColumnIndex(1); ImGui::PushItemWidth(-FLT_MIN);
        ImGui::InputTextWithHint("##crypto", "e.g., object_ids.bin", cryptoFilenameBuffer, sizeof(cryptoFilenameBuffer));
        ImGui::PopItemWidth();

        ImGui::EndTable();
    }

    // --- Render Button ---
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::BeginDisabled(isSaving);
    const char* buttonText = isSaving ? "Saving..." : "Render";
    ImGui::PushStyleColor(ImGuiCol_Button, IM_COL32(100, 180, 255, 255));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, IM_COL32(120, 200, 255, 255));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, IM_COL32(80, 160, 220, 255));
    if (ImGui::Button(buttonText, ImVec2(-FLT_MIN, 30))) saveRequested = true;
    ImGui::PopStyleColor(3);
    ImGui::EndDisabled();

    ImGui::End();
}

void RenderPanel::executeSave() {
    if (m_saveState != SaveState::IDLE) {
        std::cout << "Save operation already in progress." << std::endl;
        saveRequested = false;
        return;
    }

    std::cout << "Preparing save operation..." << std::endl;
    m_saveJobs.clear();
    const std::filesystem::path basePath(saveLocation);

    auto addJob = [&](const char* buffer, const std::string& ext, const std::function<const Image&()>& provider) {
        std::string filename(buffer);
        if (!filename.empty()) {
            if (filename.length() < ext.length() || filename.substr(filename.length() - ext.length()) != ext) {
                filename += ext;
            }
            const Image& img = provider(); // Get image to extract format
            m_saveJobs.push_back({provider, (basePath / filename).string(), img.getFormat()});
        }
    };
    
    addJob(beautyFilenameBuffer, ".hdr", [&]()->const Image& { return raytracer.getOutputColor(); });
    addJob(albedoFilenameBuffer, ".png", [&]()->const Image& { return raytracer.getOutputAlbedo(); });
    addJob(normalFilenameBuffer, ".hdr", [&]()->const Image& { return raytracer.getOutputNormal(); });
    addJob(cryptoFilenameBuffer, ".bin", [&]()->const Image& { return raytracer.getOutputCrypto(); });

    if (!m_saveJobs.empty()) {
        m_saveState = SaveState::COPY_PENDING;
    }
    
    saveRequested = false;
}

std::vector<uint8_t> RenderPanel::copyImageToHostMemory(const vk::Image srcImage, const vk::Format format, uint32_t width, uint32_t height) const {
    size_t pixelSize = 0;
    switch (format) {
        case vk::Format::eR32G32B32A32Sfloat: pixelSize = 16; break;
        case vk::Format::eR16G16B16A16Sfloat: pixelSize = 8;  break;
        case vk::Format::eR8G8B8A8Unorm:      pixelSize = 4;  break;
        case vk::Format::eR32Uint:            pixelSize = 4;  break;
        default: throw std::runtime_error("Unsupported format for saving.");
    }

    const vk::DeviceSize imageSize = static_cast<vk::DeviceSize>(width) * height * pixelSize;
    Buffer stagingBuffer(context, Buffer::Type::Custom, imageSize, nullptr, vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    context.oneTimeSubmit([&](const vk::CommandBuffer cmd) {
        vk::ImageMemoryBarrier barrier;
        barrier.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite).setDstAccessMask(vk::AccessFlagBits::eTransferRead).setOldLayout(vk::ImageLayout::eGeneral).setNewLayout(vk::ImageLayout::eTransferSrcOptimal).setImage(srcImage).setSubresourceRange({vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer, {}, nullptr, nullptr, barrier);
        vk::BufferImageCopy region;
        region.setImageSubresource({vk::ImageAspectFlagBits::eColor, 0, 0, 1}).setImageExtent({width, height, 1});
        cmd.copyImageToBuffer(srcImage, vk::ImageLayout::eTransferSrcOptimal, stagingBuffer.getBuffer(), region);
        barrier.setSrcAccessMask(vk::AccessFlagBits::eTransferRead).setDstAccessMask(vk::AccessFlagBits::eShaderWrite).setOldLayout(vk::ImageLayout::eTransferSrcOptimal).setNewLayout(vk::ImageLayout::eGeneral);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eAllCommands, {}, nullptr, nullptr, barrier);
    });

    const void* mappedData = context.getDevice().mapMemory(stagingBuffer.getMemory(), 0, imageSize);
    const vk::MappedMemoryRange memoryRange(stagingBuffer.getMemory(), 0, imageSize);
    context.getDevice().invalidateMappedMemoryRanges(memoryRange);
    std::vector<uint8_t> imageData(imageSize);
    memcpy(imageData.data(), mappedData, imageSize);
    context.getDevice().unmapMemory(stagingBuffer.getMemory());
    return imageData;
}

void RenderPanel::writeDataToFile(const std::vector<uint8_t>& imageData, const vk::Format format, const std::string& filename, const uint32_t width, const uint32_t height) const {
    int components = 0;
    size_t pixelSize = 0;

    switch (format) {
        case vk::Format::eR32G32B32A32Sfloat: 
            pixelSize = 16; components = 4;
            stbi_write_hdr(filename.c_str(), width, height, components, reinterpret_cast<const float*>(imageData.data()));
            break;
        case vk::Format::eR16G16B16A16Sfloat: {
            pixelSize = 8; components = 4;
            std::vector<float> floatData(width * height * 4);
            const uint16_t* halfData = reinterpret_cast<const uint16_t*>(imageData.data());
            for (size_t i = 0; i < width * height * 4; ++i) floatData[i] = halfToFloat(halfData[i]);
            stbi_write_hdr(filename.c_str(), width, height, components, floatData.data());
            break;
        }
        case vk::Format::eR8G8B8A8Unorm:
            pixelSize = 4; components = 4;
            stbi_write_png(filename.c_str(), width, height, components, imageData.data(), width * pixelSize);
            break;
        case vk::Format::eR32Uint: {
            pixelSize = 4; components = 1;
            std::ofstream outFile(filename, std::ios::binary);
            if (outFile.is_open()) {
                outFile.write(reinterpret_cast<const char*>(imageData.data()), imageData.size());
                outFile.close();
            } else { std::cerr << "Failed to open file for writing: " << filename << std::endl; }
            break;
        }
        default: std::cerr << "Unsupported format for saving: " << vk::to_string(format) << std::endl; break;
    }
    std::cout << "Successfully saved image to " << filename << std::endl;
}