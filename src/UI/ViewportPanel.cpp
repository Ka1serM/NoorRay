#include "ViewportPanel.h"
#include <iostream>
#include "imgui.h"
#include "ImGuizmo.h"
#include "glm/gtc/type_ptr.hpp"
#include "SDL3/SDL_mouse.h"
#include "Camera/PerspectiveCamera.h"
#include "Scene/MeshInstance.h"

ViewportPanel::ViewportPanel(const std::string& name, Context& context, Scene& scene, const Image& outputColor, Image& outputCrypto, Image& outputPosition, const uint32_t width, const uint32_t height)
    : ImGuiComponent(name), context(context), scene(scene), outputCrypto(outputCrypto), outputPosition(outputPosition), width(width), height(height),
    displayImage(context, width, height, outputColor.getFormat(), vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst),
    cryptoStagingBuffer(context, Buffer::Type::Custom, sizeof(uint32_t), nullptr, vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent),
     positionStagingBuffer(context, Buffer::Type::Custom, sizeof(float) * 4, nullptr, vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent)
{
    cryptoStagingBufferMappedPtr = context.getDevice().mapMemory(cryptoStagingBuffer.getMemory(), 0, sizeof(int));
    positionStagingBufferMappedPtr = context.getDevice().mapMemory(positionStagingBuffer.getMemory(), 0, sizeof(float) * 4);
    
    // Sampler
    vk::SamplerCreateInfo samplerInfo{};
    samplerInfo.magFilter = vk::Filter::eLinear;
    samplerInfo.minFilter = vk::Filter::eLinear;
    samplerInfo.addressModeU = vk::SamplerAddressMode::eClampToEdge;
    samplerInfo.addressModeV = vk::SamplerAddressMode::eClampToEdge;
    samplerInfo.addressModeW = vk::SamplerAddressMode::eClampToEdge;
    sampler = context.getDevice().createSamplerUnique(samplerInfo);

    // Descriptor set layout
    constexpr vk::DescriptorSetLayoutBinding binding{0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment};
    vk::DescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &binding;

    descriptorSetLayout = context.getDevice().createDescriptorSetLayoutUnique(layoutInfo);

    // Descriptor set
    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.descriptorPool = context.getDescriptorPool();
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout.get();

    auto sets = context.getDevice().allocateDescriptorSetsUnique(allocInfo);
    outputImageDescriptorSet = std::move(sets.front());

    const vk::DescriptorImageInfo imageInfo{sampler.get(), outputColor.getImageView(), vk::ImageLayout::eShaderReadOnlyOptimal};

    vk::WriteDescriptorSet write{};
    write.dstSet = outputImageDescriptorSet.get();
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    write.pImageInfo = &imageInfo;

    context.getDevice().updateDescriptorSets(write, nullptr);
    
    // ImGizmo Style
    ImGuizmo::Style& style = ImGuizmo::GetStyle();
    style.HatchedAxisLineThickness = 0;

    // Colors similar to Blender
    style.Colors[ImGuizmo::DIRECTION_X] = ImVec4(0.9f, 0.2f, 0.2f, 1.0f); // X = red
    style.Colors[ImGuizmo::DIRECTION_Y] = ImVec4(0.2f, 0.9f, 0.2f, 1.0f); // Y = green
    style.Colors[ImGuizmo::DIRECTION_Z] = ImVec4(0.2f, 0.5f, 1.0f, 1.0f); // Z = blue
    style.Colors[ImGuizmo::PLANE_X] = ImVec4(0.9f, 0.2f, 0.2f, 1.0f); // plane fill
    style.Colors[ImGuizmo::PLANE_Y] = ImVec4(0.2f, 0.9f, 0.2f, 1.0f); // plane fill
    style.Colors[ImGuizmo::PLANE_Z] = ImVec4(0.2f, 0.5f, 1.0f, 1.0f); // plane fill

    // Blender also fades inactive axes → make selection highlight bright
    style.Colors[ImGuizmo::SELECTION] = ImVec4(1.0f, 1.0f, 0.0f, 1.0f); // yellow highlight
    style.Colors[ImGuizmo::INACTIVE] = ImVec4(0.4f, 0.4f, 0.4f, 0.6f); // gray inactive

    // Rotation circles usually more saturated
    style.Colors[ImGuizmo::ROTATION_USING_BORDER] = ImVec4(1.0f, 0.8f, 0.2f, 1.0f); // golden ring
    style.Colors[ImGuizmo::ROTATION_USING_FILL] = ImVec4(1.0f, 0.8f, 0.2f, 0.3f); // subtle fill
}
void ViewportPanel::renderUi() {
    ImGui::Begin(name.c_str());

   const float aspectRatio = static_cast<float>(width) / static_cast<float>(height);

    ImVec2 imageSize;
    const ImVec2 availSize = ImGui::GetContentRegionAvail();
    if (availSize.x / availSize.y > aspectRatio) {
        imageSize.y = availSize.y;
        imageSize.x = imageSize.y * aspectRatio;
    } else {
        imageSize.x = availSize.x;
        imageSize.y = imageSize.x / aspectRatio;
    }

    const ImVec2 padding = {
        (availSize.x - imageSize.x) * 0.5f,
        (availSize.y - imageSize.y) * 0.5f
    };

    const ImVec2 cursorPos = ImGui::GetCursorPos();
    ImGui::SetCursorPos({cursorPos.x + padding.x, cursorPos.y + padding.y});
    const ImVec2 imagePos = ImGui::GetCursorScreenPos();

    // Draw checkerboard background
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    constexpr float tileSize = 10.0f; // checker size
    constexpr ImU32 col1 = IM_COL32(50, 50, 50, 255);

    const float x0 = imagePos.x;
    const float y0 = imagePos.y;
    const float x1 = imagePos.x + imageSize.x;
    const float y1 = imagePos.y + imageSize.y;

    const int numX = static_cast<int>(imageSize.x / tileSize) + 1;
    const int numY = static_cast<int>(imageSize.y / tileSize) + 1;

    for (int y = 0; y < numY; y++) {
        for (int x = 0; x < numX; x++) {
            if ((x + y) % 2 != 0)
                continue;

            ImVec2 topLeft = {x0 +  static_cast<float>(x) * tileSize, y0 +  static_cast<float>(y) * tileSize };
            ImVec2 bottomRight = { topLeft.x + tileSize, topLeft.y + tileSize };

            // Clip the square to the image bounds
            if (bottomRight.x > x1) bottomRight.x = x1;
            if (bottomRight.y > y1) bottomRight.y = y1;

            drawList->AddRectFilled(topLeft, bottomRight, col1);
        }
    }   

    // Now safe to sample in UI
    ImGui::Image(static_cast<VkDescriptorSet>(outputImageDescriptorSet.get()), imageSize);
    
    const bool isImageHovered = ImGui::IsItemHovered();

    auto* camera = scene.getActiveCamera();
    
    // Gizmo Section
    if (const auto activeObject = scene.getActiveObject()) {
        
        ImGuizmo::Style& style = ImGuizmo::GetStyle();
        
        // Clamp scale so it doesn't explode on tiny windows
        const float scale = std::max(imageSize.x /  1080.0f, 0.5f);
        style.TranslationLineThickness   = 4.0f * scale;
        style.TranslationLineArrowSize   = 6.0f * scale;
        style.RotationLineThickness      = 6.0f * scale;
        style.RotationOuterLineThickness = 2.0f * scale;
        style.ScaleLineThickness         = 4.0f * scale;
        style.ScaleLineCircleSize        = 8.0f * scale;
        style.CenterCircleSize           = 5.0f * scale;
        
        ImGuizmo::BeginFrame();
        ImGuizmo::SetRect(imagePos.x, imagePos.y, imageSize.x, imageSize.y);
        ImGuizmo::SetDrawlist(ImGui::GetWindowDrawList());

        if (isImageHovered) {
            if (ImGui::IsKeyPressed(ImGuiKey_W)) currentOperation = ImGuizmo::TRANSLATE;
            if (ImGui::IsKeyPressed(ImGuiKey_E)) currentOperation = ImGuizmo::ROTATE;
            if (ImGui::IsKeyPressed(ImGuiKey_R)) currentOperation = ImGuizmo::SCALE;
        }

        const mat4& view = camera->getViewMatrix();
        mat4 proj = camera->getProjectionMatrix();

        mat4 model = activeObject->getWorldTransform().getMatrix();
        ImGuizmo::Manipulate(value_ptr(view),value_ptr(proj), currentOperation, currentMode, value_ptr(model));

        if (ImGuizmo::IsUsing())
            activeObject->setWorldTransformFromMatrix(model);
    }

    // Mouse Control Section
    SDL_Window* sdlWindow = context.getWindow();

    if (isImageHovered && !ImGuizmo::IsUsing() && ImGui::IsMouseDown(ImGuiMouseButton_Right) && !isCapturingMouse) {
        SDL_GetMouseState(&oldX, &oldY);
        isCapturingMouse = true;
        ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NoMouse;
        SDL_SetWindowRelativeMouseMode(sdlWindow, true);
        (void)SDL_GetRelativeMouseState(nullptr, nullptr); // fix mouse jump
    }

    if (isCapturingMouse && ImGui::IsMouseReleased(ImGuiMouseButton_Right)) {
        isCapturingMouse = false;
        SDL_SetWindowRelativeMouseMode(sdlWindow, false);
        ImGui::GetIO().ConfigFlags &= ~ImGuiConfigFlags_NoMouse;
        SDL_WarpMouseInWindow(sdlWindow, oldX, oldY);
    }

    if (isCapturingMouse)
        camera->update();

    // Picking Section
    if (isImageHovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !ImGuizmo::IsOver())
        handlePicking(imageSize);

    ImGui::End();
}

void ViewportPanel::handlePicking(const ImVec2 imageSize) const
{
    const ImVec2 mousePos = ImGui::GetMousePos();
    const ImVec2 windowPos = ImGui::GetItemRectMin(); // Top-left corner of the image in screen space
    const int32_t pixelX = static_cast<int32_t>(std::clamp((mousePos.x - windowPos.x) / imageSize.x, 0.f, 1.f) * static_cast<float>(width));
    const int32_t pixelY = static_cast<int32_t>(std::clamp((mousePos.y - windowPos.y) / imageSize.y, 0.f, 1.f) * static_cast<float>(height));
    
    context.oneTimeSubmit([&](const vk::CommandBuffer cmd) {
        //Copy 1x1 pixel from image to buffer
        vk::BufferImageCopy copyRegion{};
        copyRegion.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
        copyRegion.imageOffset = vk::Offset3D{ pixelX, pixelY, 0 };
        copyRegion.imageExtent = vk::Extent3D{ 1, 1, 1 };

        //Copy crypto (instance ID)
        outputCrypto.setImageLayout(cmd, vk::ImageLayout::eTransferSrcOptimal);
        cmd.copyImageToBuffer(outputCrypto.getImage(), vk::ImageLayout::eTransferSrcOptimal, cryptoStagingBuffer.getBuffer(), copyRegion);
        outputCrypto.setImageLayout(cmd, vk::ImageLayout::eGeneral);

        //Copy Position (world space)
        outputPosition.setImageLayout(cmd, vk::ImageLayout::eTransferSrcOptimal);
        cmd.copyImageToBuffer(outputPosition.getImage(), vk::ImageLayout::eTransferSrcOptimal, positionStagingBuffer.getBuffer(), copyRegion);
        outputPosition.setImageLayout(cmd, vk::ImageLayout::eGeneral);
    });

    int instanceId = -1;
    if (cryptoStagingBufferMappedPtr)
        instanceId = *static_cast<int*>(cryptoStagingBufferMappedPtr);

    std::cout << "Picked instance ID: " << instanceId << std::endl;

    vec3 position{0.0f};
    if (positionStagingBufferMappedPtr) {
        const float* f = static_cast<float*>(positionStagingBufferMappedPtr);
        position = vec3(f[0], f[1], f[2]);
    }
    
    std::cout << "Picked Position: X: " << position.x << "Y: " <<  position.y << "Z: " << position.z  << std::endl;

    if (instanceId == -1)
        scene.setActiveObjectIndex(-1); // Deselect
    else
    {
        const SceneObject* pickedObject = scene.getMeshInstances()[instanceId];
        const auto& objects = scene.getSceneObjects();
        for (size_t i = 0; i < objects.size(); ++i) {
            if (objects[i].get() == pickedObject) {
                scene.setActiveObjectIndex(static_cast<uint32_t>(i));
                break;
            }
        }
    }
}

void ViewportPanel::recordCopy(const vk::CommandBuffer cmd, Image& srcImage) {
    srcImage.setImageLayout(cmd, vk::ImageLayout::eTransferSrcOptimal);
    displayImage.setImageLayout(cmd, vk::ImageLayout::eTransferDstOptimal);

    vk::ImageCopy copyRegion{};
    copyRegion.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
    copyRegion.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
    copyRegion.extent = vk::Extent3D{width, height, 1};

    cmd.copyImage(srcImage.getImage(), vk::ImageLayout::eTransferSrcOptimal, displayImage.getImage(), vk::ImageLayout::eTransferDstOptimal, copyRegion);

    srcImage.setImageLayout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
    displayImage.setImageLayout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
}

ViewportPanel::~ViewportPanel() {
    if (cryptoStagingBufferMappedPtr)
        context.getDevice().unmapMemory(cryptoStagingBuffer.getMemory());
    if (positionStagingBufferMappedPtr)
        context.getDevice().unmapMemory(positionStagingBuffer.getMemory());

    std::cout << "Destroying ViewportPanel" << std::endl;
}
