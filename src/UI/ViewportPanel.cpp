#include "ViewportPanel.h"
#include <iostream>
#include <ranges>
#include "imgui.h"
#include "ImGuizmo.h"
#include "glm/gtc/type_ptr.hpp"
#include "SDL3/SDL_mouse.h"
#include "Camera/PerspectiveCamera.h"
#include "Scene/MeshInstance.h"

ViewportPanel::ViewportPanel(const std::string& name, Context& context, Scene& scene, const Image& outputColor, Image& outputCrypto, Image& outputPosition, const uint32_t width, const uint32_t height)
    : ImGuiComponent(name), context(context), scene(scene), outputCrypto(outputCrypto),outputPosition(outputPosition), width(width), height(height),
    displayImage(context, width, height, outputColor.getFormat(), vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst),
    cryptoStagingBuffer(context, Buffer::Type::Custom, sizeof(uint32_t), nullptr, vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent),
    positionStagingBuffer(context, Buffer::Type::Custom, sizeof(float) * 4, nullptr, vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent)
{
    const vk::MemoryRequirements cryptoReq =context.getDevice().getBufferMemoryRequirements(cryptoStagingBuffer.getBuffer());
    cryptoStagingBufferMappedPtr = context.getDevice().mapMemory(cryptoStagingBuffer.getMemory(), 0, cryptoReq.size);
    
    const vk::MemoryRequirements positionReq = context.getDevice().getBufferMemoryRequirements(positionStagingBuffer.getBuffer());
    positionStagingBufferMappedPtr = context.getDevice().mapMemory(positionStagingBuffer.getMemory(), 0, positionReq.size);
    
    vk::SamplerCreateInfo samplerInfo{};
    samplerInfo.magFilter = vk::Filter::eLinear;
    samplerInfo.minFilter = vk::Filter::eLinear;
    samplerInfo.addressModeU = vk::SamplerAddressMode::eClampToEdge;
    samplerInfo.addressModeV = vk::SamplerAddressMode::eClampToEdge;
    samplerInfo.addressModeW = vk::SamplerAddressMode::eClampToEdge;
    sampler = context.getDevice().createSamplerUnique(samplerInfo);
    
    constexpr vk::DescriptorSetLayoutBinding binding{0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment};
    vk::DescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &binding;
    descriptorSetLayout = context.getDevice().createDescriptorSetLayoutUnique(layoutInfo);
    
    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.descriptorPool = context.getDescriptorPool();
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout.get();
    auto sets = context.getDevice().allocateDescriptorSetsUnique(allocInfo);
    outputImageDescriptorSet = std::move(sets.front());
    
    const vk::DescriptorImageInfo imageInfo{sampler.get(), displayImage.getImageView(), vk::ImageLayout::eShaderReadOnlyOptimal};
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

    // ... (Viewport Size & Position calculation is unchanged) ...
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
    const ImVec2 padding = {(availSize.x - imageSize.x) * 0.5f, (availSize.y - imageSize.y) * 0.5f};
    const ImVec2 cursorPos = ImGui::GetCursorPos();
    ImGui::SetCursorPos({cursorPos.x + padding.x, cursorPos.y + padding.y});
    const ImVec2 imagePos = ImGui::GetCursorScreenPos();
    
    // ... (Checkerboard Background drawing is unchanged) ...
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    constexpr float tileSize = 10.0f;
    constexpr ImU32 col1 = IM_COL32(50, 50, 50, 255);
    const float x0 = imagePos.x, y0 = imagePos.y;
    const float x1 = imagePos.x + imageSize.x, y1 = imagePos.y + imageSize.y;
    const int numX = static_cast<int>(imageSize.x / tileSize) + 1;
    const int numY = static_cast<int>(imageSize.y / tileSize) + 1;
    for (int y = 0; y < numY; y++) {
        for (int x = 0; x < numX; x++) {
            if ((x + y) % 2 != 0) continue;
            ImVec2 topLeft{x0 + x * tileSize, y0 + y * tileSize};
            ImVec2 bottomRight{topLeft.x + tileSize, topLeft.y + tileSize};
            if (bottomRight.x > x1) bottomRight.x = x1;
            if (bottomRight.y > y1) bottomRight.y = y1;
            drawList->AddRectFilled(topLeft, bottomRight, col1);
        }
    }
    
    // Display Image & Get State
    ImGui::Image(static_cast<VkDescriptorSet>(outputImageDescriptorSet.get()), imageSize);
    const bool isImageHovered = ImGui::IsItemHovered();
    auto* camera = scene.getActiveCamera();
    
    // ... (Gizmo handling is unchanged) ...
    if (const auto activeObject = scene.getActiveObject()) {
        ImGuizmo::Style& style = ImGuizmo::GetStyle();
        const float scale = std::max(imageSize.x / 1080.0f, 0.5f);
        style.TranslationLineThickness = 4.0f * scale;
        style.TranslationLineArrowSize = 6.0f * scale;
        style.RotationLineThickness = 6.0f * scale;
        style.RotationOuterLineThickness = 2.0f * scale;
        style.ScaleLineThickness = 4.0f * scale;
        style.ScaleLineCircleSize = 8.0f * scale;
        style.CenterCircleSize = 5.0f * scale;

        ImGuizmo::BeginFrame();
        ImGuizmo::SetRect(imagePos.x, imagePos.y, imageSize.x, imageSize.y);
        ImGuizmo::SetDrawlist(ImGui::GetWindowDrawList());
        if (isImageHovered) {
            if (ImGui::IsKeyPressed(ImGuiKey_W))
                currentOperation = ImGuizmo::TRANSLATE;
            if (ImGui::IsKeyPressed(ImGuiKey_E))
                currentOperation = ImGuizmo::ROTATE;
            if (ImGui::IsKeyPressed(ImGuiKey_R))
                currentOperation = ImGuizmo::SCALE;
        }
        const mat4& view = camera->getViewMatrix();
        mat4 proj = camera->getProjectionMatrix();
        mat4 model = activeObject->getWorldTransform().getMatrix();
        if (ImGuizmo::Manipulate(value_ptr(view), value_ptr(proj), currentOperation, currentMode, value_ptr(model)))
            activeObject->setWorldTransformFromMatrix(model);
    }

    // Handle All Camera and Picking Input
    SDL_Window* sdlWindow = context.getWindow();

    // Handle Stopping Camera Movement (triggers on mouse release)
    if (ImGui::IsMouseReleased(ImGuiMouseButton_Right)) {
        if (isCapturingMouse) { // Stop either FPS or Arcball mode capture
            isCapturingMouse = false;
            SDL_SetWindowRelativeMouseMode(sdlWindow, false);
            ImGui::GetIO().ConfigFlags &= ~ImGuiConfigFlags_NoMouse;
            SDL_WarpMouseInWindow(sdlWindow, oldX, oldY);
        }
        if (camera->getArcballActive()) // Stop Arcball mode logic
            camera->setArcballActive(false);
    }

    // Handle Starting Camera Movement (triggers on the first frame of a click)
    if (isImageHovered && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
        if (ImGui::IsKeyDown(ImGuiKey_LeftAlt)) {
            // --- START ARCBALL ---
            // Save cursor position and capture mouse
            SDL_GetMouseState(&oldX, &oldY);
            isCapturingMouse = true;
            ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NoMouse;
            SDL_SetWindowRelativeMouseMode(sdlWindow, true);

            // Set pivot and clear initial mouse delta
            const ImVec2 mousePos = ImGui::GetMousePos();
            const ImVec2 windowPos = ImGui::GetItemRectMin();
            const int32_t pixelX = static_cast<int32_t>(std::clamp((mousePos.x - windowPos.x) / imageSize.x, 0.f, 1.f) * static_cast<float>(width));
            const int32_t pixelY = static_cast<int32_t>(std::clamp((mousePos.y - windowPos.y) / imageSize.y, 0.f, 1.f) * static_cast<float>(height));
            handlePositionPicking(pixelX, pixelY);
            (void)SDL_GetRelativeMouseState(nullptr, nullptr);
        } else {
            // --- START FPS ---
            SDL_GetMouseState(&oldX, &oldY);
            isCapturingMouse = true;
            ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NoMouse;
            SDL_SetWindowRelativeMouseMode(sdlWindow, true);
            (void)SDL_GetRelativeMouseState(nullptr, nullptr);
        }
    }

    // ... (Object picking logic is unchanged) ...
    if (isImageHovered && !ImGuizmo::IsUsing() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        const ImVec2 mousePos = ImGui::GetMousePos();
        const ImVec2 windowPos = ImGui::GetItemRectMin();
        const int32_t pixelX = static_cast<int32_t>(std::clamp((mousePos.x - windowPos.x) / imageSize.x, 0.f, 1.f) * static_cast<float>(width));
        const int32_t pixelY = static_cast<int32_t>(std::clamp((mousePos.y - windowPos.y) / imageSize.y, 0.f, 1.f) * static_cast<float>(height));
        handleObjectPicking(pixelX, pixelY);
    }
    
    // The camera update is called as long as either mode is active.
    if (isCapturingMouse)
        camera->update();
    
    ImGui::End();
}

void ViewportPanel::handlePositionPicking(const int32_t pixelX, const int32_t pixelY) const {
    auto* camera = scene.getActiveCamera();
    if (!camera) return;

    vk::BufferImageCopy copyRegion{};
    copyRegion.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
    copyRegion.imageOffset = vk::Offset3D{pixelX, pixelY, 0};
    copyRegion.imageExtent = vk::Extent3D{1, 1, 1};

    context.oneTimeSubmit([&](const vk::CommandBuffer cmd) {
        outputPosition.setImageLayout(cmd, vk::ImageLayout::eTransferSrcOptimal);
        cmd.copyImageToBuffer(outputPosition.getImage(), vk::ImageLayout::eTransferSrcOptimal, positionStagingBuffer.getBuffer(), copyRegion);
        outputPosition.setImageLayout(cmd, vk::ImageLayout::eGeneral);
    });

    vec3 position{0.f};
    if (positionStagingBufferMappedPtr) {
        const float* f = static_cast<float*>(positionStagingBufferMappedPtr);
        position = vec3(f[0], f[1], f[2]);
    }

    std::cout << "Picked Position: (" << position.x << ", " << position.y << ", " << position.z << ")" << std::endl;
    
    camera->setArcballPivot(position);
    camera->setArcballActive(true);
}


void ViewportPanel::handleObjectPicking(const int32_t pixelX, const int32_t pixelY) const {    
    vk::BufferImageCopy copyRegion{};
    copyRegion.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
    copyRegion.imageOffset = vk::Offset3D{ pixelX, pixelY, 0 };
    copyRegion.imageExtent = vk::Extent3D{ 1, 1, 1 };
    
    context.oneTimeSubmit([&](const vk::CommandBuffer cmd) {
       outputCrypto.setImageLayout(cmd, vk::ImageLayout::eTransferSrcOptimal);
       cmd.copyImageToBuffer(outputCrypto.getImage(), vk::ImageLayout::eTransferSrcOptimal, cryptoStagingBuffer.getBuffer(), copyRegion);
       outputCrypto.setImageLayout(cmd, vk::ImageLayout::eGeneral);
   });
    
    uint32_t instanceId = INVALID_INSTANCE;
    if (cryptoStagingBufferMappedPtr)
        instanceId = *static_cast<uint32_t*>(cryptoStagingBufferMappedPtr);

    std::cout << "Picked instance ID: " << instanceId << std::endl;
    
    if (instanceId != INVALID_INSTANCE && instanceId < scene.getMeshInstances().size()) {
        const MeshInstance* pickedInstance = scene.getMeshInstances()[instanceId];
        const auto& objects = scene.getSceneObjects();
        const auto it = std::ranges::find_if(objects, 
         [pickedInstance](const std::unique_ptr<SceneObject>& obj) {
             return obj.get() == pickedInstance;
         });
    
        if (it != objects.end())
            scene.setActiveObjectIndex(static_cast<uint32_t>(it - objects.begin()));
        else
            scene.resetActiveObjectIndex();
    }
    else
        scene.resetActiveObjectIndex();
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
    if (cryptoStagingBufferMappedPtr) {
        context.getDevice().unmapMemory(cryptoStagingBuffer.getMemory());
        cryptoStagingBufferMappedPtr = nullptr;
    }
    
    if (positionStagingBufferMappedPtr)
    {
        context.getDevice().unmapMemory(positionStagingBuffer.getMemory());
        positionStagingBufferMappedPtr = nullptr;
    }

    std::cout << "Destroying ViewportPanel" << std::endl;
}