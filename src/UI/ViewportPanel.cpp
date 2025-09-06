#include "ViewportPanel.h"
#include <iostream>
#include <ranges>

#include "imgui.h"
#include "ImGuizmo.h"
#define IMVIEWGUIZMO_IMPLEMENTATION
#include "ImViewGuizmo.h"
#include "Log.h"
#include "glm/gtc/type_ptr.hpp"
#include "SDL3/SDL_mouse.h"
#include "Camera/PerspectiveCamera.h"
#include "Mesh/BVH/BVH.h"
#include "Mesh/BVH/BVH.h"
#include "Mesh/BVH/BVH.h"
#include "Mesh/BVH/BVH.h"
#include "Mesh/BVH/BVH.h"
#include "Mesh/BVH/BVH.h"
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

void ViewportPanel::updateLayout() {
    const float aspectRatio = static_cast<float>(width) / static_cast<float>(height);
    const ImVec2 availSize = ImGui::GetContentRegionAvail();
    
    // Calculate the scaled image size to fit while maintaining aspect ratio
    if (availSize.x / availSize.y > aspectRatio) {
        viewportSize.y = availSize.y;
        viewportSize.x = viewportSize.y * aspectRatio;
    } else {
        viewportSize.x = availSize.x;
        viewportSize.y = viewportSize.x / aspectRatio;
    }

    // Calculate the padding required to center the image
    const ImVec2 padding = {(availSize.x - viewportSize.x) * 0.5f, (availSize.y - viewportSize.y) * 0.5f};
    
    // Apply the padding by moving the cursor
    ImGui::SetCursorPos(ImVec2(ImGui::GetCursorPos().x + padding.x, ImGui::GetCursorPos().y + padding.y));
    
    // Store the final screen position for use by gizmos, picking, etc.
    viewportPos = ImGui::GetCursorScreenPos();
}

ivec2 ViewportPanel::screenToPixel() const {
    const ImVec2 screenPos = ImGui::GetMousePos();
    const ImVec2 relativePos = ImVec2(screenPos.x - viewportPos.x, screenPos.y - viewportPos.y);

    const float normX = std::clamp(relativePos.x / viewportSize.x, 0.f, 1.f);
    const float normY = std::clamp(relativePos.y / viewportSize.y, 0.f, 1.f);

    int pixelX = static_cast<int>(std::round(normX * static_cast<float>(width)));
    int pixelY = static_cast<int>(std::round(normY * static_cast<float>(height)));

    return ivec2(pixelX, pixelY);
}

void ViewportPanel::drawBackground() const {
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    constexpr float tileSize = 20.0f;
    constexpr ImU32 col1 = IM_COL32(50, 50, 50, 255);
    const float x0 = viewportPos.x, y0 = viewportPos.y;
    const float x1 = viewportPos.x + viewportSize.x, y1 = viewportPos.y + viewportSize.y;
    const int numX = static_cast<int>(viewportSize.x / tileSize) + 1;
    const int numY = static_cast<int>(viewportSize.y / tileSize) + 1;

    for (int y = 0; y < numY; y++) {
        for (int x = 0; x < numX; x++) {
            if ((x + y) % 2 != 0)
                continue;
            ImVec2 topLeft{x0 + x * tileSize, y0 + y * tileSize};
            ImVec2 bottomRight{topLeft.x + tileSize, topLeft.y + tileSize};
            if (bottomRight.x > x1) bottomRight.x = x1;
            if (bottomRight.y > y1) bottomRight.y = y1;
            drawList->AddRectFilled(topLeft, bottomRight, col1);
        }
    }
}

void ViewportPanel::drawImageAndUpdateState() {
    ImGui::Image(static_cast<VkDescriptorSet>(outputImageDescriptorSet.get()), viewportSize);
    isViewportHovered = ImGui::IsItemHovered();
    uiScale = std::max(viewportSize.x / 1080.0f, 0.5f);
}

void ViewportPanel::beginMouseCapture() {
    if (isCapturingMouse)
        return;
    isCapturingMouse = true;
    SDL_GetMouseState(&oldX, &oldY);
    SDL_SetWindowRelativeMouseMode(context.getWindow(), true);
    ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NoMouse;
    // Clear any initial delta movement
    (void)SDL_GetRelativeMouseState(nullptr, nullptr); 
}

void ViewportPanel::endMouseCapture() {
    if (!isCapturingMouse)
        return;
    isCapturingMouse = false;
    SDL_SetWindowRelativeMouseMode(context.getWindow(), false);
    ImGui::GetIO().ConfigFlags &= ~ImGuiConfigFlags_NoMouse;
    SDL_WarpMouseInWindow(context.getWindow(), oldX, oldY);

    auto* camera = scene.getActiveCamera();
    if (camera && camera->getArcballActive())
        camera->setArcballActive(false);
}

void ViewportPanel::handleInput() {
    // Stop capturing mouse
    if (isCapturingMouse && ImGui::IsMouseReleased(ImGuiMouseButton_Right)) {
        endMouseCapture();
        return; // Consume the event, don't start a new action
    }

    if (!isViewportHovered)
        return;

    // Start camera movement
    if (ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
        beginMouseCapture();
        if (ImGui::IsKeyDown(ImGuiKey_LeftAlt)) // Start Arcball
            handlePositionPicking();
    }

    // Handle object picking (only if not using a gizmo or moving the camera)
    if (!isCapturingMouse && !ImGui::IsAnyItemHovered() && !ImGuizmo::IsUsing() && !ImViewGuizmo::IsUsing() && !ImViewGuizmo::IsOver() && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
        handleObjectPicking();
}

void ViewportPanel::handleTransformGizmo() {
    const auto activeObject = scene.getActiveObject();
    if (!activeObject)
        return;

    auto* camera = scene.getActiveCamera();
    
    ImGuizmo::Style& style = ImGuizmo::GetStyle();
    style.TranslationLineThickness = 4.0f * uiScale;
    style.TranslationLineArrowSize = 6.0f * uiScale;
    style.RotationLineThickness = 6.0f * uiScale;
    style.RotationOuterLineThickness = 2.0f * uiScale;
    style.ScaleLineThickness = 4.0f * uiScale;
    style.ScaleLineCircleSize = 8.0f * uiScale;
    style.CenterCircleSize = 5.0f * uiScale;

    ImGuizmo::BeginFrame();
    ImGuizmo::SetRect(viewportPos.x, viewportPos.y, viewportSize.x, viewportSize.y);
    ImGuizmo::SetDrawlist(ImGui::GetWindowDrawList());

    if (isViewportHovered) {
        if (ImGui::IsKeyPressed(ImGuiKey_W)) currentOperation = ImGuizmo::TRANSLATE;
        if (ImGui::IsKeyPressed(ImGuiKey_E)) currentOperation = ImGuizmo::ROTATE;
        if (ImGui::IsKeyPressed(ImGuiKey_R)) currentOperation = ImGuizmo::SCALE;
    }
    
    const mat4& view = camera->getViewMatrix();
    mat4 proj = camera->getProjectionMatrix();
    mat4 model = activeObject->getWorldTransform().getMatrix();

    if (ImGuizmo::Manipulate(value_ptr(view), value_ptr(proj), currentOperation, currentMode, value_ptr(model)))
        activeObject->setWorldTransformFromMatrix(model);
}

void ViewportPanel::handleViewGizmo() const {
    auto* camera = scene.getActiveCamera();
    if (!camera)
        return;

    ImViewGuizmo::Style& style = ImViewGuizmo::GetStyle();
    style.scale = uiScale;

    vec3 position = camera->getPosition();
    quat rotation = camera->getRotation();

    ImViewGuizmo::BeginFrame();
    bool wasModified = false;
    
    ImVec2 gizmoPos = {viewportPos.x + viewportSize.x - 110.f * uiScale, viewportPos.y + 110.f * uiScale};
    wasModified |= ImViewGuizmo::Rotate(position, rotation, gizmoPos);
    
    gizmoPos.x += 30.f * uiScale; gizmoPos.y += 90.f * uiScale;
    wasModified |= ImViewGuizmo::Zoom(position, rotation, gizmoPos);
    
    gizmoPos.y += 60.f * uiScale;
    wasModified |= ImViewGuizmo::Pan(position, rotation, gizmoPos);

    if (wasModified && !ImGuizmo::IsUsing()) {
        camera->setPosition(position);
        camera->setRotation(rotation);
    }
}

void ViewportPanel::renderToolbar() {
    const ImVec2 toolbarOffset(25.0f * uiScale, 25.0f * uiScale); 
    ImGui::SetCursorScreenPos(ImVec2(viewportPos.x + toolbarOffset.x, viewportPos.y + toolbarOffset.y));

    const float buttonSize = 50.0f * uiScale;
    const ImVec2 buttonVecSize(buttonSize, buttonSize);
    const float rounding = 10.0f * uiScale;
    const float spacing = 5.0f * uiScale;

    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, rounding);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);

    constexpr ImU32 normalColor  = IM_COL32(144, 144, 144, 50);
    constexpr ImU32 hoveredColor = IM_COL32(215, 215, 215, 50);
    constexpr ImU32 activeColor  = IM_COL32(215, 215, 215, 100);

    // Translate Button
    ImGui::PushStyleColor(ImGuiCol_Button, (currentOperation == ImGuizmo::TRANSLATE) ? activeColor : normalColor);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (currentOperation == ImGuizmo::TRANSLATE) ? activeColor : hoveredColor);
    if (ImGui::Button("T", buttonVecSize))
        currentOperation = ImGuizmo::TRANSLATE;
    ImGui::PopStyleColor(2);

    ImGui::SetCursorScreenPos(ImVec2(viewportPos.x + toolbarOffset.x, ImGui::GetCursorScreenPos().y + spacing));

    // Rotate Button
    ImGui::PushStyleColor(ImGuiCol_Button, (currentOperation == ImGuizmo::ROTATE) ? activeColor : normalColor);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (currentOperation == ImGuizmo::ROTATE) ? activeColor : hoveredColor);
    if (ImGui::Button("R", buttonVecSize))
        currentOperation = ImGuizmo::ROTATE;
    ImGui::PopStyleColor(2);

    ImGui::SetCursorScreenPos(ImVec2(viewportPos.x + toolbarOffset.x, ImGui::GetCursorScreenPos().y + spacing));

    // Scale Button
    ImGui::PushStyleColor(ImGuiCol_Button, (currentOperation == ImGuizmo::SCALE) ? activeColor : normalColor);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (currentOperation == ImGuizmo::SCALE) ? activeColor : hoveredColor);
    if (ImGui::Button("S", buttonVecSize))
        currentOperation = ImGuizmo::SCALE;
    ImGui::PopStyleColor(2);

    ImGui::PopStyleVar(2);
}

void ViewportPanel::renderUi() {
    ImGui::Begin(name.c_str());

    updateLayout();

    drawBackground();
    drawImageAndUpdateState();

    renderToolbar();
    handleTransformGizmo();
    handleViewGizmo();
    handleInput();
    
    if (isCapturingMouse)
        scene.getActiveCamera()->update();
    
    ImGui::End();
}

void ViewportPanel::handlePositionPicking() const {

    const ivec2 pixelCoords = screenToPixel();
    
    auto* camera = scene.getActiveCamera();
    if (!camera)
        return;

    vk::BufferImageCopy copyRegion{};
    copyRegion.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
    copyRegion.imageOffset = vk::Offset3D{pixelCoords.x, pixelCoords.y, 0};
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

    LOG_INFO( "Picked Position: (" << position.x << ", " << position.y << ", " << position.z << ")");
    
    camera->setArcballPivot(position);
    camera->setArcballActive(true);
}


void ViewportPanel::handleObjectPicking() const {
    const ivec2 pixel = screenToPixel();
    
    vk::BufferImageCopy copyRegion{};
    copyRegion.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
    copyRegion.imageOffset = vk::Offset3D{ pixel.x, pixel.y, 0 };
    copyRegion.imageExtent = vk::Extent3D{ 1, 1, 1 };
    
    context.oneTimeSubmit([&](const vk::CommandBuffer cmd) {
       outputCrypto.setImageLayout(cmd, vk::ImageLayout::eTransferSrcOptimal);
       cmd.copyImageToBuffer(outputCrypto.getImage(), vk::ImageLayout::eTransferSrcOptimal, cryptoStagingBuffer.getBuffer(), copyRegion);
       outputCrypto.setImageLayout(cmd, vk::ImageLayout::eGeneral);
   });
    
    uint32_t instanceId = INVALID_INSTANCE;
    if (cryptoStagingBufferMappedPtr)
        instanceId = *static_cast<uint32_t*>(cryptoStagingBufferMappedPtr);

    LOG_INFO( "Picked instance ID: " << instanceId);
    
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

void ViewportPanel::updateDisplayImage(const vk::CommandBuffer cmd, Image& srcImage) {
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

    LOG_INFO( "Destroying ViewportPanel");
}