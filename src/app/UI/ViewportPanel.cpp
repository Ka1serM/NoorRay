#include "ViewportPanel.h"
#include <cmath>
#include <iostream>
#include <ranges>

#include "imgui.h"
#include "ImGuizmo.h"
#define IMVIEWGUIZMO_IMPLEMENTATION
#include "ImViewGuizmo.h"
#include "Log.h"
#include "glm/gtc/type_ptr.hpp"
#include "SDL3/SDL_mouse.h"
#include "Camera/CameraInstance.h"
#include "Scene/MeshInstance.h"
#include "Scene/LightInstance.h"
#include "Vulkan/Viewport.h"
#include "UI/Window.h"

ViewportPanel::ViewportPanel(const std::string& name, Window& window, Context& context,
    Scene& scene, const Image& outputColor, Image& outputCrypto, Image& outputPosition,
    const uint32_t width, const uint32_t height)
    : ImGuiComponent(name), window(window), context(context), scene(scene),
    outputCrypto(&outputCrypto), outputPosition(&outputPosition), width(width), height(height),
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
    
    updateDisplayDescriptor();

    // DPI scale is fixed for the lifetime of the window (see Context::dpiScale), so
    // everything derived from it below only needs to be set up once, not per frame.
    uiScale = window.getDpiScale();

    // ImGizmo Style
    ImGuizmo::Style& style = ImGuizmo::GetStyle();
    style.HatchedAxisLineThickness = 0;
    style.TranslationLineThickness = 4.0f * uiScale;
    style.TranslationLineArrowSize = 6.0f * uiScale;
    style.RotationLineThickness = 6.0f * uiScale;
    style.RotationOuterLineThickness = 2.0f * uiScale;
    style.ScaleLineThickness = 4.0f * uiScale;
    style.ScaleLineCircleSize = 8.0f * uiScale;
    style.CenterCircleSize = 5.0f * uiScale;

    // Colors match ImViewGuizmo's axis colors 1:1 (IM_COL32(233,62,85) / (140,206,40) / (49,155,249))
    style.Colors[ImGuizmo::DIRECTION_X] = ImVec4(0.9137f, 0.2431f, 0.3333f, 1.0f); // X = red
    style.Colors[ImGuizmo::DIRECTION_Y] = ImVec4(0.5490f, 0.8078f, 0.1569f, 1.0f); // Y = green
    style.Colors[ImGuizmo::DIRECTION_Z] = ImVec4(0.1922f, 0.6078f, 0.9765f, 1.0f); // Z = blue
    style.Colors[ImGuizmo::PLANE_X] = ImVec4(0.9137f, 0.2431f, 0.3333f, 1.0f); // plane fill
    style.Colors[ImGuizmo::PLANE_Y] = ImVec4(0.5490f, 0.8078f, 0.1569f, 1.0f); // plane fill
    style.Colors[ImGuizmo::PLANE_Z] = ImVec4(0.1922f, 0.6078f, 0.9765f, 1.0f); // plane fill

    // Blender also fades inactive axes → make selection highlight bright
    style.Colors[ImGuizmo::SELECTION] = ImVec4(1.0f, 1.0f, 0.0f, 1.0f); // yellow highlight
    style.Colors[ImGuizmo::INACTIVE] = ImVec4(0.4f, 0.4f, 0.4f, 0.6f); // gray inactive

    // Rotation circles usually more saturated
    style.Colors[ImGuizmo::ROTATION_USING_BORDER] = ImVec4(1.0f, 0.8f, 0.2f, 1.0f); // golden ring
    style.Colors[ImGuizmo::ROTATION_USING_FILL] = ImVec4(1.0f, 0.8f, 0.2f, 0.3f); // subtle fill

    // Keep axis handles fixed to their world-space direction instead of flipping
    // to always face the camera.
    ImGuizmo::AllowAxisFlip(false);

    // ImGuizmo's default gizmo size is a fraction of clip space, so it grows in
    // screen pixels as the viewport gets bigger. Counteract that so it stays a
    // constant pixel size (only DPI-scaled), matching ImViewGuizmo's behavior.
    // Only viewportSize.x varies per frame, so precompute the rest here.
    constexpr float referenceViewportWidth = 1080.0f;
    constexpr float baseGizmoSizeClipSpace = 0.1f;
    gizmoSizeClipSpaceScale = baseGizmoSizeClipSpace * uiScale * referenceViewportWidth;

    ImViewGuizmo::Style& viewGizmoStyle = ImViewGuizmo::GetStyle();
    viewGizmoStyle.scale = uiScale;
}

void ViewportPanel::updateDisplayDescriptor()
{
    const vk::DescriptorImageInfo imageInfo{sampler.get(), displayImage.getView(), vk::ImageLayout::eShaderReadOnlyOptimal};
    vk::WriteDescriptorSet write{};
    write.dstSet = outputImageDescriptorSet.get();
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    write.pImageInfo = &imageInfo;
    context.getDevice().updateDescriptorSets(write, nullptr);
}

void ViewportPanel::resize(const uint32_t newWidth, const uint32_t newHeight, const vk::Format imageFormat)
{
    if (newWidth == 0 || newHeight == 0 || (newWidth == width && newHeight == height && displayImage.getFormat() == imageFormat))
        return;

    context.getDevice().waitIdle();
    width = newWidth;
    height = newHeight;
    displayImage = Image(context, width, height, imageFormat, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst);
    updateDisplayDescriptor();
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
}

void ViewportPanel::processEvent(const SDL_Event& event)
{
    if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN &&
        event.button.button == SDL_BUTTON_RIGHT) {
        rightButtonDown = true;
        rightButtonPressPending = true;
        rightButtonPressX = event.button.x;
        rightButtonPressY = event.button.y;
        pendingMouseDeltaX = 0.f;
        pendingMouseDeltaY = 0.f;
    } else if (event.type == SDL_EVENT_MOUSE_BUTTON_UP &&
               event.button.button == SDL_BUTTON_RIGHT) {
        rightButtonDown = false;
    } else if (event.type == SDL_EVENT_MOUSE_MOTION && rightButtonDown) {
        pendingMouseDeltaX += event.motion.xrel;
        pendingMouseDeltaY += event.motion.yrel;
    } else if (event.type == SDL_EVENT_WINDOW_FOCUS_LOST) {
        rightButtonDown = false;
        rightButtonPressPending = false;
        pendingMouseDeltaX = 0.f;
        pendingMouseDeltaY = 0.f;
    }
}

void ViewportPanel::beginMouseCapture() {
    if (isCapturingMouse)
        return;
    SDL_GetMouseState(&oldX, &oldY);
    if (!window.setRelativeMouseMode(true)) {
        LOG_ERROR("Failed to enable relative mouse mode: " << SDL_GetError());
        return;
    }
    isCapturingMouse = true;
}

void ViewportPanel::endMouseCapture() {
    if (!isCapturingMouse)
        return;
    isCapturingMouse = false;
    if (!window.setRelativeMouseMode(false))
        LOG_ERROR("Failed to disable relative mouse mode: " << SDL_GetError());
    window.warpMouse(oldX, oldY);

    auto* camera = scene.getRenderCamera();
    if (camera && camera->getArcballActive())
        camera->setArcballActive(false);
    pendingMouseDeltaX = 0.f;
    pendingMouseDeltaY = 0.f;
}

void ViewportPanel::synchronizeCameraTransition()
{
    const uint64_t revision = scene.getActiveCameraRevision();
    if (revision == observedCameraRevision)
        return;

    if (isCapturingMouse)
        endMouseCapture();
    if (auto* camera = scene.getRenderCamera())
        camera->setArcballActive(false);
    observedCameraRevision = revision;
    // Discard motion accumulated for the previous camera.
    pendingMouseDeltaX = 0.f;
    pendingMouseDeltaY = 0.f;
    rightButtonPressPending = false;
}

void ViewportPanel::handleInput() {
    if (!scene.getRenderCamera()) {
        if (isCapturingMouse)
            endMouseCapture();
        rightButtonPressPending = false;
        return;
    }

    // Stop capturing mouse
    if (isCapturingMouse &&
        (!rightButtonDown || !window.hasInputFocus() ||
         !window.isRelativeMouseMode())) {
        if (auto* camera = scene.getRenderCamera(); camera &&
            (pendingMouseDeltaX != 0.f || pendingMouseDeltaY != 0.f))
            camera->update(pendingMouseDeltaX, pendingMouseDeltaY);
        endMouseCapture();
        rightButtonPressPending = false;
        return; // Consume the event, don't start a new action
    }

    if (!isViewportHovered && !rightButtonPressPending)
        return;

    if (isViewportHovered)
        handleScrollZoom();

    // Use SDL's ordered button edge rather than ImGui's frame-delayed click edge.
    const bool pressIsInViewport =
        rightButtonPressX >= viewportPos.x &&
        rightButtonPressY >= viewportPos.y &&
        rightButtonPressX < viewportPos.x + viewportSize.x &&
        rightButtonPressY < viewportPos.y + viewportSize.y;
    if (rightButtonPressPending && rightButtonDown && pressIsInViewport) {
        beginMouseCapture();
        if (ImGui::IsKeyDown(ImGuiKey_LeftAlt)) // Start Arcball
            handlePositionPicking();
    }
    rightButtonPressPending = false;

    // Toggle overlays
    if (ImGui::IsKeyPressed(ImGuiKey_H))
        m_showOverlays = !m_showOverlays;

    // Handle object picking (only if not using a gizmo or moving the camera)
    if (!isCapturingMouse && !ImGui::IsAnyItemHovered() && !ImGuizmo::IsUsing() && !ImViewGuizmo::IsUsing() && !ImViewGuizmo::IsOver() && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
        handleObjectPicking();
}

void ViewportPanel::handleScrollZoom() {
    auto* camera = scene.getRenderCamera();
    if (!camera || isCapturingMouse || ImGuizmo::IsUsing() ||
        ImViewGuizmo::IsUsing() || ImViewGuizmo::IsOver())
    {
        return;
    }

    const float wheel = ImGui::GetIO().MouseWheel;
    if (wheel == 0.0f)
        return;

    constexpr float dollySpeed = 0.05f;
    constexpr float wheelToDragPixels = 10.0f;
    const vec3 forward = normalize(camera->getRotation() * CameraInstance::LocalForward);
    camera->setPosition(camera->getPosition() + forward * (wheel * wheelToDragPixels * dollySpeed));
}

void ViewportPanel::handleTransformGizmo() {
    const auto activeObject = scene.getActiveObjectPtr();
    if (!activeObject)
        return;

    auto* camera = scene.getRenderCamera();
    if (!camera)
        return;
    
    ImGuizmo::SetGizmoSizeClipSpace(gizmoSizeClipSpaceScale / std::max(viewportSize.x, 1.0f));

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
    auto* camera = scene.getRenderCamera();
    if (!camera)
        return;

    vec3 position = camera->getPosition();
    quat rotation = camera->getRotation();

    vec3 pivot = camera->getArcballPivot();
    if (const auto activeObject = scene.getActiveObjectPtr();
        activeObject && activeObject.get() != camera)
        pivot = activeObject->getPosition();

    ImViewGuizmo::BeginFrame();
    bool wasModified = false;

    // ImViewGuizmo draws straight to the window draw list with no clipping of its
    // own (unlike ImGuizmo::Manipulate), so on a small/narrow viewport its fixed
    // pixel offsets can push it outside the rendered image. Clip to the viewport
    // image bounds, same as renderToolbar() does.
    ImGui::PushClipRect(viewportPos, ImVec2(viewportPos.x + viewportSize.x, viewportPos.y + viewportSize.y), true);

    ImVec2 gizmoPos = {viewportPos.x + viewportSize.x - 110.f * uiScale, viewportPos.y + 110.f * uiScale};
    wasModified |= ImViewGuizmo::Rotate(position, rotation, pivot, gizmoPos);

    gizmoPos.x += 30.f * uiScale; gizmoPos.y += 90.f * uiScale;
    wasModified |= ImViewGuizmo::Dolly(position, rotation, gizmoPos);

    gizmoPos.y += 60.f * uiScale;
    wasModified |= ImViewGuizmo::Pan(position, rotation, gizmoPos);

    ImGui::PopClipRect();

    if (wasModified && !ImGuizmo::IsUsing()) {
        camera->setPosition(position);
        camera->setRotation(rotation);
    }
}

void ViewportPanel::renderToolbar() {
    // Clip to the viewport image bounds, same as ImGuizmo::Manipulate does internally,
    // so the buttons can't spill outside the rendered image.
    ImGui::PushClipRect(viewportPos, ImVec2(viewportPos.x + viewportSize.x, viewportPos.y + viewportSize.y), true);

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

    ImGui::PopClipRect();
}

void ViewportPanel::renderUi() {
    ImGui::Begin(name.c_str());

    synchronizeCameraTransition();

    updateLayout();

    drawBackground();
    drawImageAndUpdateState();

    if (m_showOverlays) {
        renderToolbar();
        handleTransformGizmo();
        handleViewGizmo();
    }
    handleInput();
    
    if (isCapturingMouse) {
        if (auto* camera = scene.getRenderCamera())
            camera->update(pendingMouseDeltaX, pendingMouseDeltaY);
    }
    pendingMouseDeltaX = 0.f;
    pendingMouseDeltaY = 0.f;
    
    ImGui::End();
}

void ViewportPanel::handlePositionPicking() const {

    const ivec2 pixelCoords = screenToPixel();
    
    auto* camera = scene.getRenderCamera();
    if (!camera)
        return;

    vk::BufferImageCopy copyRegion{};
    copyRegion.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
    copyRegion.imageOffset = vk::Offset3D{pixelCoords.x, pixelCoords.y, 0};
    copyRegion.imageExtent = vk::Extent3D{1, 1, 1};

    context.oneTimeSubmit([&](const vk::CommandBuffer cmd) {
        outputPosition->setImageLayout(cmd, vk::ImageLayout::eTransferSrcOptimal);
        cmd.copyImageToBuffer(outputPosition->getImage(), vk::ImageLayout::eTransferSrcOptimal, positionStagingBuffer.getBuffer(), copyRegion);
        outputPosition->setImageLayout(cmd, vk::ImageLayout::eGeneral);
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


bool ViewportPanel::handleBillboardPicking() const {
    auto* camera = scene.getRenderCamera();
    if (!camera)
        return false;

    // A little more forgiving than the visual icon so small targets are still easy to click.
    constexpr float pickRadius = ViewportBillboardPixelRadius;

    const vec2 pixel = vec2(screenToPixel());
    const mat4 viewProjection = camera->getProjectionMatrix() * camera->getViewMatrix();

    std::shared_ptr<LightInstance> closest;
    float closestDistSq = pickRadius * pickRadius;

    for (const auto& obj : scene.getSceneObjects()) {
        const auto light = std::dynamic_pointer_cast<LightInstance>(obj);
        if (!light)
            continue;

        const vec4 clip = viewProjection * vec4(light->getWorldTransform().getPosition(), 1.0f);
        if (clip.w <= 0.0f)
            continue; // behind the camera

        const vec2 ndc = vec2(clip) / clip.w;
        const vec2 center(
            (ndc.x * 0.5f + 0.5f) * static_cast<float>(width),
            (1.0f - (ndc.y * 0.5f + 0.5f)) * static_cast<float>(height));

        const vec2 delta = pixel - center;
        const float distSq = dot(delta, delta);
        if (distSq <= closestDistSq) {
            closestDistSq = distSq;
            closest = light;
        }
    }

    if (!closest)
        return false;

    scene.setActiveObjectId(closest->getId());
    return true;
}

void ViewportPanel::handleObjectPicking() const {
    if (m_showOverlays && handleBillboardPicking())
        return;

    const ivec2 pixel = screenToPixel();

    vk::BufferImageCopy copyRegion{};
    copyRegion.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
    copyRegion.imageOffset = vk::Offset3D{ pixel.x, pixel.y, 0 };
    copyRegion.imageExtent = vk::Extent3D{ 1, 1, 1 };
    
    context.oneTimeSubmit([&](const vk::CommandBuffer cmd) {
       outputCrypto->setImageLayout(cmd, vk::ImageLayout::eTransferSrcOptimal);
       cmd.copyImageToBuffer(outputCrypto->getImage(), vk::ImageLayout::eTransferSrcOptimal, cryptoStagingBuffer.getBuffer(), copyRegion);
       outputCrypto->setImageLayout(cmd, vk::ImageLayout::eGeneral);
   });
    
    uint32_t instanceId = ~0u;
    if (cryptoStagingBufferMappedPtr)
        instanceId = *static_cast<uint32_t*>(cryptoStagingBufferMappedPtr);

    LOG_INFO( "Picked instance ID: " << instanceId);
    
    const auto meshInstances = scene.getMeshInstances();
    if (instanceId != ~0u && instanceId < meshInstances.size()) {
        scene.setActiveObjectId(meshInstances[instanceId]->getId());
    }
    else
        scene.clearActiveObject();
}

void ViewportPanel::setAovImages(Image& crypto, Image& position)
{
    outputCrypto = &crypto;
    outputPosition = &position;
}

void ViewportPanel::onComputeFinished(const vk::CommandBuffer cmd, Image& srcImage) {
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
    if (isCapturingMouse)
        endMouseCapture();
    if (cryptoStagingBufferMappedPtr)
        context.getDevice().unmapMemory(cryptoStagingBuffer.getMemory());
    
    if (positionStagingBufferMappedPtr)
        context.getDevice().unmapMemory(positionStagingBuffer.getMemory());

    LOG_INFO( "Destroying ViewportPanel");
}
