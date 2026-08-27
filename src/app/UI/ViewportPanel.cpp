#include "ViewportPanel.h"
#include <cmath>
#include <iostream>
#include <ranges>

#include "imgui.h"
#include "backends/imgui_impl_vulkan.h"
#include "ImGuizmo.h"
#define IMVIEWGUIZMO_IMPLEMENTATION
#include "ImViewGuizmo.h"
#include "Log.h"
#include "glm/gtc/type_ptr.hpp"
#include "SDL3/SDL_mouse.h"
#include "Rendering/Camera/CameraInstance.h"
#include "Scene/Objects/MeshInstance.h"
#include "Scene/Objects/GaussianInstance.h"
#include "Geometry/Mesh/Assets/GaussianAsset.h"
#include "Scene/Objects/LightInstance.h"
#include "Backend/Vulkan/Raytracer/RaytracerRenderer.h"
#include "Backend/Vulkan/Viewport/Viewport.h"
#include "UI/Window.h"

namespace
{
CameraInstance::InputState cameraInputState()
{
    const ImGuiIO& io = ImGui::GetIO();
    return {
        .deltaTime = io.DeltaTime,
        .accelerate = ImGui::IsKeyDown(ImGuiKey_LeftShift),
        .forward = ImGui::IsKeyDown(ImGuiKey_W),
        .backward = ImGui::IsKeyDown(ImGuiKey_S),
        .left = ImGui::IsKeyDown(ImGuiKey_A),
        .right = ImGui::IsKeyDown(ImGuiKey_D),
        .up = ImGui::IsKeyDown(ImGuiKey_E),
        .down = ImGui::IsKeyDown(ImGuiKey_Q),
    };
}
}

ViewportPanel::ViewportPanel(const std::string& name, Window& window,
    Scene& scene, VulkanRaytracer& raytracer)
    : ImGuiComponent(name), window(window), scene(scene),
    raytracer(raytracer), width(raytracer.width()), height(raytracer.height()),
    uiScale(window.getDpiScale())
{
    const ViewportInputs inputs{raytracer.colorHandle(), raytracer.albedoHandle(),
        raytracer.normalHandle(), raytracer.cryptomatteHandle(),
        raytracer.positionHandle(), raytracer.gaussianOverdrawHandle()};
    compositor = std::make_unique<Viewport>(raytracer.device(), width, height,
        inputs, gpu::ImageFormat::Bgra8Unorm);
    vk::SamplerCreateInfo samplerInfo{};
    samplerInfo.magFilter = vk::Filter::eLinear;
    samplerInfo.minFilter = vk::Filter::eLinear;
    samplerInfo.addressModeU = vk::SamplerAddressMode::eClampToEdge;
    samplerInfo.addressModeV = vk::SamplerAddressMode::eClampToEdge;
    samplerInfo.addressModeW = vk::SamplerAddressMode::eClampToEdge;
    const auto handles = gpu::interop::device_handles(raytracer.device());
    sampler = vk::Device(reinterpret_cast<VkDevice>(handles.device)).createSamplerUnique(samplerInfo);
    
    updateDisplayDescriptor();

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
    const vk::ImageView view(reinterpret_cast<VkImageView>(gpu::interop::image_view(
        raytracer.device(), compositor->getOutputImage().handle())));
    width = compositor->getOutputImage().width();
    height = compositor->getOutputImage().height();
    if (!view || view == observedImageView)
        return;
    if (outputImageDescriptorSet != VK_NULL_HANDLE)
        ImGui_ImplVulkan_RemoveTexture(outputImageDescriptorSet);
    outputImageDescriptorSet = ImGui_ImplVulkan_AddTexture(
        sampler.get(), view, VK_IMAGE_LAYOUT_GENERAL);
    observedImageView = view;
}

void ViewportPanel::recordPresentation()
{
    const ViewportInputs inputs{raytracer.colorHandle(), raytracer.albedoHandle(),
        raytracer.normalHandle(), raytracer.cryptomatteHandle(),
        raytracer.positionHandle(), raytracer.gaussianOverdrawHandle()};
    compositor->resize(raytracer.width(), raytracer.height(), inputs,
        gpu::ImageFormat::Bgra8Unorm);
    compositor->updateBillboards(scene);
    glm::mat4 viewProjection(1.0f);
    if (const CameraInstance* camera = scene.getRenderCamera())
        viewProjection = camera->getProjectionMatrix() * camera->getViewMatrix();
    const RenderSettings& settings = scene.getRenderSettings();
    compositor->dispatch(~0u, viewProjection,
        0.0f, static_cast<int>(settings.bufferVisualization),
        settings.gaussianProxyOverdrawMax,
        settings.tonemappingEnabled, m_showOverlays);
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
    if (width == 0 || height == 0)
        return ivec2(0);

    const ImVec2 screenPos = ImGui::GetMousePos();
    const ImVec2 relativePos = ImVec2(screenPos.x - viewportPos.x, screenPos.y - viewportPos.y);

    const float normX = std::clamp(relativePos.x / viewportSize.x, 0.f, 1.f);
    const float normY = std::clamp(relativePos.y / viewportSize.y, 0.f, 1.f);

    // The rightmost/bottommost half pixel maps to width/height, which is one
    // past the last valid texel. AOV readbacks copy from renderer-owned images, so
    // an out-of-bounds copy region reads past the shared allocation.
    const int pixelX = std::clamp(static_cast<int>(normX * static_cast<float>(width)),
        0, static_cast<int>(width) - 1);
    const int pixelY = std::clamp(static_cast<int>((1.0f - normY)
        * static_cast<float>(height)),
        0, static_cast<int>(height) - 1);

    return ivec2(pixelX, pixelY);
}

void ViewportPanel::drawBackground() const {
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    const float x0 = viewportPos.x, y0 = viewportPos.y;
    const float x1 = viewportPos.x + viewportSize.x, y1 = viewportPos.y + viewportSize.y;
    drawList->AddRectFilled({x0, y0}, {x1, y1}, IM_COL32(42, 42, 44, 255));
}

void ViewportPanel::drawImageAndUpdateState() {
    updateDisplayDescriptor();
    // Raygen stores rows in the same bottom-left convention as the former
    // renderer output. ImGui texture UVs are top-left, so presentation alone
    // flips V; the render and camera math remain unchanged.
    ImGui::Image(reinterpret_cast<ImTextureID>(outputImageDescriptorSet),
        viewportSize, ImVec2(0.0f, 1.0f), ImVec2(1.0f, 0.0f));
    isViewportHovered = ImGui::IsItemHovered();
}

bool ViewportPanel::processEvent(const SDL_Event& event)
{
    const bool wasCapturing = isCapturingMouse;
    bool viewportPress = false;
    if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN &&
        event.button.button == SDL_BUTTON_RIGHT) {
        rightButtonDown = true;
        rightButtonPressPending = true;
        rightButtonPressX = event.button.x;
        rightButtonPressY = event.button.y;
        // SDL events arrive before the next ImGui frame is rendered.  Use the
        // hover state recorded from the actual image item as an additional
        // guard; the panel/window bounds alone are stale when another docked
        // tab is covering the viewport.
        viewportPress = isViewportHovered && rightButtonPressX >= viewportPos.x &&
            rightButtonPressY >= viewportPos.y &&
            rightButtonPressX < viewportPos.x + viewportSize.x &&
            rightButtonPressY < viewportPos.y + viewportSize.y;
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

    const bool mouseEvent = event.type == SDL_EVENT_MOUSE_MOTION ||
        event.type == SDL_EVENT_MOUSE_BUTTON_DOWN ||
        event.type == SDL_EVENT_MOUSE_BUTTON_UP ||
        event.type == SDL_EVENT_MOUSE_WHEEL;
    return (wasCapturing && mouseEvent) || viewportPress;
}

void ViewportPanel::beginMouseCapture() {
    if (isCapturingMouse)
        return;
    SDL_GetMouseState(&oldX, &oldY);
    if (!window.setRelativeMouseMode(true)) {
        LOG_ERROR("Failed to enable relative mouse mode: " << SDL_GetError());
        return;
    }
    ImGuiIO& io = ImGui::GetIO();
    imguiMouseWasDisabled = (io.ConfigFlags & ImGuiConfigFlags_NoMouse) != 0;
    io.ConfigFlags |= ImGuiConfigFlags_NoMouse;
    isCapturingMouse = true;
}

void ViewportPanel::endMouseCapture() {
    if (!isCapturingMouse)
        return;
    isCapturingMouse = false;
    if (!window.setRelativeMouseMode(false))
        LOG_ERROR("Failed to disable relative mouse mode: " << SDL_GetError());
    window.warpMouse(oldX, oldY);
    if (!imguiMouseWasDisabled)
        ImGui::GetIO().ConfigFlags &= ~ImGuiConfigFlags_NoMouse;
    imguiMouseWasDisabled = false;

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
            camera->update(pendingMouseDeltaX, pendingMouseDeltaY, cameraInputState());
        endMouseCapture();
        rightButtonPressPending = false;
        return; // Consume the event, don't start a new action
    }

    if (!isViewportHovered && !rightButtonPressPending)
        return;

    if (isViewportHovered)
        handleScrollZoom();

    // Use SDL's ordered button edge rather than ImGui's frame-delayed click edge.
    const bool pressIsInViewport = isViewportHovered &&
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
    GaussianInstance* gaussianInstance = dynamic_cast<GaussianInstance*>(activeObject.get());
    Gaussian gaussian{};
    uint32_t gaussianLocalIndex = ~0u;
    if (gaussianInstance && selectedGaussianIndex != ~0u)
    {
        uint32_t offset = 0;
        for (const auto& instance : scene.getGaussianInstances())
        {
            const uint32_t count = instance->getGaussianAsset().getGaussianCount();
            if (instance.get() == gaussianInstance && selectedGaussianIndex >= offset
                && selectedGaussianIndex < offset + count)
            {
                gaussianLocalIndex = selectedGaussianIndex - offset;
                gaussian = instance->getGaussianAsset().getGaussian(gaussianLocalIndex);
                mat4 gaussianModel(1.0f);
                gaussianModel[0] = glm::vec4(gaussian.transform[0], 0.0f);
                gaussianModel[1] = glm::vec4(gaussian.transform[1], 0.0f);
                gaussianModel[2] = glm::vec4(gaussian.transform[2], 0.0f);
                gaussianModel[3] = glm::vec4(gaussian.transform[3], 1.0f);
                model = instance->getWorldTransform().getMatrix()
                    * gaussianModel;
                break;
            }
            offset += count;
        }
    }

    if (ImGuizmo::Manipulate(value_ptr(view), value_ptr(proj), currentOperation, currentMode, value_ptr(model)))
    {
        if (gaussianInstance && gaussianLocalIndex != ~0u)
        {
            const mat4 local = inverse(gaussianInstance->getWorldTransform().getMatrix()) * model;
            gaussian.transform = glm::mat4x3(
                glm::vec3(local[0]), glm::vec3(local[1]),
                glm::vec3(local[2]), glm::vec3(local[3]));
            gaussianInstance->getGaussianAsset().setGaussian(gaussianLocalIndex, gaussian);
        }
        else
            activeObject->setWorldTransformFromMatrix(model);
    }
}

void ViewportPanel::handleViewGizmo() const {
    auto* camera = scene.getRenderCamera();
    if (!camera)
        return;

    vec3 position = camera->getPosition();
    quat rotation = camera->getRotation();

    vec3 pivot = position + normalize(rotation * CameraInstance::LocalForward) * 5.0f;
    if (const auto activeObject = scene.getActiveObjectPtr();
        activeObject && activeObject.get() != camera)
        pivot = activeObject->getPosition();

    ImViewGuizmo::BeginFrame();
    bool wasModified = false;

    // ImViewGuizmo draws straight to the window draw list with no clipping of its
    // own (unlike ImGuizmo::Manipulate), so on a small/narrow viewport its fixed
    // pixel offsets can push it outside the rendered image. Clip to the viewport
    // image bounds, same as renderToolbar() does.
    // Draw-list clipping, never ImGui::PushClipRect: the ImGui clip stack also
    // governs item hit-testing, and these gizmos leave it unbalanced, so every
    // widget submitted after the viewport in the same window silently stops
    // responding to the mouse. Invisible while the viewport owns its own ImGui
    // window, fatal the moment it shares one with other controls.
    ImDrawList* viewGizmoDrawList = ImGui::GetWindowDrawList();
    viewGizmoDrawList->PushClipRect(viewportPos, ImVec2(viewportPos.x + viewportSize.x, viewportPos.y + viewportSize.y), true);

    ImVec2 gizmoPos = {viewportPos.x + viewportSize.x - 110.f * uiScale, viewportPos.y + 110.f * uiScale};
    wasModified |= ImViewGuizmo::Rotate(position, rotation, pivot, gizmoPos);

    gizmoPos.x += 30.f * uiScale; gizmoPos.y += 90.f * uiScale;
    wasModified |= ImViewGuizmo::Dolly(position, rotation, gizmoPos);

    gizmoPos.y += 60.f * uiScale;
    wasModified |= ImViewGuizmo::Pan(position, rotation, gizmoPos);

    viewGizmoDrawList->PopClipRect();

    if (wasModified && !ImGuizmo::IsUsing()) {
        camera->setPosition(position);
        camera->setRotation(rotation);
    }
}

void ViewportPanel::renderToolbar() {
    // Clip to the viewport image bounds, same as ImGuizmo::Manipulate does
    // internally, so the buttons can't spill outside the rendered image.
    // Draw-list level: an ImGui::PushClipRect here would also affect item
    // hit-testing for everything submitted afterwards.
    ImDrawList* toolbarDrawList = ImGui::GetWindowDrawList();
    toolbarDrawList->PushClipRect(viewportPos, ImVec2(viewportPos.x + viewportSize.x, viewportPos.y + viewportSize.y), true);

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

    toolbarDrawList->PopClipRect();
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
            camera->update(pendingMouseDeltaX, pendingMouseDeltaY, cameraInputState());
    }
    pendingMouseDeltaX = 0.f;
    pendingMouseDeltaY = 0.f;
    
    ImGui::End();
}

void ViewportPanel::handlePositionPicking() const {

    auto* camera = scene.getRenderCamera();
    if (!camera)
        return;

    const ivec2 pixel = screenToPixel();
    const std::vector<gpu::float4> positions = raytracer.readPosition();
    const size_t index = static_cast<size_t>(pixel.y) * width + pixel.x;
    if (index >= positions.size())
        return;
    const gpu::float4 value = positions[index];
    const vec3 position(value.x, value.y, value.z);

    LOG_INFO( "Picked Position: (" << position.x << ", " << position.y << ", " << position.z << ")");
    
    camera->setArcballPivot(position);
    camera->setArcballActive(true);
}


bool ViewportPanel::handleBillboardPicking() const {
    auto* camera = scene.getRenderCamera();
    if (!camera)
        return false;

    // Match the visual icon so hit-testing scales with the billboard.
    constexpr float pickRadius = ViewportBillboardPixelRadius;

    const vec2 pixel = vec2(screenToPixel());
    const mat4 viewProjection = camera->getProjectionMatrix() * camera->getViewMatrix();

    std::shared_ptr<LightInstance> closest;
    float closestDistSq = pickRadius * pickRadius;

    for (const auto& obj : scene.getSceneObjects()) {
        const auto light = std::dynamic_pointer_cast<LightInstance>(obj);
        if (!light)
            continue;

        vec2 center;
        if (!projectViewportBillboard(viewProjection,
                light->getWorldTransform().getPosition(), width, height, center))
            continue; // behind the camera

        const vec2 delta = pixel - center;
        const float distSq = dot(delta, delta);
        if (distSq <= closestDistSq) {
            closestDistSq = distSq;
            closest = light;
        }
    }

    if (!closest)
        return false;

    scene.setActiveObject(closest->getHandle());
    return true;
}

void ViewportPanel::handleObjectPicking() {
    if (m_showOverlays && handleBillboardPicking())
        return;

    const ivec2 pixel = screenToPixel();
    const std::vector<uint32_t> ids = raytracer.readCryptomatte();
    const size_t pixelIndex = static_cast<size_t>(pixel.y) * width + pixel.x;
    const uint32_t instanceId = pixelIndex < ids.size() ? ids[pixelIndex] : ~0u;

    LOG_INFO( "Picked instance ID: " << instanceId);
    
    const auto meshInstances = scene.getMeshInstances();
    if (instanceId != ~0u && instanceId < meshInstances.size()) {
        selectedGaussianIndex = ~0u;
        scene.setActiveObject(meshInstances[instanceId]->getHandle());
    }
    else if (instanceId != ~0u && instanceId >= meshInstances.size()) {
        const uint32_t gaussianInstanceIndex = instanceId
            - static_cast<uint32_t>(meshInstances.size());
        const auto& gaussianInstances = scene.getGaussianInstances();
        uint32_t offset = 0;
        for (const auto& gaussian : gaussianInstances) {
            const uint32_t count = gaussian->getGaussianAsset().getGaussianCount();
            if (gaussianInstanceIndex < offset + count) {
                selectedGaussianIndex = gaussianInstanceIndex;
                scene.setActiveObject(gaussian->getHandle());
                return;
            }
            offset += count;
        }
        selectedGaussianIndex = ~0u;
        scene.clearActiveObject();
    }
    else {
        selectedGaussianIndex = ~0u;
        scene.clearActiveObject();
    }
}

ViewportPanel::~ViewportPanel() {
    if (isCapturingMouse)
        endMouseCapture();
    if (outputImageDescriptorSet != VK_NULL_HANDLE)
        ImGui_ImplVulkan_RemoveTexture(outputImageDescriptorSet);

    LOG_INFO( "Destroying ViewportPanel");
}
