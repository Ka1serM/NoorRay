#include "ViewportViewModel.h"
#include "Log.h"
#include "Camera/PerspectiveCamera.h"
#include "Vulkan/Context.h"
#include <imgui.h>
#include "backends/imgui_impl_sdl3.cpp"
#include "backends/imgui_impl_vulkan.h"
#include <array>
#define IMVIEWGUIZMO_IMPLEMENTATION
#include "ImViewGuizmo.h"
#include "Vulkan/Renderer.h"

ViewportViewModel::ViewportViewModel(Context& appContext, Scene& scene, Rml::Context* rmlContext, RmlRenderInterface& renderInterface, Image& renderImage, Image& cryptoImage,  Image& positionImage, const std::string& elementId)
: ViewModelBase(rmlContext, elementId),
    renderInterface(renderInterface),
    context(appContext),
    scene(scene),
    cryptoImage(cryptoImage),
    positionImage(positionImage),
    cryptoStagingBuffer(context, Buffer::Type::Custom, sizeof(uint32_t), nullptr, vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent),
    positionStagingBuffer(context, Buffer::Type::Custom, sizeof(float) * 4, nullptr, vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent),
    imGuiImage(appContext, renderImage.getWidth(), renderImage.getHeight(), renderImage.getFormat(), vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled)
{
    rmlUIContext = rmlContext;

    uiScale = std::max(renderImage.getWidth() / 1080.0f, 0.5f);
    
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

#ifdef NDEBUG // Release mode: embed ini into binary
    io.IniFilename = nullptr;  // don't use external file
    static constexpr char ini[] = {
        #embed "../../../../assets/imgui.ini"
    };
    ImGui::LoadIniSettingsFromMemory(ini, sizeof(ini));
#else
    // Debug mode: use external file for easy tweaking
    io.IniFilename = "imgui.ini";  // ImGui will load and save this file
#endif

    static constexpr unsigned char font[] = {
        #embed "../../../../assets/fonts/Inter.ttf"
    };
    ImFontConfig font_config;
    font_config.FontDataOwnedByAtlas = false;

    const float font_size = 18.0f * context.getDPIScale();
    io.Fonts->AddFontFromMemoryTTF(const_cast<unsigned char*>(font), sizeof(font), font_size, &font_config);

    ImGuiStyle& imGuiStyle = ImGui::GetStyle();
    imGuiStyle.ScaleAllSizes(context.getDPIScale());

    applyImGuiTheme();
    
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = context.getInstance();
    init_info.PhysicalDevice = context.getPhysicalDevice();
    init_info.Device = context.getDevice();
    init_info.QueueFamily = context.getGraphicsFamilyIndex();
    init_info.Queue = context.getGraphicsQueue();
    init_info.DescriptorPool = context.getDescriptorPool();
    init_info.MinImageCount = Renderer::MAX_FRAMES_IN_FLIGHT;
    init_info.ImageCount = Renderer::MAX_FRAMES_IN_FLIGHT; //
    init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    
    init_info.UseDynamicRendering = true;
    init_info.PipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR;
    init_info.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    const auto renderFormat = static_cast<VkFormat>(imGuiImage.getFormat());
    init_info.PipelineRenderingCreateInfo.pColorAttachmentFormats = &renderFormat;
    
    ImGui_ImplVulkan_Init(&init_info);
    
    // Map memory
    const vk::MemoryRequirements cryptoReq = context.getDevice().getBufferMemoryRequirements(cryptoStagingBuffer.getBuffer());
    cryptoStagingBufferMappedPtr = context.getDevice().mapMemory(cryptoStagingBuffer.getMemory(), 0, cryptoReq.size);

    const vk::MemoryRequirements positionReq = context.getDevice().getBufferMemoryRequirements(positionStagingBuffer.getBuffer());
    positionStagingBufferMappedPtr = context.getDevice().mapMemory(positionStagingBuffer.getMemory(), 0, positionReq.size);
    
    Rml::Element* element = rmlUIContext->GetDocument("main")->GetElementById(elementId);
    const Rml::String viewportOverlay = "viewport-overlay";
    imGuiElement = element->GetElementById(viewportOverlay);
    renderInterface.registerVulkanTexture(viewportOverlay, imGuiImage.getView(), Rml::Vector2i{static_cast<int>(imGuiImage.getWidth()), static_cast<int>(imGuiImage.getHeight())});
    imGuiElement->SetAttribute("src", "vulkan://" + viewportOverlay);

    const Rml::String viewportImg = "viewport-img";
    renderInterface.registerVulkanTexture(viewportImg, renderImage.getView(), Rml::Vector2i{static_cast<int>(renderImage.getWidth()), static_cast<int>(renderImage.getHeight())});
    element->GetElementById(viewportImg)->SetAttribute("src", "vulkan://" + viewportImg);
    
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
    
    style.TranslationLineThickness = 4.0f * uiScale;
    style.TranslationLineArrowSize = 6.0f * uiScale;
    style.RotationLineThickness = 6.0f * uiScale;
    style.RotationOuterLineThickness = 2.0f * uiScale;
    style.ScaleLineThickness = 4.0f * uiScale;
    style.ScaleLineCircleSize = 8.0f * uiScale;
    style.CenterCircleSize = 5.0f * uiScale;
}

bool ViewportViewModel::isMouseOverViewport() const
{
    const Rml::Element* hoveredElement = rmlUIContext->GetHoverElement();
    if (!hoveredElement)
        return false;
    
    if (hoveredElement == imGuiElement)
        return true;

    return false;
}

void ViewportViewModel::addMouseInput(ImGuiIO& io)
{
    float mouse_x, mouse_y;
    SDL_GetMouseState(&mouse_x, &mouse_y);
    const ImVec2 mouse_pos_global(mouse_x, mouse_y);
    // Translate
    const ImVec2 mouse_pos_relative(mouse_pos_global.x - viewportPos.x, mouse_pos_global.y - viewportPos.y);
    // Scale
    if (viewportSize.x > 0 && viewportSize.y > 0) {
        const ImVec2 scale_factor(
            static_cast<float>(imGuiImage.getWidth()) / viewportSize.x,
            static_cast<float>(imGuiImage.getHeight()) /  viewportSize.y
        );
        const ImVec2 remapped_mouse_pos(mouse_pos_relative.x * scale_factor.x, mouse_pos_relative.y * scale_factor.y);
        io.AddMousePosEvent(remapped_mouse_pos.x, remapped_mouse_pos.y);
    }
}

void ViewportViewModel::render(const vk::CommandBuffer commandBuffer) {
    if (!imGuiElement)
        return;
    
    const Rml::Vector2f layout_offset = imGuiElement->GetAbsoluteOffset();
    const Rml::Vector2f size = imGuiElement->GetBox().GetSize();
    // Get actual top-left corner
    const Rml::Vector2f visual_offset = {
        layout_offset.x - size.x * 0.5f,
        layout_offset.y - size.y * 0.5f
    };
    viewportPos = ImVec2{ visual_offset.x, visual_offset.y};
    viewportSize = ImVec2{ size.x,size.y };

    ImGui_ImplVulkan_NewFrame();

    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(static_cast<float>(imGuiImage.getWidth()), static_cast<float>(imGuiImage.getHeight()));
    io.DisplayFramebufferScale = ImVec2(1.0f, 1.0f);
    
    //TODO
    addMouseInput(io);

    ImGui::NewFrame();

    const ImGuiViewport* mainViewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(mainViewport->Pos);
    ImGui::SetNextWindowSize(mainViewport->Size);
    ImGui::SetNextWindowViewport(mainViewport->ID);

        constexpr ImGuiWindowFlags overlayFlags =
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
        ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoDocking |
        ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBringToFrontOnFocus;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0)); // fully transparent
    ImGui::Begin("ViewportOverlay", nullptr, overlayFlags);
    
    renderToolbar();
    renderViewGizmo();
    renderTransformGizmo();
    
    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar();

    ImGui::Render();
    
    imGuiImage.setImageLayout(commandBuffer, vk::ImageLayout::eColorAttachmentOptimal);

    // Dynamic rendering
    vk::RenderingAttachmentInfo colorAttachment{};
    colorAttachment.setImageView(imGuiImage.getView());
    colorAttachment.setImageLayout(vk::ImageLayout::eColorAttachmentOptimal);
    colorAttachment.setLoadOp(vk::AttachmentLoadOp::eClear);
    colorAttachment.setStoreOp(vk::AttachmentStoreOp::eStore);
    colorAttachment.setClearValue(vk::ClearValue{std::array{0.0f, 0.0f, 0.0f, 0.0f}});

    vk::RenderingInfo renderingInfo{};
    renderingInfo.setRenderArea(vk::Rect2D({0, 0},  vk::Extent2D{imGuiImage.getWidth(), imGuiImage.getHeight()}));
    renderingInfo.setLayerCount(1);
    renderingInfo.setColorAttachments(colorAttachment);

    commandBuffer.beginRendering(renderingInfo);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);
    commandBuffer.endRendering();
    
    imGuiImage.setImageLayout(commandBuffer, vk::ImageLayout::eShaderReadOnlyOptimal);
}

void ViewportViewModel::renderToolbar() {
    const ImVec2 toolbarOffset(25.0f * uiScale, 25.0f * uiScale);
    ImGui::SetCursorPos(toolbarOffset);

    const float buttonSize = 50.0f * uiScale;
    const ImVec2 buttonVecSize(buttonSize, buttonSize);
    const float rounding = 10.0f * uiScale;
    const float spacing = 5.0f * uiScale;

    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, rounding);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.56f, 0.56f, 0.56f, 0.20f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.84f, 0.84f, 0.84f, 0.20f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.84f, 0.84f, 0.84f, 0.39f));

    ImGui::BeginGroup();

    // Translate Button
    const bool isActiveT = (currentOperation == ImGuizmo::TRANSLATE);
    if (isActiveT) ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyle().Colors[ImGuiCol_ButtonActive]);
    if (ImGui::Button("T", buttonVecSize)) currentOperation = ImGuizmo::TRANSLATE;
    if (isActiveT) ImGui::PopStyleColor();

    ImGui::Dummy(ImVec2(0.0f, spacing));

    // Rotate Button
    const bool isActiveR = (currentOperation == ImGuizmo::ROTATE);
    if (isActiveR) ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyle().Colors[ImGuiCol_ButtonActive]);
    if (ImGui::Button("R", buttonVecSize)) currentOperation = ImGuizmo::ROTATE;
    if (isActiveR) ImGui::PopStyleColor();

    ImGui::Dummy(ImVec2(0.0f, spacing));

    // Scale Button
    const bool isActiveS = (currentOperation == ImGuizmo::SCALE);
    if (isActiveS) ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyle().Colors[ImGuiCol_ButtonActive]);
    if (ImGui::Button("S", buttonVecSize)) currentOperation = ImGuizmo::SCALE;
    if (isActiveS) ImGui::PopStyleColor();

    ImGui::EndGroup();

    ImGui::PopStyleColor(3);
    ImGui::PopStyleVar(2);
}

void ViewportViewModel::renderTransformGizmo() {
    const auto activeObject = scene.getActiveObject();
    if (!activeObject)
        return;

    auto* camera = scene.getActiveCamera();
    
    ImGuizmo::BeginFrame();
    ImGuizmo::SetRect(0, 0, static_cast<float>(imGuiImage.getWidth()), static_cast<float>(imGuiImage.getHeight()));
    ImGuizmo::SetDrawlist(ImGui::GetWindowDrawList());

    if (ImGui::IsKeyPressed(ImGuiKey_W))
        currentOperation = ImGuizmo::TRANSLATE;
    if (ImGui::IsKeyPressed(ImGuiKey_E))
        currentOperation = ImGuizmo::ROTATE;
    if (ImGui::IsKeyPressed(ImGuiKey_R))
        currentOperation = ImGuizmo::SCALE;

    const mat4& view = camera->getViewMatrix();
    mat4 proj = camera->getProjectionMatrix();
    mat4 model = activeObject->getWorldTransform().getMatrix();

    if (ImGuizmo::Manipulate(value_ptr(view), value_ptr(proj), currentOperation, currentMode, value_ptr(model)))
        activeObject->setWorldTransformFromMatrix(model);
}

void ViewportViewModel::renderViewGizmo() {
    auto* camera = scene.getActiveCamera();
    if (!camera) return;

    ImViewGuizmo::Style& style = ImViewGuizmo::GetStyle();
    style.scale = uiScale;

    vec3 position = camera->getPosition();
    quat rotation = camera->getRotation();
    
    vec3 pivot = scene.getActiveObject() ? scene.getActiveObject()->getTransform().getPosition() : vec3(0.0f);

    ImViewGuizmo::BeginFrame();

    const float gizmoSize = 110.f * uiScale;
    ImVec2 gizmoPos = {ImGui::GetMainViewport()->Size.x - gizmoSize, gizmoSize};
    bool wasModified = ImViewGuizmo::Rotate(position, rotation, pivot, gizmoPos);
    
    gizmoPos.x += 30.f * uiScale; 
    gizmoPos.y += 90.f *uiScale;
    wasModified |= ImViewGuizmo::Dolly(position, rotation, gizmoPos);
    
    gizmoPos.y += 60.f * uiScale;
    wasModified |= ImViewGuizmo::Pan(position, rotation, gizmoPos);

    if (wasModified && !ImGuizmo::IsUsing()) {
        camera->setPosition(position);
        camera->setRotation(rotation);
    }
}

void ViewportViewModel::processEvent(const SDL_Event& event) {
    ImGuiIO& io = ImGui::GetIO();
    switch (event.type)
    {
    case SDL_EVENT_MOUSE_BUTTON_DOWN:
    case SDL_EVENT_MOUSE_BUTTON_UP:
        {
            int mouse_button = -1;
            if (event.button.button == SDL_BUTTON_LEFT) { mouse_button = 0; }
            if (event.button.button == SDL_BUTTON_RIGHT) { mouse_button = 1; }
            if (event.button.button == SDL_BUTTON_MIDDLE) { mouse_button = 2; }
            if (mouse_button != -1)
                io.AddMouseButtonEvent(mouse_button, (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN));
            return;
        }

    case SDL_EVENT_MOUSE_WHEEL:
        {
            io.AddMouseWheelEvent(event.wheel.x, event.wheel.y);
            return;
        }

    case SDL_EVENT_TEXT_INPUT:
        {
            io.AddInputCharactersUTF8(event.text.text);
            return;
        }
    case SDL_EVENT_KEY_DOWN:
    case SDL_EVENT_KEY_UP:
        {
            io.AddKeyEvent(ImGuiMod_Ctrl, (event.key.mod & SDL_KMOD_CTRL) != 0);
            io.AddKeyEvent(ImGuiMod_Shift, (event.key.mod & SDL_KMOD_SHIFT) != 0);
            io.AddKeyEvent(ImGuiMod_Alt, (event.key.mod & SDL_KMOD_ALT) != 0);
            io.AddKeyEvent(ImGuiMod_Super, (event.key.mod & SDL_KMOD_GUI) != 0);

            const ImGuiKey key = ImGui_ImplSDL3_KeyEventToImGuiKey(event.key.key, event.key.scancode);
            io.AddKeyEvent(key, (event.type == SDL_EVENT_KEY_DOWN));
            return;
        }
    case SDL_EVENT_WINDOW_FOCUS_GAINED:
        io.AddFocusEvent(true);
        return;
    case SDL_EVENT_WINDOW_FOCUS_LOST:
        io.AddFocusEvent(false);
    default: ;
    }
}

void ViewportViewModel::handleObjectPicking(const ImVec2& mousePos)
{
    vk::BufferImageCopy copyRegion{};
    copyRegion.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
    copyRegion.imageOffset = vk::Offset3D{ static_cast<int>(mousePos.x), static_cast<int>(mousePos.y), 0 };
    copyRegion.imageExtent = vk::Extent3D{ 1, 1, 1 };
    
    context.oneTimeSubmit([&](const vk::CommandBuffer commandBuffer) {
       cryptoImage.setImageLayout(commandBuffer, vk::ImageLayout::eTransferSrcOptimal);
       commandBuffer.copyImageToBuffer(cryptoImage.getImage(), vk::ImageLayout::eTransferSrcOptimal, cryptoStagingBuffer.getBuffer(), copyRegion);
       cryptoImage.setImageLayout(commandBuffer, vk::ImageLayout::eGeneral); 
   });
    
    uint32_t instanceId = 0; // Use a known invalid ID like 0
    if (cryptoStagingBufferMappedPtr)
        instanceId = *static_cast<uint32_t*>(cryptoStagingBufferMappedPtr);

    LOG_INFO("Picked instance ID: " << instanceId);
    
    // Example logic to set active object - adjust as needed
    // if (instanceId != 0 && instanceId <= scene.getMeshInstances().size())
    //    scene.setActiveObject(scene.getMeshInstances()[instanceId - 1]->getId());
}

void ViewportViewModel::handlePositionPicking(const ImVec2& mousePos)
{
    auto* camera = scene.getActiveCamera();
    if (!camera) return;

    vk::BufferImageCopy copyRegion{};
    copyRegion.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
    copyRegion.imageOffset = vk::Offset3D{static_cast<int>(mousePos.x), static_cast<int>(mousePos.y), 0};
    copyRegion.imageExtent = vk::Extent3D{1, 1, 1};

    context.oneTimeSubmit([&](const vk::CommandBuffer commandBuffer) {
        positionImage.setImageLayout(commandBuffer, vk::ImageLayout::eTransferSrcOptimal);
        commandBuffer.copyImageToBuffer(positionImage.getImage(), vk::ImageLayout::eTransferSrcOptimal, positionStagingBuffer.getBuffer(), copyRegion);
        positionImage.setImageLayout(commandBuffer, vk::ImageLayout::eGeneral);
    });

    vec3 position{0.f};
    if (positionStagingBufferMappedPtr) {
        const float* f = static_cast<float*>(positionStagingBufferMappedPtr);
        position = vec3(f[0], f[1], f[2]);
    }

    LOG_INFO( "Picked Position: (" << position.x << ", " << position.y << ", " << position.z << ")");
    
    // Example logic for arcball camera - adjust as needed
    // camera->setArcballPivot(position);
    // camera->setArcballActive(true);
}

void ViewportViewModel::applyImGuiTheme() {
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding    = 6.0f;
    style.ChildRounding     = 6.0f;
    style.PopupRounding     = 6.0f;
    style.FrameRounding     = 6.0f;
    style.GrabRounding      = 6.0f;
    style.TabRounding       = 6.0f;
    style.ScrollbarRounding = 6.0f;
    style.FrameBorderSize   = 1.0f;
    style.PopupBorderSize   = 1.0f;
    style.ScrollbarSize     = 12.0f;
    style.GrabMinSize       = 12.0f;
    style.WindowPadding     = ImVec2(12, 12);
    style.FramePadding      = ImVec2(6, 4);
    style.ItemSpacing       = ImVec2(6, 4);
    style.ItemInnerSpacing  = ImVec2(6, 4);
    ImVec4* colors = style.Colors;
    colors[ImGuiCol_WindowBg]           = ImVec4(0.15f, 0.15f, 0.16f, 1.00f);
    colors[ImGuiCol_ChildBg]            = ImVec4(0.13f, 0.13f, 0.14f, 1.00f);
    colors[ImGuiCol_PopupBg]            = ImVec4(0.11f, 0.11f, 0.11f, 1.00f);
    colors[ImGuiCol_MenuBarBg]          = ImVec4(0.11f, 0.11f, 0.11f, 1.00f);
    colors[ImGuiCol_Header]             = ImVec4(0.20f, 0.20f, 0.21f, 1.00f);
    colors[ImGuiCol_HeaderHovered]      = ImVec4(0.30f, 0.30f, 0.31f, 1.00f);
    colors[ImGuiCol_HeaderActive]       = ImVec4(0.25f, 0.25f, 0.26f, 1.00f);
    colors[ImGuiCol_Button]             = ImVec4(0.18f, 0.18f, 0.19f, 1.00f);
    colors[ImGuiCol_ButtonHovered]      = ImVec4(0.24f, 0.24f, 0.26f, 1.00f);
    colors[ImGuiCol_ButtonActive]       = ImVec4(0.28f, 0.28f, 0.30f, 1.00f);
    colors[ImGuiCol_FrameBg]            = ImVec4(0.20f, 0.20f, 0.21f, 1.00f);
    colors[ImGuiCol_FrameBgHovered]     = ImVec4(0.30f, 0.30f, 0.31f, 1.00f);
    colors[ImGuiCol_FrameBgActive]      = ImVec4(0.35f, 0.35f, 0.36f, 1.00f);
    colors[ImGuiCol_Tab]                = ImVec4(0.16f, 0.16f, 0.18f, 1.00f);
    colors[ImGuiCol_TabHovered]         = ImVec4(0.22f, 0.22f, 0.24f, 1.00f);
    colors[ImGuiCol_TabActive]          = ImVec4(0.19f, 0.19f, 0.21f, 1.00f);
    colors[ImGuiCol_TabUnfocused]       = ImVec4(0.12f, 0.12f, 0.14f, 1.00f);
    colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.16f, 0.16f, 0.18f, 1.00f);
    colors[ImGuiCol_TitleBg]            = ImVec4(0.13f, 0.13f, 0.14f, 1.00f);
    colors[ImGuiCol_TitleBgActive]      = ImVec4(0.15f, 0.15f, 0.17f, 1.00f);
    colors[ImGuiCol_TitleBgCollapsed]   = ImVec4(0.10f, 0.10f, 0.12f, 1.00f);
    colors[ImGuiCol_Border]             = ImVec4(0.10f, 0.10f, 0.10f, 0.40f);
    colors[ImGuiCol_BorderShadow]       = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_Text]               = ImVec4(0.90f, 0.90f, 0.92f, 1.00f);
    colors[ImGuiCol_TextDisabled]       = ImVec4(0.50f, 0.50f, 0.52f, 1.00f);
    colors[ImGuiCol_CheckMark]          = ImVec4(0.75f, 0.75f, 0.75f, 1.00f);
    colors[ImGuiCol_SliderGrab]         = ImVec4(0.65f, 0.65f, 0.65f, 1.00f);
    colors[ImGuiCol_SliderGrabActive]   = ImVec4(0.80f, 0.80f, 0.80f, 1.00f);
    colors[ImGuiCol_ResizeGrip]         = ImVec4(0.65f, 0.65f, 0.65f, 0.60f);
    colors[ImGuiCol_ResizeGripHovered]  = ImVec4(0.75f, 0.75f, 0.75f, 0.80f);
    colors[ImGuiCol_ResizeGripActive]   = ImVec4(0.85f, 0.85f, 0.85f, 1.00f);
    colors[ImGuiCol_ScrollbarBg]        = ImVec4(0.10f, 0.10f, 0.12f, 1.00f);
    colors[ImGuiCol_ScrollbarGrab]      = ImVec4(0.20f, 0.20f, 0.25f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabHovered]=ImVec4(0.25f, 0.25f, 0.30f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabActive]= ImVec4(0.30f, 0.30f, 0.35f, 1.00f);
}

ViewportViewModel::~ViewportViewModel()
{
    if (cryptoStagingBufferMappedPtr)
        context.getDevice().unmapMemory(cryptoStagingBuffer.getMemory());
    
    if (positionStagingBufferMappedPtr)
        context.getDevice().unmapMemory(positionStagingBuffer.getMemory());

    ImGui_ImplVulkan_Shutdown();
    ImGui::DestroyContext();
    LOG_INFO( "Destroyed ViewportViewModel");
}
