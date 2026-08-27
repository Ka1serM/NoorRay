#include "ImGuiManager.h"
#include "ImGuiComponent.h"
#include <imgui.h>
#include "imgui_internal.h"
#include "backends/imgui_impl_sdl3.h"
#include "backends/imgui_impl_vulkan.h"
#include "glm/gtc/type_ptr.inl"
#include <gpu/interop.hpp>
#include "UI/Window.h"
#include <array>
#include <cstddef>
#include "Log.h"

namespace
{
constexpr unsigned char noorRayImGuiFont[] = {
    #embed "../../../assets/fonts/Inter.ttf"
};
constexpr std::size_t noorRayImGuiFontLength = sizeof(noorRayImGuiFont);

constexpr unsigned char noorRayImGuiLayout[] = {
    #embed "../../../assets/imgui.ini"
};
constexpr std::size_t noorRayImGuiLayoutLength = sizeof(noorRayImGuiLayout);
}

ImGuiManager::ImGuiManager(Window& window, gpu::Device& device, const uint32_t numImages,
    const gpu::ImageFormat targetFormat)
    : device(device)
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    // Laptop-friendly node-editor navigation: right-drag pans the canvas.
    // SDL touchpad pinch gestures are delivered as ImGui mouse-wheel events,
    // so they use the same navigation path as a physical scroll wheel.
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigDragClickToInputText = true;

    // The application layout is an asset, not per-working-directory state.
    // Loading ./imgui.ini in debug builds made a stale local single-pane
    // layout silently replace the embedded editor arrangement.
    io.IniFilename = nullptr;
    ImGui::LoadIniSettingsFromMemory(
        reinterpret_cast<const char*>(noorRayImGuiLayout), noorRayImGuiLayoutLength);

    ImFontConfig font_config;
    font_config.FontDataOwnedByAtlas = false;
    font_config.RasterizerDensity = window.getDpiScale();

    constexpr float font_size = 18.0f;
    io.Fonts->AddFontFromMemoryTTF(
        const_cast<unsigned char*>(noorRayImGuiFont), noorRayImGuiFontLength,
        font_size, &font_config);

    if (const SDL_SystemTheme theme = SDL_GetSystemTheme(); theme == SDL_SYSTEM_THEME_LIGHT)
        SetTheme(Theme::Light);
    else
        SetTheme(Theme::Dark);

    ImGui_ImplSDL3_InitForVulkan(window.nativeHandle());

    const auto handles = gpu::interop::device_handles(device);
    const vk::Device nativeDevice(reinterpret_cast<VkDevice>(handles.device));
    const std::array poolSizes{
        vk::DescriptorPoolSize{vk::DescriptorType::eSampler, 64},
        vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, 10000},
        vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 64},
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 64},
        vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 128},
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 30128},
    };
    descriptorPool = nativeDevice.createDescriptorPoolUnique({
        vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 210,
        static_cast<std::uint32_t>(poolSizes.size()), poolSizes.data()});
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = reinterpret_cast<VkInstance>(handles.instance);
    init_info.PhysicalDevice = reinterpret_cast<VkPhysicalDevice>(handles.physical_device);
    init_info.Device = reinterpret_cast<VkDevice>(handles.device);
    init_info.QueueFamily = handles.queue_family;
    init_info.Queue = reinterpret_cast<VkQueue>(handles.queue);
    init_info.DescriptorPool = descriptorPool.get();
    init_info.MinImageCount = 2;
    init_info.ImageCount = numImages;
    init_info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    
    init_info.UseDynamicRendering = true;
    init_info.PipelineInfoMain.PipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR;
    init_info.PipelineInfoMain.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    const auto renderTargetFormat = static_cast<VkFormat>(gpu::interop::native_format(targetFormat));
    init_info.PipelineInfoMain.PipelineRenderingCreateInfo.pColorAttachmentFormats = &renderTargetFormat;
    
    ImGui_ImplVulkan_Init(&init_info);
}

ImGuiManager::~ImGuiManager() {
    // Components may own ImGui Vulkan texture registrations. Destroy them
    // while the backend is still alive; otherwise their destructors would
    // call RemoveTexture after ImGui_ImplVulkan_Shutdown.
    components.clear();
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
    LOG_INFO( "Destroyed ImGuiManager");
}

void ImGuiManager::updateUi() {

    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();
    ImGui::PushItemFlag(ImGuiItemFlags_LiveEditOnInputScalar, false);

    const auto* mainViewport = ImGui::GetMainViewport();
    const float menuBarSize = ImGui::GetFrameHeight();

    ImGui::SetNextWindowPos(ImVec2(mainViewport->Pos.x, mainViewport->Pos.y + menuBarSize));
    ImGui::SetNextWindowSize(ImVec2(mainViewport->Size.x, mainViewport->Size.y - menuBarSize));
    ImGui::SetNextWindowViewport(mainViewport->ID);

    constexpr ImGuiWindowFlags flags = ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoTitleBar |
                                       ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
                                       ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus |
                                       ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_NoDecoration;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{0, 0});
    ImGui::Begin("DockSpaceHost", nullptr, flags);
    ImGui::PopStyleVar(3);
    ImGui::DockSpace(ImGui::GetID("MyDockSpace"), ImVec2(0, 0), ImGuiDockNodeFlags_PassthruCentralNode);

    for (const auto& component : components)
        component->renderUi();

    ImGui::End();
    ImGui::PopItemFlag();

    ImGui::Render();
}

void ImGuiManager::renderDrawData(const gpu::Frame& frame) {
    const vk::CommandBuffer commandBuffer(reinterpret_cast<VkCommandBuffer>(
        gpu::interop::command_buffer(frame)));
    const vk::ImageView targetView(reinterpret_cast<VkImageView>(
        gpu::interop::image_view(device, frame.target())));

    // Vulkan dynamic rendering
    vk::RenderingAttachmentInfo colorAttachment{};
    colorAttachment.setImageView(targetView);
    // gpu keeps presentation images in GENERAL under
    // VK_KHR_unified_image_layouts.  The external ImGui draw is part of that
    // same frame, so it must use the layout the device established.
    colorAttachment.setImageLayout(vk::ImageLayout::eGeneral);
    // The raytracer pass (and the optional compositor copy) has already written
    // the acquired swapchain image. Loading it is required to preserve the
    // renderer output underneath the ImGui overlay.
    colorAttachment.setLoadOp(vk::AttachmentLoadOp::eLoad);
    colorAttachment.setStoreOp(vk::AttachmentStoreOp::eStore);
    colorAttachment.setClearValue(vk::ClearValue{std::array{0.0f, 0.0f, 0.0f, 0.0f}});

    vk::RenderingInfo renderingInfo{};
    renderingInfo.setRenderArea(vk::Rect2D({0, 0}, {frame.width(), frame.height()}));
    renderingInfo.setLayerCount(1);
    renderingInfo.setColorAttachments(colorAttachment);

    commandBuffer.beginRendering(renderingInfo);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);
    commandBuffer.endRendering();
}

void ImGuiManager::processEvent(const SDL_Event& event)
{
        ImGui_ImplSDL3_ProcessEvent(&event);
}

ImGuiComponent* ImGuiManager::getComponent(const std::string& name) const {
    for (const auto& component : components)
        if (component->getName() == name)
            return component.get();
    return nullptr;
}

static ImVec4 mult(const ImVec4& c, float a) {
    return ImVec4(c.x * a, c.y * a, c.z * a, c.w);
}

void ImGuiManager::SetTheme(const Theme theme)
{
    currentTheme = theme;
    ImGuiStyle& style = ImGui::GetStyle();
    ImVec4* colors = style.Colors;

    // Utility: invert colors if light theme
    auto maybeInvert = [&](const ImVec4& c) -> ImVec4 {
        if (theme == Theme::Light)
            return ImVec4(1.0f - c.x, 1.0f - c.y, 1.0f - c.z, c.w);
        return c;
    };

    // ----- COLOR PALETTE -----
    const ImVec4 col_base   = maybeInvert(ImVec4(0.15f, 0.15f, 0.16f, 1.0f)); // Main background
    const ImVec4 col_text   = maybeInvert(ImVec4(0.90f, 0.90f, 0.92f, 1.0f)); // Text color
    const ImVec4 col_accent = maybeInvert(ImVec4(0.75f, 0.75f, 0.75f, 1.0f)); // Neutral accent

    // ----- STYLE SETTINGS -----
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

    // ----- COLORS -----
    const ImVec4 col_elem = mult(col_base, 1.33f);

    // Backgrounds
    colors[ImGuiCol_WindowBg]           = col_base;
    colors[ImGuiCol_ChildBg]            = mult(col_base, 0.87f);
    colors[ImGuiCol_PopupBg]            = mult(col_base, 0.73f);
    colors[ImGuiCol_MenuBarBg]          = mult(col_base, 0.73f);
    colors[ImGuiCol_ScrollbarBg]        = mult(col_base, 0.67f);

    // Borders
    colors[ImGuiCol_Border]             = ImVec4(0.10f, 0.10f, 0.10f, 0.40f);
    colors[ImGuiCol_BorderShadow]       = ImVec4(0, 0, 0, 0);

    // Text
    colors[ImGuiCol_Text]               = col_text;
    colors[ImGuiCol_TextDisabled]       = mult(col_text, 0.55f);

    // Title Bars
    colors[ImGuiCol_TitleBg]            = mult(col_base, 0.87f);
    colors[ImGuiCol_TitleBgActive]      = mult(col_base, 1.0f);
    colors[ImGuiCol_TitleBgCollapsed]   = mult(col_base, 0.67f);

    // Frames, headers, buttons
    colors[ImGuiCol_FrameBg]            = col_elem;
    colors[ImGuiCol_FrameBgHovered]     = mult(col_elem, 1.5f);
    colors[ImGuiCol_FrameBgActive]      = mult(col_elem, 1.75f);

    colors[ImGuiCol_Header]             = col_elem;
    colors[ImGuiCol_HeaderHovered]      = mult(col_elem, 1.5f);
    colors[ImGuiCol_HeaderActive]       = mult(col_elem, 1.25f);

    colors[ImGuiCol_Button]             = mult(col_base, 1.2f);
    colors[ImGuiCol_ButtonHovered]      = mult(col_base, 1.6f);
    colors[ImGuiCol_ButtonActive]       = mult(col_base, 1.85f);

    // Tabs
    colors[ImGuiCol_Tab]                = mult(col_base, 1.07f);
    colors[ImGuiCol_TabHovered]         = mult(col_base, 1.47f);
    colors[ImGuiCol_TabActive]          = mult(col_base, 1.27f);
    colors[ImGuiCol_TabUnfocused]       = mult(col_base, 0.80f);
    colors[ImGuiCol_TabUnfocusedActive] = mult(col_base, 1.07f);

    // Controls
    colors[ImGuiCol_CheckMark]          = col_accent;
    colors[ImGuiCol_SliderGrab]         = mult(col_accent, 0.87f);
    colors[ImGuiCol_SliderGrabActive]   = mult(col_accent, 1.07f);

    // Resize grips
    colors[ImGuiCol_ResizeGrip]         = mult(col_accent, 0.87f);
    colors[ImGuiCol_ResizeGripHovered]  = mult(col_accent, 1.0f);
    colors[ImGuiCol_ResizeGripActive]   = mult(col_accent, 1.13f);

    // Scrollbar
    colors[ImGuiCol_ScrollbarGrab]      = mult(col_elem, 1.25f);
    colors[ImGuiCol_ScrollbarGrabHovered]=mult(col_elem, 1.5f);
    colors[ImGuiCol_ScrollbarGrabActive]= mult(col_elem, 1.75f);
}
