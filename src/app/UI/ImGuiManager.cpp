#include "ImGuiManager.h"
#include "ImGuiComponent.h"
#include "ImGuiFont.h"
#include "ImGuiLayout.h"
#include <imgui.h>
#include "imgui_internal.h"
#include "backends/imgui_impl_sdl3.h"
#include "backends/imgui_impl_vulkan.h"
#include "glm/gtc/type_ptr.inl"
#include "Vulkan/Context.h"
#include "UI/Window.h"
#include <array>
#include "Log.h"

ImGuiManager::ImGuiManager(Window& window, Context& context, const uint32_t numImages,
    const vk::SurfaceFormatKHR targetFormat)
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

#ifdef NDEBUG // Release mode: embed ini into binary
    io.IniFilename = nullptr;  // don't use external file
    ImGui::LoadIniSettingsFromMemory(
        reinterpret_cast<const char*>(noorRayImGuiLayout), noorRayImGuiLayoutLength);
#else
    // Debug mode: use external file for easy tweaking
    io.IniFilename = "imgui.ini";  // ImGui will load and save this file
#endif

    ImFontConfig font_config;
    font_config.FontDataOwnedByAtlas = false;

    const float font_size = 18.0f * window.getDpiScale();
    io.Fonts->AddFontFromMemoryTTF(
        const_cast<unsigned char*>(noorRayImGuiFont), noorRayImGuiFontLength,
        font_size, &font_config);

    ImGuiStyle& style = ImGui::GetStyle();
    style.ScaleAllSizes(window.getDpiScale());

    if (const SDL_SystemTheme theme = SDL_GetSystemTheme(); theme == SDL_SYSTEM_THEME_LIGHT)
        SetTheme(Theme::Light);
    else
        SetTheme(Theme::Dark);

    ImGui_ImplSDL3_InitForVulkan(window.nativeHandle());

    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = context.getInstance();
    init_info.PhysicalDevice = context.getPhysicalDevice();
    init_info.Device = context.getDevice();
    init_info.QueueFamily = context.getGraphicsFamilyIndex();
    init_info.Queue = context.getGraphicsQueue();
    init_info.DescriptorPool = context.getDescriptorPool();
    init_info.MinImageCount = 2;
    init_info.ImageCount = numImages;
    init_info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    
    init_info.UseDynamicRendering = true;
    init_info.PipelineInfoMain.PipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR;
    init_info.PipelineInfoMain.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    const auto renderTargetFormat = static_cast<VkFormat>(targetFormat.format);
    init_info.PipelineInfoMain.PipelineRenderingCreateInfo.pColorAttachmentFormats = &renderTargetFormat;
    
    ImGui_ImplVulkan_Init(&init_info);
}

ImGuiManager::~ImGuiManager() {
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

void ImGuiManager::renderDrawData(vk::CommandBuffer commandBuffer, const vk::ImageView targetView, const vk::Extent2D targetExtent) {

    // Vulkan dynamic rendering
    vk::RenderingAttachmentInfo colorAttachment{};
    colorAttachment.setImageView(targetView);
    colorAttachment.setImageLayout(vk::ImageLayout::eColorAttachmentOptimal);
    colorAttachment.setLoadOp(vk::AttachmentLoadOp::eClear);
    colorAttachment.setStoreOp(vk::AttachmentStoreOp::eStore);
    colorAttachment.setClearValue(vk::ClearValue{std::array{0.0f, 0.0f, 0.0f, 0.0f}});

    vk::RenderingInfo renderingInfo{};
    renderingInfo.setRenderArea(vk::Rect2D({0, 0}, targetExtent));
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
