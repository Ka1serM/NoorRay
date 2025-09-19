#include "ImGuiManager.h"
#include "ImGuiComponent.h"
#include <imgui.h>
#include "imgui_internal.h"
#include "backends/imgui_impl_sdl3.h"
#include "backends/imgui_impl_vulkan.h"
#include "glm/gtc/type_ptr.inl"
#include "Vulkan/Context.h"
#include <array>
#include "Log.h"

ImGuiManager::ImGuiManager(Context& context, uint32_t numSwapchainImages, const vk::SurfaceFormatKHR swapchainFormat)
    : context(context)
{
    m_swapchainFormat = static_cast<VkFormat>(swapchainFormat.format);
    
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

#ifdef NDEBUG // Release mode: embed ini into binary
    io.IniFilename = nullptr;  // don't use external file
    static constexpr char ini[] = {
        #embed "../../../assets/imgui.ini"
    };
    ImGui::LoadIniSettingsFromMemory(ini, sizeof(ini));
#else
    // Debug mode: use external file for easy tweaking
    io.IniFilename = "imgui.ini";  // ImGui will load and save this file
#endif

    static constexpr unsigned char font[] = {
        #embed "../../../assets/fonts/Inter-Regular.ttf"
    };
    ImFontConfig font_config;
    font_config.FontDataOwnedByAtlas = false;

    const float font_size = 18.0f * context.getDPIScale();
    io.Fonts->AddFontFromMemoryTTF(const_cast<unsigned char*>(font), sizeof(font), font_size, &font_config);

    ImGuiStyle& style = ImGui::GetStyle();
    style.ScaleAllSizes(context.getDPIScale());
    
    SetBlenderTheme();

    ImGui_ImplSDL3_InitForVulkan(context.getWindow());

    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = context.getInstance();
    init_info.PhysicalDevice = context.getPhysicalDevice();
    init_info.Device = context.getDevice();
    init_info.QueueFamily = context.getGraphicsFamilyIndex();
    init_info.Queue = context.getGraphicsQueue();
    init_info.DescriptorPool = context.getDescriptorPool();
    init_info.MinImageCount = 2;
    init_info.ImageCount = numSwapchainImages;
    init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    
    init_info.UseDynamicRendering = true;
    init_info.PipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR;
    init_info.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    init_info.PipelineRenderingCreateInfo.pColorAttachmentFormats = &m_swapchainFormat;
    
    ImGui_ImplVulkan_Init(&init_info);
}

ImGuiManager::~ImGuiManager() {
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
    LOG_INFO( "Destroyed ImGuiManager");
}

void ImGuiManager::render(const vk::CommandBuffer commandBuffer,  const vk::ImageView target_image_view,  const vk::Extent2D currentExtent)
{
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();
    
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
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0)); // fully transparent
    ImGui::Begin("DockSpaceHost", nullptr, flags);
    ImGui::PopStyleVar(3);
    ImGui::PopStyleColor();
    ImGui::DockSpace(ImGui::GetID("MyDockSpace"), ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_PassthruCentralNode);
    
    for (const auto& component : components)
        component->renderUi();
    
    ImGui::End(); // End the DockSpace

    ImGui::Render();
    
    // DYNAMIC RENDERING
    vk::RenderingAttachmentInfo colorAttachmentInfo{};
    colorAttachmentInfo.setImageView(target_image_view);
    colorAttachmentInfo.setImageLayout(vk::ImageLayout::eColorAttachmentOptimal); // ImGui backend expects this layout
    colorAttachmentInfo.setLoadOp(vk::AttachmentLoadOp::eNone);
    colorAttachmentInfo.setStoreOp(vk::AttachmentStoreOp::eStore);
    colorAttachmentInfo.setClearValue(vk::ClearValue{std::array{0.0f, 0.0f, 0.0f, 1.0f}});
    
    vk::RenderingInfo renderingInfo{};
    renderingInfo.setRenderArea(vk::Rect2D({0, 0}, currentExtent));
    renderingInfo.setLayerCount(1);
    renderingInfo.setColorAttachments(colorAttachmentInfo);

    commandBuffer.beginRendering(renderingInfo);
    
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);
    
    commandBuffer.endRendering();
}

void ImGuiManager::processEvent(const SDL_Event& event)
{
        ImGui_ImplSDL3_ProcessEvent(&event);
}

void ImGuiManager::SetBlenderTheme() {
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

ImGuiComponent* ImGuiManager::getComponent(const std::string& name) const {
    for (const auto& component : components)
        if (component->getName() == name)
            return component.get();
    return nullptr;
}

void ImGuiManager::tableRowLabel(const char* label) {
    if (ImGui::GetCurrentTable()) {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted(label);
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-FLT_MIN);
    } else {
        ImGui::TextUnformatted(label);
        ImGui::SameLine();
    }
}

void ImGuiManager::checkboxRow(const char* label, bool value, const std::function<void(bool)>& setter) {
    tableRowLabel(label);
    if (ImGui::Checkbox((std::string("##") + label).c_str(), &value))
        setter(value);
}

void ImGuiManager::dragFloatRow(const char* label, float value, const float speed, const float min, const float max, const std::function<void(float)>& setter) {
    tableRowLabel(label);
    if (ImGui::DragFloat((std::string("##") + label).c_str(), &value, speed, min, max, "%.3f", ImGuiSliderFlags_AlwaysClamp))
        setter(value);
}

void ImGuiManager::dragFloat3Row(const char* label, glm::vec3 value, const float speed, const std::function<void(glm::vec3)>& setter) {
    tableRowLabel(label);
    if (ImGui::DragFloat3((std::string("##") + label).c_str(), glm::value_ptr(value), speed))
        setter(value);
}

void ImGuiManager::colorEdit3Row(const char* label, const glm::vec3 value, const std::function<void(glm::vec3)>& setter) {
    tableRowLabel(label);
    if (glm::vec3 temp = value; ImGui::ColorEdit3((std::string("##") + label).c_str(), glm::value_ptr(temp)))
        setter(temp);
}

void ImGuiManager::colorEdit4Row(const char* label, const glm::vec4 value, const std::function<void(glm::vec4)>& setter) {
    tableRowLabel(label);
    if (glm::vec4 temp = value; ImGui::ColorEdit4((std::string("##") + label).c_str(), glm::value_ptr(temp)))
        setter(temp);
}