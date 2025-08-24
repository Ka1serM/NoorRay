#include "ImGuiManager.h"
#include "UI/ImGuiComponent.h"
#include <imgui.h>
#include <iostream>
#include "imgui_internal.h"
#include "backends/imgui_impl_sdl3.h"
#include "backends/imgui_impl_vulkan.h"
#include "glm/gtc/type_ptr.inl"
#include "Vulkan/Context.h"
#include <array>

ImGuiManager::ImGuiManager(Context& context, const std::vector<vk::Image>& swapchainImages, uint32_t width, uint32_t height)
    : context(context)
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;


#ifdef NDEBUG // Release mode: embed ini into binary
    io.IniFilename = nullptr;  // don't use external file
    static constexpr char ini[] = {
        #embed "../../assets/imgui.ini"
    };
    ImGui::LoadIniSettingsFromMemory(ini, sizeof(ini));
#else
    // Debug mode: use external file for easy tweaking
    io.IniFilename = "imgui.ini";  // ImGui will load and save this file
#endif

    static constexpr unsigned char font[] = {
        #embed "../../assets/Inter-Regular.ttf"
    };
    ImFontConfig font_config;
    font_config.FontDataOwnedByAtlas = false;
    io.Fonts->AddFontFromMemoryTTF(const_cast<unsigned char*>(font), sizeof(font), 18.0f, &font_config);

    SetBlenderTheme();

    ImGui_ImplSDL3_InitForVulkan(context.getWindow());

    CreateRenderPass();
    CreateFrameBuffers(swapchainImages, width, height);

    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = context.getInstance();
    init_info.PhysicalDevice = context.getPhysicalDevice();
    init_info.Device = context.getDevice();
    init_info.QueueFamily = context.getQueueFamilyIndices().front();
    init_info.Queue = context.getQueue();
    init_info.DescriptorPool = context.getDescriptorPool();
    init_info.RenderPass = renderPass.get();
    init_info.Subpass = 0;
    init_info.MinImageCount = 2;
    init_info.ImageCount = static_cast<uint32_t>(swapchainImages.size());
    init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

    ImGui_ImplVulkan_Init(&init_info);
}

ImGuiManager::~ImGuiManager() {
    // Resources are cleaned up by vk::Unique... handles and ImGui shutdown functions.
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
    std::cout << "Destroyed ImGuiManager" << std::endl;
}

// --- MODIFIED: This is the more efficient way to handle resizing ---
// We only recreate the resources that are dependent on the swapchain size.
void ImGuiManager::recreateForSwapChain(const std::vector<vk::Image>& swapchainImages, uint32_t width, uint32_t height)
{
    // Clean up only the swapchain-dependent resources.
    // The core ImGui Vulkan backend remains initialized.
    cleanupSwapChainResources();

    // Recreate the render pass and framebuffers for the new swapchain.
    CreateRenderPass();
    CreateFrameBuffers(swapchainImages, width, height);
}

void ImGuiManager::cleanupSwapChainResources() {
    frameBuffers.clear();
    imageViews.clear();
    renderPass.reset(); // vk::UniqueRenderPass automatically handles destruction
}

void ImGuiManager::CreateRenderPass() {
    vk::AttachmentDescription colorAttachment(
        {}, // flags
        context.chooseSwapSurfaceFormat().format,
        vk::SampleCountFlagBits::e1,
        vk::AttachmentLoadOp::eClear,
        vk::AttachmentStoreOp::eStore,
        vk::AttachmentLoadOp::eDontCare,
        vk::AttachmentStoreOp::eDontCare,
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::ePresentSrcKHR
    );

    vk::AttachmentReference colorAttachmentRef(0, vk::ImageLayout::eColorAttachmentOptimal);

    vk::SubpassDescription subpass(
        {}, vk::PipelineBindPoint::eGraphics,
        0, nullptr, 1, &colorAttachmentRef
    );

    vk::SubpassDependency dependency(
        VK_SUBPASS_EXTERNAL, 0,
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
        {}, vk::AccessFlagBits::eColorAttachmentWrite
    );

    vk::RenderPassCreateInfo renderPassInfo({}, 1, &colorAttachment, 1, &subpass, 1, &dependency);
    renderPass = context.getDevice().createRenderPassUnique(renderPassInfo);
}

void ImGuiManager::CreateFrameBuffers(const std::vector<vk::Image>& images, uint32_t width, uint32_t height) {
    imageViews.reserve(images.size());
    frameBuffers.reserve(images.size());

    for (const auto& image : images) {
        vk::ImageViewCreateInfo viewInfo(
            {}, image, vk::ImageViewType::e2D, context.chooseSwapSurfaceFormat().format,
            {}, { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 }
        );
        imageViews.push_back(context.getDevice().createImageViewUnique(viewInfo));

        std::array<vk::ImageView, 1> attachments = { imageViews.back().get() };
        vk::FramebufferCreateInfo framebufferInfo(
            {}, renderPass.get(), attachments, width, height, 1
        );
        frameBuffers.push_back(context.getDevice().createFramebufferUnique(framebufferInfo));
    }
}

void ImGuiManager::SetBlenderTheme() {
    ImGuiStyle& style = ImGui::GetStyle();

    // Rounded corners everywhere
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

    //  Slightly bigger feel
    style.WindowPadding     = ImVec2(12, 12);
    style.FramePadding      = ImVec2(6, 4);
    style.ItemSpacing       = ImVec2(6, 4);
    style.ItemInnerSpacing  = ImVec2(6, 4);
    
    ImVec4* colors = style.Colors;

    // Backgrounds
    colors[ImGuiCol_WindowBg]           = ImVec4(0.15f, 0.15f, 0.16f, 1.00f);
    colors[ImGuiCol_ChildBg]            = ImVec4(0.13f, 0.13f, 0.14f, 1.00f);
    colors[ImGuiCol_PopupBg]            = ImVec4(0.11f, 0.11f, 0.11f, 1.00f);
    colors[ImGuiCol_MenuBarBg]          = ImVec4(0.11f, 0.11f, 0.11f, 1.00f);

    // Headers (selection highlight, tree nodes, list items)
    colors[ImGuiCol_Header]             = ImVec4(0.20f, 0.20f, 0.21f, 1.00f);
    colors[ImGuiCol_HeaderHovered]      = ImVec4(0.30f, 0.30f, 0.31f, 1.00f);
    colors[ImGuiCol_HeaderActive]       = ImVec4(0.25f, 0.25f, 0.26f, 1.00f);

    // Buttons
    colors[ImGuiCol_Button]             = ImVec4(0.18f, 0.18f, 0.19f, 1.00f);
    colors[ImGuiCol_ButtonHovered]      = ImVec4(0.24f, 0.24f, 0.26f, 1.00f);
    colors[ImGuiCol_ButtonActive]       = ImVec4(0.28f, 0.28f, 0.30f, 1.00f);

    // Frames (inputs, selection highlights use rounding here)
    colors[ImGuiCol_FrameBg]            = ImVec4(0.20f, 0.20f, 0.21f, 1.00f);
    colors[ImGuiCol_FrameBgHovered]     = ImVec4(0.30f, 0.30f, 0.31f, 1.00f);
    colors[ImGuiCol_FrameBgActive]      = ImVec4(0.35f, 0.35f, 0.36f, 1.00f);

    // Tabs
    colors[ImGuiCol_Tab]                = ImVec4(0.16f, 0.16f, 0.18f, 1.00f);
    colors[ImGuiCol_TabHovered]         = ImVec4(0.22f, 0.22f, 0.24f, 1.00f);
    colors[ImGuiCol_TabActive]          = ImVec4(0.19f, 0.19f, 0.21f, 1.00f);
    colors[ImGuiCol_TabUnfocused]       = ImVec4(0.12f, 0.12f, 0.14f, 1.00f);
    colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.16f, 0.16f, 0.18f, 1.00f);

    // Titles
    colors[ImGuiCol_TitleBg]            = ImVec4(0.13f, 0.13f, 0.14f, 1.00f);
    colors[ImGuiCol_TitleBgActive]      = ImVec4(0.15f, 0.15f, 0.17f, 1.00f);
    colors[ImGuiCol_TitleBgCollapsed]   = ImVec4(0.10f, 0.10f, 0.12f, 1.00f);

    // Borders
    colors[ImGuiCol_Border]             = ImVec4(0.10f, 0.10f, 0.10f, 0.40f);
    colors[ImGuiCol_BorderShadow]       = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);

    // Text
    colors[ImGuiCol_Text]               = ImVec4(0.90f, 0.90f, 0.92f, 1.00f);
    colors[ImGuiCol_TextDisabled]       = ImVec4(0.50f, 0.50f, 0.52f, 1.00f);

    // Checks, sliders, grabs
    colors[ImGuiCol_CheckMark]          = ImVec4(0.75f, 0.75f, 0.75f, 1.00f);
    colors[ImGuiCol_SliderGrab]         = ImVec4(0.65f, 0.65f, 0.65f, 1.00f);
    colors[ImGuiCol_SliderGrabActive]   = ImVec4(0.80f, 0.80f, 0.80f, 1.00f);

    // Resize grips
    colors[ImGuiCol_ResizeGrip]         = ImVec4(0.65f, 0.65f, 0.65f, 0.60f);
    colors[ImGuiCol_ResizeGripHovered]  = ImVec4(0.75f, 0.75f, 0.75f, 0.80f);
    colors[ImGuiCol_ResizeGripActive]   = ImVec4(0.85f, 0.85f, 0.85f, 1.00f);

    // Scrollbars
    colors[ImGuiCol_ScrollbarBg]        = ImVec4(0.10f, 0.10f, 0.12f, 1.00f);
    colors[ImGuiCol_ScrollbarGrab]      = ImVec4(0.20f, 0.20f, 0.25f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabHovered]=ImVec4(0.25f, 0.25f, 0.30f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabActive]= ImVec4(0.30f, 0.30f, 0.35f, 1.00f);
}


void ImGuiManager::setupDockSpace() {
    auto* mainViewport = ImGui::GetMainViewport();
    float menuBarSize = ImGui::GetFrameHeight();
    ImGui::SetNextWindowPos(ImVec2(mainViewport->Pos.x, mainViewport->Pos.y + menuBarSize));
    ImGui::SetNextWindowSize(ImVec2(mainViewport->Size.x, mainViewport->Size.y - menuBarSize));
    ImGui::SetNextWindowViewport(mainViewport->ID);

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoTitleBar |
                             ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
                             ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus |
                             ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_NoDecoration;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{0, 0});

    ImGui::Begin("DockSpaceHost", nullptr, flags);
    ImGui::PopStyleVar(3);
    
    ImGui::DockSpace(ImGui::GetID("MyDockSpace"), ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_PassthruCentralNode);
}

void ImGuiManager::Draw(const vk::CommandBuffer commandBuffer, const uint32_t imageIndex, const uint32_t width, const uint32_t height)
{
    vk::ClearValue clearValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f});
    vk::RenderPassBeginInfo renderPassInfo(
        renderPass.get(),
        frameBuffers[imageIndex].get(),
        vk::Rect2D({0, 0}, {width, height}),
        clearValue
    );

    commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);
    commandBuffer.endRenderPass();
}

void ImGuiManager::renderUi()
{
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();

    setupDockSpace();

    for (const auto& component : components) {
        component->renderUi();
    }

    ImGui::End(); // End the DockSpace window

    ImGui::Render();
}

ImGuiComponent* ImGuiManager::getComponent(const std::string& name) const
{
    for (const auto& component : components)
        if (component->getName() == name) 
            return component.get();
    return nullptr;
}

// --- UI Helper Functions ---
void ImGuiManager::tableRowLabel(const char* label) {
    if (ImGui::GetCurrentTable()) {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted(label);
        ImGui::TableSetColumnIndex(1);
    } else {
        ImGui::TextUnformatted(label);
        ImGui::SameLine();
    }
}

void ImGuiManager::checkboxRow(const char* label, bool value, const std::function<void(bool)>& setter) {
    tableRowLabel(label);
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (ImGui::Checkbox((std::string("##") + label).c_str(), &value)) {
        setter(value);
    }
}

void ImGuiManager::dragFloatRow(const char* label, float value, float speed, float min, float max, const std::function<void(float)>& setter) {
    tableRowLabel(label);
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (ImGui::DragFloat((std::string("##") + label).c_str(), &value, speed, min, max, "%.3f", ImGuiSliderFlags_AlwaysClamp)) {
        setter(value);
    }
}

void ImGuiManager::dragFloat3Row(const char* label, glm::vec3 value, float speed, const std::function<void(glm::vec3)>& setter) {
    tableRowLabel(label);
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (ImGui::DragFloat3((std::string("##") + label).c_str(), glm::value_ptr(value), speed)) {
        setter(value);
    }
}

void ImGuiManager::colorEdit3Row(const char* label, glm::vec3 value, const std::function<void(glm::vec3)>& setter) {
    tableRowLabel(label);
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (glm::vec3 temp = value; ImGui::ColorEdit3((std::string("##") + label).c_str(), glm::value_ptr(temp))) {
        setter(temp);
    }
}

auto ImGuiManager::colorEdit4Row(const char* label, glm::vec4 value, const std::function<void(glm::vec4)>& setter) -> void
{
    tableRowLabel(label);
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (glm::vec4 temp = value; ImGui::ColorEdit4((std::string("##") + label).c_str(), glm::value_ptr(temp))) {
        setter(temp);
    }
}