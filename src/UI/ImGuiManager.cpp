#include "ImGuiManager.h"

#include "../Vulkan/Context.h"
#include "../Globals.h"
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <memory>
#include <vulkan/vulkan.hpp>
#include <vector>
#include <functional>

#include "imgui_internal.h"
#include "glm/gtc/type_ptr.inl"

ImGuiManager::ImGuiManager(Context& context, const std::vector<vk::Image>& swapchainImages, uint32_t width, uint32_t height)
{
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    static constexpr unsigned char font[] = {
        #embed "../../assets/Inter-Regular.ttf"
    };
    io.Fonts->AddFontFromMemoryTTF(const_cast<unsigned char*>(font), sizeof(font), 24.0f);

    SetBlenderTheme();

    // Setup Platform backend
    ImGui_ImplGlfw_InitForVulkan(context.window, true);

    CreateDescriptorPool(context);
    CreateRenderPass(context);
    CreateFrameBuffers(context, swapchainImages, width, height);

    // Setup ImGui Vulkan backend
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = context.instance.get();
    init_info.PhysicalDevice = context.physicalDevice;
    init_info.Device = context.device.get();
    init_info.QueueFamily = context.queueFamilyIndex;
    init_info.Queue = context.queue;
    init_info.PipelineCache = VK_NULL_HANDLE;
    init_info.DescriptorPool = descriptorPool.get();
    init_info.RenderPass = renderPass.get();
    init_info.Subpass = 0;
    init_info.MinImageCount = 2;
    init_info.ImageCount = 3; //TODO match pipeline
    init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    init_info.Allocator = nullptr;
    init_info.CheckVkResultFn = nullptr;

    ImGui_ImplVulkan_Init(&init_info);
}

void ImGuiManager::SetBlenderTheme() {
    ImGuiStyle& style = ImGui::GetStyle();
    ImVec4* colors = style.Colors;

    // Top Bar
    colors[ImGuiCol_MenuBarBg] = ImVec4(0.11f, 0.11f, 0.11f, 1.00f);

    // Window Backgrounds
    colors[ImGuiCol_WindowBg]         = ImVec4(0.15f, 0.15f, 0.16f, 1.00f); // #262627
    colors[ImGuiCol_ChildBg]          = ImVec4(0.13f, 0.13f, 0.14f, 1.00f);
    colors[ImGuiCol_PopupBg]          = ImVec4(0.11f, 0.11f, 0.11f, 1.00f);

    // Headers (like tree nodes, menu bar)
    colors[ImGuiCol_Header]           = ImVec4(0.20f, 0.20f, 0.21f, 1.00f);
    colors[ImGuiCol_HeaderHovered]    = ImVec4(0.30f, 0.30f, 0.31f, 1.00f);
    colors[ImGuiCol_HeaderActive]     = ImVec4(0.25f, 0.25f, 0.26f, 1.00f);

    // Buttons
    colors[ImGuiCol_Button]           = ImVec4(0.18f, 0.18f, 0.19f, 1.00f);
    colors[ImGuiCol_ButtonHovered]    = ImVec4(0.24f, 0.24f, 0.26f, 1.00f);
    colors[ImGuiCol_ButtonActive]     = ImVec4(0.28f, 0.28f, 0.30f, 1.00f);

    // Frame BG (input boxes, sliders, etc.)
    colors[ImGuiCol_FrameBg]          = ImVec4(0.20f, 0.20f, 0.21f, 1.00f);
    colors[ImGuiCol_FrameBgHovered]   = ImVec4(0.30f, 0.30f, 0.31f, 1.00f);
    colors[ImGuiCol_FrameBgActive]    = ImVec4(0.35f, 0.35f, 0.36f, 1.00f);

    // Tabs
    colors[ImGuiCol_Tab]              = ImVec4(0.16f, 0.16f, 0.18f, 1.00f);
    colors[ImGuiCol_TabHovered]       = ImVec4(0.22f, 0.22f, 0.24f, 1.00f);
    colors[ImGuiCol_TabActive]        = ImVec4(0.19f, 0.19f, 0.21f, 1.00f);
    colors[ImGuiCol_TabUnfocused]     = ImVec4(0.12f, 0.12f, 0.14f, 1.00f);
    colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.16f, 0.16f, 0.18f, 1.00f);

    // Title
    colors[ImGuiCol_TitleBg]          = ImVec4(0.13f, 0.13f, 0.14f, 1.00f);
    colors[ImGuiCol_TitleBgActive]    = ImVec4(0.15f, 0.15f, 0.17f, 1.00f);
    colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.10f, 0.10f, 0.12f, 1.00f);

    // Borders
    colors[ImGuiCol_Border]           = ImVec4(0.10f, 0.10f, 0.10f, 0.40f);
    colors[ImGuiCol_BorderShadow]     = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);

    // Text
    colors[ImGuiCol_Text]             = ImVec4(0.90f, 0.90f, 0.92f, 1.00f);
    colors[ImGuiCol_TextDisabled]     = ImVec4(0.50f, 0.50f, 0.52f, 1.00f);

    // Highlights
    colors[ImGuiCol_CheckMark]        = ImVec4(0.75f, 0.75f, 0.75f, 1.00f);
    colors[ImGuiCol_SliderGrab]       = ImVec4(0.65f, 0.65f, 0.65f, 1.00f);
    colors[ImGuiCol_SliderGrabActive] = ImVec4(0.80f, 0.80f, 0.80f, 1.00f);
    colors[ImGuiCol_ResizeGrip]       = ImVec4(0.65f, 0.65f, 0.65f, 0.60f);
    colors[ImGuiCol_ResizeGripHovered]= ImVec4(0.75f, 0.75f, 0.75f, 0.80f);
    colors[ImGuiCol_ResizeGripActive] = ImVec4(0.85f, 0.85f, 0.85f, 1.00f);

    // Scrollbar
    colors[ImGuiCol_ScrollbarBg]          = ImVec4(0.10f, 0.10f, 0.12f, 1.00f);
    colors[ImGuiCol_ScrollbarGrab]        = ImVec4(0.20f, 0.20f, 0.25f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.25f, 0.25f, 0.30f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabActive]  = ImVec4(0.30f, 0.30f, 0.35f, 1.00f);

    // Style tweaks — small border radius on tabs/windows
    style.WindowRounding = 4.0f;       // subtle rounding on window corners
    style.FrameRounding = 4.0f;        // rounding on buttons/input frames
    style.GrabRounding = 4.0f;
    style.TabRounding = 4.0f;           // small rounded corners on tabs
    style.PopupRounding = 4.0f;
    style.ScrollbarRounding = 4.0f;
    style.WindowPadding = ImVec2(8, 8);
    style.FramePadding = ImVec2(5, 3);
    style.ItemSpacing = ImVec2(6, 4);
    style.PopupBorderSize = 1.f;
}

void ImGuiManager::CreateDescriptorPool(Context& context) {
    std::vector<vk::DescriptorPoolSize> poolSizes = {
        { vk::DescriptorType::eSampler, 1000 },
        { vk::DescriptorType::eCombinedImageSampler, 1000 },
        { vk::DescriptorType::eSampledImage, 1000 },
        { vk::DescriptorType::eStorageImage, 1000 },
        { vk::DescriptorType::eUniformTexelBuffer, 1000 },
        { vk::DescriptorType::eStorageTexelBuffer, 1000 },
        { vk::DescriptorType::eUniformBuffer, 1000 },
        { vk::DescriptorType::eStorageBuffer, 1000 },
        { vk::DescriptorType::eUniformBufferDynamic, 1000 },
        { vk::DescriptorType::eStorageBufferDynamic, 1000 },
        { vk::DescriptorType::eInputAttachment, 1000 }
    };

    vk::DescriptorPoolCreateInfo poolInfo{};
    poolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
    poolInfo.maxSets = 1000; // Max number of descriptor sets allocated from this pool
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();

    descriptorPool = context.device.get().createDescriptorPoolUnique(poolInfo);
}

void ImGuiManager::CreateRenderPass(Context& context) {
    vk::AttachmentDescription colorAttachment{};
    colorAttachment.format = vk::Format::eB8G8R8A8Unorm; // Must match swapchain format
    colorAttachment.samples = vk::SampleCountFlagBits::e1;
    colorAttachment.loadOp = vk::AttachmentLoadOp::eLoad;
    colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;
    colorAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
    colorAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
    colorAttachment.initialLayout = vk::ImageLayout::eColorAttachmentOptimal;
    colorAttachment.finalLayout = vk::ImageLayout::eColorAttachmentOptimal;

    vk::AttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = vk::ImageLayout::eColorAttachmentOptimal;

    vk::SubpassDescription subpass{};
    subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;

    vk::SubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    dependency.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    dependency.srcAccessMask = {};
    dependency.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;

    vk::RenderPassCreateInfo renderPassInfo{};
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    renderPass = context.device->createRenderPassUnique(renderPassInfo);
}

void ImGuiManager::CreateFrameBuffers(Context& context, const std::vector<vk::Image>& images, uint32_t width, uint32_t height) {
    for (const auto& image : images) {
        vk::ImageViewCreateInfo viewInfo{};
        viewInfo.image = image;
        viewInfo.viewType = vk::ImageViewType::e2D;
        viewInfo.format = vk::Format::eB8G8R8A8Unorm; //TODO match Swapchain
        viewInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.layerCount = 1;

        imageViews.push_back(context.device->createImageViewUnique(viewInfo));

        vk::FramebufferCreateInfo framebufferInfo{};
        framebufferInfo.renderPass = renderPass.get();
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.pAttachments = &imageViews.back().get();
        framebufferInfo.width = width; //TODO
        framebufferInfo.height = height;
        framebufferInfo.layers = 1;

        frameBuffers.push_back(context.device->createFramebufferUnique(framebufferInfo));
    }
}

void ImGuiManager::setupDockSpace() {
    auto* mainViewport = ImGui::GetMainViewport();
    float menuBarSize = ImGui::GetFrameHeight(); // Height of menu bar
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

    // Create the dockspace
    ImGuiID dockspaceID = ImGui::GetID("MyDockSpace");
    ImGui::DockSpace(dockspaceID, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_PassthruCentralNode);
}

void ImGuiManager::Draw(const vk::CommandBuffer commandBuffer, uint32_t imageIndex, uint32_t width, uint32_t height)
{
    constexpr VkClearValue clear_value = { { 0.0f, 0.0f, 0.0f, 1.0f } };
    VkRenderPassBeginInfo rpInfo{};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpInfo.renderPass = renderPass.get();
    rpInfo.framebuffer = frameBuffers[imageIndex].get();
    rpInfo.renderArea.extent.width = width;
    rpInfo.renderArea.extent.height = height;
    rpInfo.clearValueCount = 1;
    rpInfo.pClearValues = &clear_value;

    vkCmdBeginRenderPass(commandBuffer, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);
    vkCmdEndRenderPass(commandBuffer);
}

void ImGuiManager::renderUi() const
{
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    //Setup dockspace
    setupDockSpace();

    for (auto& component : components)
        component->renderUi();

    // End dockspace
    ImGui::End();

    ImGui::Render();

}

void ImGuiManager::addComponent(std::unique_ptr<ImGuiComponent> component) {
    components.push_back(std::move(component));
}

ImGuiComponent* ImGuiManager::getComponent(const std::string& name) const
{
    for (auto& component : components)
        if (component->getType() == name)
            return component.get();
    return nullptr;
}

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

void ImGuiManager::dragFloatRow(const char* label, float value, float speed, float min, float max, const std::function<void(float)>& setter) {
    tableRowLabel(label);
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (ImGui::DragFloat((std::string("##") + label).c_str(), &value, speed, min, max, "%.3f", ImGuiSliderFlags_AlwaysClamp))
        setter(value);
}

void ImGuiManager::dragFloat3Row(const char* label, glm::vec3 value, float speed, const std::function<void(glm::vec3)>& setter) {
    tableRowLabel(label);
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (ImGui::DragFloat3((std::string("##") + label).c_str(), glm::value_ptr(value), speed))
        setter(value);
}

void ImGuiManager::colorEdit3Row(const char* label, glm::vec3 value, const std::function<void(glm::vec3)>& setter) {
    tableRowLabel(label);
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (glm::vec3 temp = value; ImGui::ColorEdit3((std::string("##") + label).c_str(), glm::value_ptr(temp)))
        setter(temp);
}

ImGuiManager::~ImGuiManager() {
    components.clear();
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}
