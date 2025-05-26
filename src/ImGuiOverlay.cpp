#include "ImGuiOverlay.h"

#include "Context.h"
#include "Globals.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <memory>
#include <vulkan/vulkan.hpp>
#include <vector>

#include "Image.h"
#include "glm/gtc/type_ptr.hpp"

ImGuiOverlay::ImGuiOverlay(std::shared_ptr<Context> context, const std::vector<vk::Image>& swapchainImages, Image& outputImage) : context(context)
{
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    io.Fonts->AddFontFromFileTTF("../assets/Inter-Regular.ttf", 24);

    SetBlenderTheme();

    // Setup Platform backend
    ImGui_ImplGlfw_InitForVulkan(context->window, true);

    CreateDescriptorPool();
    CreateRenderPass();
    CreateFrameBuffers(swapchainImages);

    // Setup ImGui Vulkan backend
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = context->instance.get();
    init_info.PhysicalDevice = context->physicalDevice;
    init_info.Device = context->device.get();
    init_info.QueueFamily = context->queueFamilyIndex;
    init_info.Queue = context->queue;
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

    CreateOutputImageDescriptor(outputImage);
}

void ImGuiOverlay::CreateOutputImageDescriptor(Image& outputImage)
{
    vk::SamplerCreateInfo samplerInfo{};
    samplerInfo.magFilter = vk::Filter::eLinear;
    samplerInfo.minFilter = vk::Filter::eLinear;
    samplerInfo.addressModeU = vk::SamplerAddressMode::eClampToEdge;
    samplerInfo.addressModeV = vk::SamplerAddressMode::eClampToEdge;
    samplerInfo.addressModeW = vk::SamplerAddressMode::eClampToEdge;

    sampler = context->device->createSamplerUnique(samplerInfo);

    // Descriptor Set Layout
    vk::DescriptorSetLayoutBinding binding{
        0, vk::DescriptorType::eCombinedImageSampler, 1,
        vk::ShaderStageFlagBits::eFragment
    };

    vk::DescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &binding;

    descriptorSetLayout = context->device->createDescriptorSetLayoutUnique(layoutInfo);

    // Allocate Descriptor Set
    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.descriptorPool = descriptorPool.get();
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout.get();

    outputImageDescriptorSet = std::move(
        context->device->allocateDescriptorSetsUnique(allocInfo).front()
    );

    // Update Descriptor Set
    vk::DescriptorImageInfo imageInfo{
        sampler.get(), outputImage.view.get(), vk::ImageLayout::eShaderReadOnlyOptimal
    };

    vk::WriteDescriptorSet write{};
    write.dstSet = outputImageDescriptorSet.get();
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    write.pImageInfo = &imageInfo;

    context->device->updateDescriptorSets(write, nullptr);
}

void ImGuiOverlay::SetBlenderTheme() {
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

void ImGuiOverlay::CreateDescriptorPool() {
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

    descriptorPool = context->device.get().createDescriptorPoolUnique(poolInfo);
}

void ImGuiOverlay::CreateRenderPass() {
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

    renderPass = context->device->createRenderPassUnique(renderPassInfo);
}

void ImGuiOverlay::CreateFrameBuffers(const std::vector<vk::Image>& images) {
    for (const auto& image : images) {
        vk::ImageViewCreateInfo viewInfo{};
        viewInfo.image = image;
        viewInfo.viewType = vk::ImageViewType::e2D;
        viewInfo.format = vk::Format::eB8G8R8A8Unorm; //TODO match Swapchain
        viewInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.layerCount = 1;

        imageViews.push_back(context->device->createImageViewUnique(viewInfo));

        vk::FramebufferCreateInfo framebufferInfo{};
        framebufferInfo.renderPass = renderPass.get();
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.pAttachments = &imageViews.back().get();
        framebufferInfo.width = WIDTH; //TODO
        framebufferInfo.height = HEIGHT;
        framebufferInfo.layers = 1;

        frameBuffers.push_back(context->device->createFramebufferUnique(framebufferInfo));
    }
}

void ImGuiOverlay::Render(vk::CommandBuffer commandBuffer, uint32_t imageIndex) {
    ImGui::Render();

    constexpr VkClearValue clear_value = { { 0.0f, 0.0f, 0.0f, 1.0f } };
    VkRenderPassBeginInfo rpInfo{};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpInfo.renderPass = renderPass.get();
    rpInfo.framebuffer = frameBuffers[imageIndex].get();
    rpInfo.renderArea.extent.width = WIDTH;
    rpInfo.renderArea.extent.height = HEIGHT;
    rpInfo.clearValueCount = 1;
    rpInfo.pClearValues = &clear_value;

    vkCmdBeginRenderPass(commandBuffer, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);
    vkCmdEndRenderPass(commandBuffer);
}

void ImGuiOverlay::tableRowLabel(const char* label) {
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::TextUnformatted(label);
    ImGui::TableSetColumnIndex(1);
}

ImGuiOverlay::~ImGuiOverlay() {
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}