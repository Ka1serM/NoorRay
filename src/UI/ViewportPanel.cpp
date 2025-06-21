#include "ViewportPanel.h"
#include <imgui.h>


ViewportPanel::ViewportPanel(Context& context, const vk::DescriptorPool descriptorPool, const vk::ImageView imageView,uint32_t width, uint32_t height)
: width(width), height(height)
{
    // Create Vulkan sampler
    vk::SamplerCreateInfo samplerInfo{};
    samplerInfo.magFilter = vk::Filter::eLinear;
    samplerInfo.minFilter = vk::Filter::eLinear;
    samplerInfo.addressModeU = vk::SamplerAddressMode::eClampToEdge;
    samplerInfo.addressModeV = vk::SamplerAddressMode::eClampToEdge;
    samplerInfo.addressModeW = vk::SamplerAddressMode::eClampToEdge;

    sampler = context.device->createSamplerUnique(samplerInfo);

    // Create descriptor set layout
    vk::DescriptorSetLayoutBinding binding{0,vk::DescriptorType::eCombinedImageSampler,1,vk::ShaderStageFlagBits::eFragment};

    vk::DescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &binding;

    descriptorSetLayout = context.device->createDescriptorSetLayoutUnique(layoutInfo);

    // Allocate descriptor set
    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout.get();

    auto sets = context.device->allocateDescriptorSetsUnique(allocInfo);
    outputImageDescriptorSet = std::move(sets.front());

    // Update descriptor set with image info
    vk::DescriptorImageInfo imageInfo{sampler.get(),imageView,vk::ImageLayout::eShaderReadOnlyOptimal};

    vk::WriteDescriptorSet write{};
    write.dstSet = outputImageDescriptorSet.get();
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    write.pImageInfo = &imageInfo;

    context.device->updateDescriptorSets(write, nullptr);
}

void ViewportPanel::renderUi() {
    ImGui::Begin(getType().c_str());

    ImVec2 availSize = ImGui::GetContentRegionAvail();
    float aspectRatio = static_cast<float>(width) / static_cast<float>(height);
    float availAspectRatio = availSize.x / availSize.y;

    ImVec2 imageSize;
    if (availAspectRatio > aspectRatio) {
        imageSize.y = availSize.y;
        imageSize.x = imageSize.y * aspectRatio;
    } else {
        imageSize.x = availSize.x;
        imageSize.y = imageSize.x / aspectRatio;
    }

    ImVec2 padding = {
        (availSize.x - imageSize.x) * 0.5f,
        (availSize.y - imageSize.y) * 0.5f
    };

    ImVec2 cursorPos = ImGui::GetCursorPos();
    ImGui::SetCursorPos(ImVec2(cursorPos.x + padding.x, cursorPos.y + padding.y));

    ImGui::Image(reinterpret_cast<ImTextureID>(static_cast<VkDescriptorSet>(outputImageDescriptorSet.get())), imageSize, ImVec2(0.0f, 0.0f), ImVec2(1.0f, 1.0f));

    ImGui::End();
}
