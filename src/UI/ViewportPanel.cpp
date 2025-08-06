#include "ViewportPanel.h"
#include <iostream>
#include "imgui.h"
#include "Camera/PerspectiveCamera.h"
#include "glm/gtc/type_ptr.hpp"
#include "SDL3/SDL_mouse.h"

ViewportPanel::ViewportPanel(Scene& scene, const Image& inputImage,uint32_t width, uint32_t height, const std::string& title)
: scene(scene), width(width), height(height), title(title), displayImage(scene.getContext(), width, height, inputImage.getFormat(), vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst)
{
    Context& context = scene.getContext();
    context.oneTimeSubmit([&](const vk::CommandBuffer cmd) {
        displayImage.setImageLayout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
    });

    // Create Vulkan sampler
    vk::SamplerCreateInfo samplerInfo{};
    samplerInfo.magFilter = vk::Filter::eLinear;
    samplerInfo.minFilter = vk::Filter::eLinear;
    samplerInfo.addressModeU = vk::SamplerAddressMode::eClampToEdge;
    samplerInfo.addressModeV = vk::SamplerAddressMode::eClampToEdge;
    samplerInfo.addressModeW = vk::SamplerAddressMode::eClampToEdge;

    sampler = context.getDevice().createSamplerUnique(samplerInfo);

    // Create descriptor set layout
    vk::DescriptorSetLayoutBinding binding{0,vk::DescriptorType::eCombinedImageSampler,1,vk::ShaderStageFlagBits::eFragment};

    vk::DescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &binding;

    descriptorSetLayout = context.getDevice().createDescriptorSetLayoutUnique(layoutInfo);

    // Allocate descriptor set
    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.descriptorPool = context.getDescriptorPool();
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout.get();

    auto sets = context.getDevice().allocateDescriptorSetsUnique(allocInfo);
    outputImageDescriptorSet = std::move(sets.front());

    vk::DescriptorImageInfo imageInfo{sampler.get(), displayImage.getImageView(),vk::ImageLayout::eShaderReadOnlyOptimal};

    vk::WriteDescriptorSet write{};
    write.dstSet = outputImageDescriptorSet.get();
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    write.pImageInfo = &imageInfo;

    context.getDevice().updateDescriptorSets(write, nullptr);
}

void ViewportPanel::recordCopy(const vk::CommandBuffer cmd, Image& srcImage)
{
    // Transition layouts for the copy operation
    srcImage.setImageLayout(cmd, vk::ImageLayout::eTransferSrcOptimal);
    displayImage.setImageLayout(cmd, vk::ImageLayout::eTransferDstOptimal);

    // Execute the copy from the tonemapped image to our display image
    vk::ImageCopy copyRegion{};
    copyRegion.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
    copyRegion.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
    copyRegion.extent = vk::Extent3D{width, height, 1};
    cmd.copyImage(srcImage.getImage(), vk::ImageLayout::eTransferSrcOptimal, displayImage.getImage(), vk::ImageLayout::eTransferDstOptimal, copyRegion);

    // Transition both images back to ShaderReadOnlyOptimal so they can be used as textures again
    srcImage.setImageLayout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
    displayImage.setImageLayout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
}

void ViewportPanel::renderUi() {
    ImGui::Begin(title.c_str());

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
    ImVec2 imagePos = ImGui::GetCursorScreenPos();
    
    ImGui::Image(reinterpret_cast<ImTextureID>(static_cast<VkDescriptorSet>(outputImageDescriptorSet.get())), imageSize, ImVec2(0.0f, 0.0f), ImVec2(1.0f, 1.0f));
    bool isImageHovered = ImGui::IsItemHovered();

    auto* camera = scene.getActiveCamera();
        
    //SECTION RENDER GIZMO
    if (auto activeObject = scene.getActiveObject())
    {
        ImGuizmo::BeginFrame();
        ImGuizmo::SetRect(imagePos.x, imagePos.y, imageSize.x, imageSize.y);
        ImGuizmo::SetDrawlist(ImGui::GetWindowDrawList());

        if (isImageHovered) {
            if (ImGui::IsKeyPressed(ImGuiKey_W))
                currentOperation = ImGuizmo::TRANSLATE;
            if (ImGui::IsKeyPressed(ImGuiKey_E))
                currentOperation = ImGuizmo::ROTATE;
            if (ImGui::IsKeyPressed(ImGuiKey_R))
                currentOperation = ImGuizmo::SCALE;
        }

        const mat4& view = camera->getViewMatrix();
        mat4 proj = camera->getProjectionMatrix();
        
        mat4 model = activeObject->getTransform().getMatrix();
        static ImGuizmo::MODE currentMode = ImGuizmo::LOCAL;
        ImGuizmo::Manipulate(value_ptr(view), value_ptr(proj), currentOperation, currentMode, value_ptr(model));
        
        if (ImGuizmo::IsUsing())
            activeObject->setTransformMatrix(model);
    }

    //SECTION HANDLE INPUT
    SDL_Window* sdlWindow = scene.getContext().getWindow();

    if (isImageHovered && !ImGuizmo::IsOver() && ImGui::IsMouseDown(ImGuiMouseButton_Right) && !isCapturingMouse) {
        SDL_GetMouseState(&oldX, &oldY);
        isCapturingMouse = true;
        ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NoMouse;
        SDL_SetWindowRelativeMouseMode(sdlWindow, true);
    }

    if (isCapturingMouse && ImGui::IsMouseReleased(ImGuiMouseButton_Right)) {
        isCapturingMouse = false;
        SDL_SetWindowRelativeMouseMode(sdlWindow, false);
        ImGui::GetIO().ConfigFlags &= ~ImGuiConfigFlags_NoMouse;
        SDL_WarpMouseInWindow(sdlWindow, oldX, oldY);
    }

    if (isCapturingMouse)
        camera->update();
    
    ImGui::End();
}

ViewportPanel::~ViewportPanel()
{
    std::cout << "Destroying ViewportPanel" << std::endl;
}
