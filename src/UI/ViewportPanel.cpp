#include "ViewportPanel.h"
#include <iostream>
#include "imgui.h"
#include "ImGuizmo.h"
#include "glm/gtc/type_ptr.hpp"
#include "SDL3/SDL_mouse.h"
#include "Camera/PerspectiveCamera.h"

ViewportPanel::ViewportPanel(Scene& scene, const Image& outputColor, Image& outputCrypto, uint32_t width, uint32_t height, const std::string& title)
    : scene(scene), outputCrypto(outputCrypto), width(width), height(height), title(title),
    displayImage(scene.getContext(), width, height, outputColor.getFormat(), vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst),
    pickingBuffer(scene.getContext(), Buffer::Type::Custom, sizeof(uint32_t), nullptr, vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent)
{
    const Context& context = scene.getContext();

    pickingBufferMappedPtr = context.getDevice().mapMemory(pickingBuffer.getMemory(), 0, sizeof(int));

    // Transition image layout
    context.oneTimeSubmit([&](const vk::CommandBuffer cmd) {
        displayImage.setImageLayout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
    });

    // Sampler
    vk::SamplerCreateInfo samplerInfo{};
    samplerInfo.magFilter = vk::Filter::eLinear;
    samplerInfo.minFilter = vk::Filter::eLinear;
    samplerInfo.addressModeU = vk::SamplerAddressMode::eClampToEdge;
    samplerInfo.addressModeV = vk::SamplerAddressMode::eClampToEdge;
    samplerInfo.addressModeW = vk::SamplerAddressMode::eClampToEdge;
    sampler = context.getDevice().createSamplerUnique(samplerInfo);

    // Descriptor set layout
    vk::DescriptorSetLayoutBinding binding{0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment};
    vk::DescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &binding;

    descriptorSetLayout = context.getDevice().createDescriptorSetLayoutUnique(layoutInfo);

    // Descriptor set
    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.descriptorPool = context.getDescriptorPool();
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout.get();

    auto sets = context.getDevice().allocateDescriptorSetsUnique(allocInfo);
    outputImageDescriptorSet = std::move(sets.front());

    vk::DescriptorImageInfo imageInfo{sampler.get(), displayImage.getImageView(), vk::ImageLayout::eShaderReadOnlyOptimal};

    vk::WriteDescriptorSet write{};
    write.dstSet = outputImageDescriptorSet.get();
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    write.pImageInfo = &imageInfo;

    context.getDevice().updateDescriptorSets(write, nullptr);


    // ImGuizmo Style
    ImGuizmo::Style& style = ImGuizmo::GetStyle();
    style.HatchedAxisLineThickness   = 0;
        
    // --- Colors similar to Blender ---
    style.Colors[ImGuizmo::DIRECTION_X]       = ImVec4(0.9f, 0.2f, 0.2f, 1.0f); // X = red
    style.Colors[ImGuizmo::DIRECTION_Y]       = ImVec4(0.2f, 0.9f, 0.2f, 1.0f); // Y = green
    style.Colors[ImGuizmo::DIRECTION_Z]       = ImVec4(0.2f, 0.5f, 1.0f, 1.0f); // Z = blue
    style.Colors[ImGuizmo::PLANE_X]= ImVec4(0.9f, 0.2f, 0.2f, 1.0f); // plane fill
    style.Colors[ImGuizmo::PLANE_Y]= ImVec4(0.2f, 0.9f, 0.2f, 1.0f); // plane fill
    style.Colors[ImGuizmo::PLANE_Z]= ImVec4(0.2f, 0.5f, 1.0f, 1.0f); // plane fill

    // Blender also fades inactive axes → make selection highlight bright
    style.Colors[ImGuizmo::SELECTION]         = ImVec4(1.0f, 1.0f, 0.0f, 1.0f); // yellow highlight
    style.Colors[ImGuizmo::INACTIVE]          = ImVec4(0.4f, 0.4f, 0.4f, 0.6f); // gray inactive
    
    // Rotation circles usually more saturated
    style.Colors[ImGuizmo::ROTATION_USING_BORDER] = ImVec4(1.0f, 0.8f, 0.2f, 1.0f); // golden ring
    style.Colors[ImGuizmo::ROTATION_USING_FILL]   = ImVec4(1.0f, 0.8f, 0.2f, 0.3f); // subtle fill
}

void ViewportPanel::recordCopy(const vk::CommandBuffer cmd, Image& srcImage) {
    srcImage.setImageLayout(cmd, vk::ImageLayout::eTransferSrcOptimal);
    displayImage.setImageLayout(cmd, vk::ImageLayout::eTransferDstOptimal);

    vk::ImageCopy copyRegion{};
    copyRegion.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
    copyRegion.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
    copyRegion.extent = vk::Extent3D{width, height, 1};

    cmd.copyImage(srcImage.getImage(), vk::ImageLayout::eTransferSrcOptimal, displayImage.getImage(), vk::ImageLayout::eTransferDstOptimal, copyRegion);

    srcImage.setImageLayout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
    displayImage.setImageLayout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
}

void ViewportPanel::renderUi() {
    ImGui::Begin(title.c_str());

    float aspectRatio = static_cast<float>(width) / height;

    ImVec2 imageSize;
    ImVec2 availSize = ImGui::GetContentRegionAvail();
    if (availSize.x / availSize.y > aspectRatio) {
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
    ImGui::SetCursorPos({cursorPos.x + padding.x, cursorPos.y + padding.y});
    ImVec2 imagePos = ImGui::GetCursorScreenPos();

    ImGui::Image(static_cast<VkDescriptorSet>(outputImageDescriptorSet.get()), imageSize);
    bool isImageHovered = ImGui::IsItemHovered();

    auto* camera = scene.getActiveCamera();
    
    // --- Gizmo Section ---
    if (auto activeObject = scene.getActiveObject()) {
        
        ImGuizmo::Style& style = ImGuizmo::GetStyle();
        
        // Clamp scale so it doesn't explode on tiny windows
        float scale = std::max(imageSize.x /  1080.0f, 0.5f);
        style.TranslationLineThickness   = 4.0f * scale;
        style.TranslationLineArrowSize   = 6.0f * scale;
        style.RotationLineThickness      = 6.0f * scale;
        style.RotationOuterLineThickness = 2.0f * scale;
        style.ScaleLineThickness         = 4.0f * scale;
        style.ScaleLineCircleSize        = 8.0f * scale;
        style.CenterCircleSize           = 5.0f * scale;
        
        ImGuizmo::BeginFrame();
        ImGuizmo::SetRect(imagePos.x, imagePos.y, imageSize.x, imageSize.y);
        ImGuizmo::SetDrawlist(ImGui::GetWindowDrawList());

        if (isImageHovered) {
            if (ImGui::IsKeyPressed(ImGuiKey_W)) currentOperation = ImGuizmo::TRANSLATE;
            if (ImGui::IsKeyPressed(ImGuiKey_E)) currentOperation = ImGuizmo::ROTATE;
            if (ImGui::IsKeyPressed(ImGuiKey_R)) currentOperation = ImGuizmo::SCALE;
        }

        const mat4& view = camera->getViewMatrix();
        mat4 proj = camera->getProjectionMatrix();

        mat4 model = activeObject->getTransform().getMatrix();
        ImGuizmo::Manipulate(value_ptr(view),value_ptr(proj), currentOperation, currentMode, value_ptr(model));

        if (ImGuizmo::IsUsing())
            activeObject->setTransformMatrix(model);
    }

    // --- Mouse Control Section ---
    SDL_Window* sdlWindow = scene.getContext().getWindow();

    if (isImageHovered && !ImGuizmo::IsUsing() && ImGui::IsMouseDown(ImGuiMouseButton_Right) && !isCapturingMouse) {
        SDL_GetMouseState(&oldX, &oldY);
        isCapturingMouse = true;
        ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NoMouse;
        SDL_SetWindowRelativeMouseMode(sdlWindow, true);
        (void)SDL_GetRelativeMouseState(nullptr, nullptr); // fix mouse jump
    }

    if (isCapturingMouse && ImGui::IsMouseReleased(ImGuiMouseButton_Right)) {
        isCapturingMouse = false;
        SDL_SetWindowRelativeMouseMode(sdlWindow, false);
        ImGui::GetIO().ConfigFlags &= ~ImGuiConfigFlags_NoMouse;
        SDL_WarpMouseInWindow(sdlWindow, oldX, oldY);
    }

    if (isCapturingMouse)
        camera->update();

    // --- Picking Section ---
    if (isImageHovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !ImGuizmo::IsOver())
        handlePicking(imageSize);

    ImGui::End();
}

void ViewportPanel::handlePicking(const ImVec2 imageSize) const
{
    const ImVec2 mousePos = ImGui::GetMousePos();
    const ImVec2 windowPos = ImGui::GetItemRectMin(); // Top-left corner of the image in screen space
    const float pixelX = static_cast<uint32_t>(std::clamp((mousePos.x - windowPos.x) / imageSize.x, 0.f, 1.f) * width);
    const float pixelY =  static_cast<uint32_t>(std::clamp((mousePos.y - windowPos.y) / imageSize.y, 0.f, 1.f) * height);
    
    scene.getContext().oneTimeSubmit([&](const vk::CommandBuffer cmd) {
        outputCrypto.setImageLayout(cmd, vk::ImageLayout::eTransferSrcOptimal);

        //Copy 1x1 pixel from image to buffer
        vk::BufferImageCopy copyRegion{};
        copyRegion.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
        copyRegion.imageOffset = vk::Offset3D{ static_cast<int32_t>(pixelX), static_cast<int32_t>(pixelY), 0 };
        copyRegion.imageExtent = vk::Extent3D{ 1, 1, 1 };
        cmd.copyImageToBuffer(outputCrypto.getImage(), vk::ImageLayout::eTransferSrcOptimal, pickingBuffer.getBuffer(), copyRegion);

        outputCrypto.setImageLayout(cmd, vk::ImageLayout::eGeneral);
    });

    int instanceId = -1;
    if (pickingBufferMappedPtr)
        instanceId = *static_cast<int*>(pickingBufferMappedPtr);
    
    std::cout << "Picked instance ID: " << instanceId << std::endl;

    if (instanceId == -1)
        scene.setActiveObjectIndex(instanceId);
    else
    {
        SceneObject* pickedObject = scene.getMeshInstances()[instanceId];
        const auto& objects = scene.getSceneObjects();
        for (size_t i = 0; i < objects.size(); ++i) {
            if (objects[i].get() == pickedObject) {
                scene.setActiveObjectIndex(static_cast<uint32_t>(i));
                break;
            }
        }
    }
}

ViewportPanel::~ViewportPanel() {
    if (pickingBufferMappedPtr)
        scene.getContext().getDevice().unmapMemory(pickingBuffer.getMemory());

    std::cout << "Destroying ViewportPanel" << std::endl;
}