#pragma once

#include <memory>
#include <vector>
#include <string>
#include <glm/vec3.hpp>

#include "ImGuiComponent.h"
#include "Vulkan/Context.h"

class ImGuiManager {
private:
    vk::UniqueDescriptorPool descriptorPool;
    std::vector<vk::UniqueImageView> imageViews;
    std::vector<vk::UniqueFramebuffer> frameBuffers;
    vk::UniqueRenderPass renderPass;

    // GUI component system
    std::vector<std::unique_ptr<ImGuiComponent>> components;

    void CreateDescriptorPool(Context& context);
    void CreateRenderPass(Context& context);
    void CreateFrameBuffers(Context& context, const std::vector<vk::Image>& images, uint32_t width, uint32_t height);
    static void SetBlenderTheme();
    static void setupDockSpace();

public:
    ImGuiManager(Context& context, const std::vector<vk::Image>& swapchainImages, uint32_t width, uint32_t height);
    ~ImGuiManager();

    vk::DescriptorPool getDescriptorPool() const { return descriptorPool.get(); }

    void renderUi() const;

    void Draw(vk::CommandBuffer commandBuffer, uint32_t imageIndex, uint32_t width, uint32_t height);

    // Component system
    void addComponent(std::unique_ptr<ImGuiComponent> component);
    ImGuiComponent* getComponent(const std::string& name) const;

    // Utility
    static void tableRowLabel(const char* label);

    static void dragFloatRow(const char* label, float value, float speed, float min, float max, const std::function<void(float)>& setter);

    // Vec3 row
    static void dragFloat3Row(const char* label, glm::vec3 value, float speed, const std::function<void(glm::vec3)>& setter);
    
    static void colorEdit3Row(const char* label, glm::vec3 value, const std::function<void(glm::vec3)>& setter);

};
