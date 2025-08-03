#pragma once

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include "glm/glm.hpp"
#include "Vulkan/Context.h"

class ImGuiComponent;

class ImGuiManager {
public:
    ImGuiManager(Context& context, const std::vector<vk::Image>& swapchainImages, uint32_t width, uint32_t height);
    ~ImGuiManager();
    
    void recreateForSwapChain(const std::vector<vk::Image>& swapchainImages, uint32_t width, uint32_t height);

    void renderUi();
    void Draw(vk::CommandBuffer commandBuffer, uint32_t imageIndex, uint32_t width, uint32_t height);

    void add(std::unique_ptr<ImGuiComponent> component);
    ImGuiComponent* getComponent(const std::string& name) const;

    // UI Helpers
    static void tableRowLabel(const char* label);
    static void dragFloatRow(const char* label, float value, float speed, float min, float max, const std::function<void(float)>& setter);
    static void dragFloat3Row(const char* label, glm::vec3 value, float speed, const std::function<void(glm::vec3)>& setter);
    static void colorEdit3Row(const char* label, glm::vec3 value, const std::function<void(glm::vec3)>& setter);

private:
    void SetBlenderTheme();
    void setupDockSpace();
    void CreateRenderPass();
    void CreateFrameBuffers(const std::vector<vk::Image>& images, uint32_t width, uint32_t height);

    void cleanupSwapChainResources();

    Context& context;

    std::vector<std::unique_ptr<ImGuiComponent>> components;
    vk::UniqueRenderPass renderPass;
    std::vector<vk::UniqueImageView> imageViews;
    std::vector<vk::UniqueFramebuffer> frameBuffers;
};
