#pragma once

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include "glm/glm.hpp"
#include "SDL3/SDL_events.h"
#include "Vulkan/Context.h"

class ImGuiComponent;

class ImGuiManager {
public:
    ImGuiManager(Context& context, const std::vector<vk::Image>& swapchainImages);
    ~ImGuiManager();
    
    void recreateForSwapChain(const std::vector<vk::Image>& swapchainImages);

    void renderUi();
    void Draw(vk::CommandBuffer commandBuffer, uint32_t imageIndex);

    // Create and add a component of type T with constructor arguments Args
    template<typename T, typename... Args>
    void addComponent(Args&&... args) {
        static_assert(std::is_base_of_v<ImGuiComponent, T>, "T must derive from ImGuiComponent");
        components.emplace_back(std::make_unique<T>(std::forward<Args>(args)...));
    }
    
    ImGuiComponent* getComponent(const std::string& name) const;

    // UI Helpers
    static void tableRowLabel(const char* label);
    static void checkboxRow(const char* label, bool value, const std::function<void(bool)>& setter);
    static void dragFloatRow(const char* label, float value, float speed, float min, float max, const std::function<void(float)>& setter);
    static void dragFloat3Row(const char* label, glm::vec3 value, float speed, const std::function<void(glm::vec3)>& setter);
    static void colorEdit3Row(const char* label, glm::vec3 value, const std::function<void(glm::vec3)>& setter);
    static void colorEdit4Row(const char* label, glm::vec4 value, const std::function<void(glm::vec4)>& setter);

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
