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
    ImGuiManager(Context& context, uint32_t numImages, vk::SurfaceFormatKHR renderTargetFormat);
    ~ImGuiManager();
    void render(vk::CommandBuffer commandBuffer, vk::ImageView targetImageView, vk::Extent2D currentExtent);
    void renderToTarget(vk::CommandBuffer commandBuffer, vk::ImageView targetImageView, vk::Extent2D textureExtent, vk::Offset2D mouseOffset, vk::Extent2D mouseExtent, vk::Extent2D windowExtent);
    void processEvent(const SDL_Event& event);

    template<typename T, typename... Args>
    T* addComponent(Args&&... args) {
        static_assert(std::is_base_of_v<ImGuiComponent, T>, "T must derive from ImGuiComponent");
    
        auto ptr = std::make_unique<T>(std::forward<Args>(args)...);
        T* rawPtr = ptr.get();                  // store raw pointer to return
        components.emplace_back(std::move(ptr)); // store in container
        return rawPtr;
    }
    
    ImGuiComponent* getComponent(const std::string& name) const;

    static void tableRowLabel(const char* label);
    static void checkboxRow(const char* label, bool value, const std::function<void(bool)>& setter);
    static void dragFloatRow(const char* label, float value, float speed, float min, float max, const std::function<void(float)>& setter);
    static void dragFloat3Row(const char* label, glm::vec3 value, float speed, const std::function<void(glm::vec3)>& setter);
    static void colorEdit3Row(const char* label, glm::vec3 value, const std::function<void(glm::vec3)>& setter);
    static void colorEdit4Row(const char* label, glm::vec4 value, const std::function<void(glm::vec4)>& setter);

private:
    void renderInternal(vk::CommandBuffer commandBuffer, vk::ImageView targetImageView, vk::Extent2D textureExtent, vk::Offset2D mouseOffset, vk::Extent2D mouseExtent, vk::Extent2D windowExtent, bool remapMouse);
    static void SetBlenderTheme();

    Context& context;
    std::vector<std::unique_ptr<ImGuiComponent>> components;
    VkFormat renderTargetFormat;
};