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
    enum class Theme {
        Dark,
        Light
    };
    
    ImGuiManager(Context& context, uint32_t numImages, vk::SurfaceFormatKHR renderTargetFormat);
    ~ImGuiManager();
    void render(vk::CommandBuffer commandBuffer, vk::ImageView targetView, vk::Extent2D targetExtent);
    void processEvent(const SDL_Event& event);

    template<typename T, typename... Args>
    T* addComponent(Args&&... args) {
        static_assert(std::is_base_of_v<ImGuiComponent, T>, "T must derive from ImGuiComponent");
    
        auto ptr = std::make_unique<T>(std::forward<Args>(args)...);
        T* rawPtr = ptr.get();                  // store raw pointer to return
        components.emplace_back(std::move(ptr)); // store in container
        return rawPtr;
    }

    static void SetDarkTheme();
    static void SetLightTheme();
    ImGuiComponent* getComponent(const std::string& name) const;

    static void tableRowLabel(const char* label);
    static void checkboxRow(const char* label, bool value, const std::function<void(bool)>& setter);
    static void dragFloatRow(const char* label, float value, float speed, float min, float max, const std::function<void(float)>& setter);
    static void dragFloat3Row(const char* label, glm::vec3 value, float speed, const std::function<void(glm::vec3)>& setter);
    static void colorEdit3Row(const char* label, glm::vec3 value, const std::function<void(glm::vec3)>& setter);
    static void colorEdit4Row(const char* label, glm::vec4 value, const std::function<void(glm::vec4)>& setter);

    void SetTheme(Theme theme);
    Theme GetCurrentTheme() const { return currentTheme; }
private:
    Context& context;
    std::vector<std::unique_ptr<ImGuiComponent>> components;
    Theme currentTheme = Theme::Dark;
};