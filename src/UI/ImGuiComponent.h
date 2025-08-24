#pragma once

#include <string>
#include <map>
#include <functional>

class ImGuiComponent {
public:
    virtual ~ImGuiComponent() = default;

    // Main ImGui draw function
    virtual void renderUi() {}
    virtual std::string getType() const { return "ImGuiComponent"; }

    void setCallback(const std::string& action, std::function<void()> callback) {
        callbacks[action] = std::move(callback);
    }

protected:
    void invokeCallback(const std::string& action) {
        auto it = callbacks.find(action);
        if (it != callbacks.end() && it->second)
            it->second();
    }

private:
    std::map<std::string, std::function<void()>> callbacks;
};
