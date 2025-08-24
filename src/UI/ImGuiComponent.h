#pragma once

#include <string>
#include <functional>

class ImGuiComponent {
public:
    ImGuiComponent() : name("") {}

    //explicit so no conversions happen
    explicit ImGuiComponent(std::string name) : name(std::move(name)) {}
    
    virtual ~ImGuiComponent() = default;

    virtual void renderUi() {}
    virtual std::string getType() const { return "ImGuiComponent"; }
    const std::string& getName() const { return name; }
    
protected:
    std::string name;
};