#pragma once

#include <string>

class ImGuiComponent {
protected:
    std::string name;
public:
    ImGuiComponent() : name("") {}

    //explicit so no conversions happen
    explicit ImGuiComponent(std::string name) : name(std::move(name)) {}
    
    virtual ~ImGuiComponent() = default;

    virtual void renderUi() {}
    virtual std::string getType() const { return "ImGuiComponent"; }
    const std::string& getName() const { return name; }
};