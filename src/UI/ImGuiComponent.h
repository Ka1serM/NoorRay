#pragma once

#include <string>

class ImGuiComponent {
public:
    virtual ~ImGuiComponent() = default;
    virtual void renderUi() {}
    virtual  std::string getType() const { return "ImGuiComponent"; }
};
