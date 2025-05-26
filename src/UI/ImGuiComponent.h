#pragma once

#include <string>

class ImGuiComponent {
public:
    virtual ~ImGuiComponent() = default;
    virtual void renderUi() = 0;
    virtual std::string getType() const = 0;
};