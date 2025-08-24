#pragma once

#include "ImGuiComponent.h"
#include <string>

class RenderPanel : public ImGuiComponent {
private:
    std::string saveLocation;
    
public:
    RenderPanel();

    void renderUi() override;

    std::string getType() const override { return "Render"; }
};