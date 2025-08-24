#pragma once

#include "ImGuiComponent.h"
#include <string>

class MainMenuBar : public ImGuiComponent {
public:

    MainMenuBar();

    void renderUi() override;
    std::string getType() const override { return "Main Menu"; }

private:
   
    void renderFileMenu();
    void renderAddMenu();
};