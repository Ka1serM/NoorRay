#pragma once

#include "ImGuiComponent.h"
#include <functional>
#include <map>
#include <string>

class MainMenuBar : public ImGuiComponent {
public:

    MainMenuBar();
    virtual ~MainMenuBar() = default;

    void renderUi() override;
    std::string getType() const override { return "Main Menu"; }
    
    // Register callbacks for menu actions
    void setCallback(const std::string& action, std::function<void()> callback);

private:
    std::map<std::string, std::function<void()>> callbacks;
    
    void renderFileMenu();
};