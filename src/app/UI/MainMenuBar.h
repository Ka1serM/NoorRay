#pragma once

#include "UI/ImGuiComponent.h"
#include "portable-file-dialogs.h"
#include <string>

#include "ImGuiManager.h"

class Scene;
class Context;

class MainMenuBar : public ImGuiComponent {
public:
    MainMenuBar(std::string name, Context& context, Scene& scene, ImGuiManager& manager);
    void renderUi() override;

private:
    void renderFileMenu();
    void renderViewMenu() const;
    void renderAddMenu() const;
    void handleFileImport(const std::string& filePath);
    void openScene(const std::string& filePath);
    void saveScene(const std::string& filePath);

    Scene& scene;
    Context& context;
    ImGuiManager& imGuiManager;

    std::unique_ptr<pfd::open_file> openDialog;
    std::unique_ptr<pfd::open_file> sceneOpenDialog;
    std::unique_ptr<pfd::save_file> sceneSaveDialog;
    std::string currentScenePath;
};
