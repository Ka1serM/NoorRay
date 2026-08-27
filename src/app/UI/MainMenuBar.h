#pragma once

#include "UI/ImGuiComponent.h"
#include "portable-file-dialogs.h"
#include <string>

#include "ImGuiManager.h"

class Scene;

class MainMenuBar : public ImGuiComponent {
public:
    MainMenuBar(std::string name, Scene& scene,
        ImGuiManager& manager, std::string currentScenePath = {});
    void renderUi() override;
    const std::string& getCurrentScenePath() const { return currentScenePath; }

private:
    void renderFileMenu();
    void renderViewMenu() const;
    void renderAddMenu();
    void handleFileImport(const std::string& filePath);
    void handleMaterialXImport(const std::string& filePath);
    void openScene(const std::string& filePath);
    void saveScene(const std::string& filePath);

    Scene& scene;
    ImGuiManager& imGuiManager;

    std::unique_ptr<pfd::open_file> openDialog;
    std::unique_ptr<pfd::open_file> sceneOpenDialog;
    std::unique_ptr<pfd::save_file> sceneSaveDialog;
    std::string currentScenePath;
};
