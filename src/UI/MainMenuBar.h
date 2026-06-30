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
    // Enum to keep track of the file type for the pending import.
    enum class FileType {
        NONE,
        OBJ,
        GLTF,
        TEXTURE,
        NRSCENE
    };

    void renderFileMenu();
    void renderViewMenu() const;
    void renderAddMenu() const;
    void handleFileImport(const std::string& filePath, FileType type) const;

    Scene& scene;
    Context& context;
    ImGuiManager& imGuiManager;

    std::unique_ptr<pfd::open_file> openDialog;
    FileType pendingFileType = FileType::NONE;
};