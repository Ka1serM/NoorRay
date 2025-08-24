#pragma once

#include "UI/ImGuiComponent.h"
#include <string>
#include <future>
#include <vector>

class Scene;
class Context;

class MainMenuBar : public ImGuiComponent {
public:
    MainMenuBar(std::string name, Context& context, Scene& scene);
    void renderUi() override;

    std::string getType() const override { return "Main Menu"; }

private:
    // Enum to keep track of the file type for the pending async import.
    enum class FileType {
        OBJ,
        CRTSCENE,
        TEXTURE
    };

    void renderFileMenu();
    void renderAddMenu() const;
    void handleFileImport(const std::string& filePath, FileType type) const;

    // Member references to core application systems.
    Scene& scene;
    Context& context;

    // Asynchronous file dialog management.
    // openFuture stores the result of the async file dialog.
    std::future<std::vector<std::string>> openFuture;
    // pendingFileType tracks which import action to take once the dialog returns.
    FileType pendingFileType;
};