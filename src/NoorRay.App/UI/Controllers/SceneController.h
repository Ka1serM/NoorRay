#pragma once

#include <cstdint>
#include <filesystem>
#include <unordered_set>
#include <variant>
#include <vector>

#include <RmlUi/Core/DataModelHandle.h>
#include <RmlUi/Core/Event.h>
#include <RmlUi/Core/Types.h>
#include <SDL3/SDL_video.h>

#include "NoorRay.App/UI/Controllers/Selection.h"
#include "NoorRay/Session/Session.h"

namespace RmlKit::UI {
class Model;
}

namespace NoorRay::App {

class AppModel;
class InspectorController;
class MaterialController;
class ViewportElementInstancer;

class SceneController {
public:
    SceneController(RmlKit::UI::Model& model, AppModel& app, SDL_Window* window, Session::Session& session, ViewportElementInstancer& viewports,
                    InspectorController& inspector, MaterialController& materials);
    SceneController(const SceneController&) = delete;
    SceneController& operator=(const SceneController&) = delete;

    void openFile(const std::filesystem::path& path, bool replace);
    void select(const Session::PickResult& picked);

private:
    void onOpen(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onImport(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onImportEnvironment(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onSave(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onNew(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onAdd(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onSelect(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onRemove(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onFilter(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onFrame(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onCopy(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onPaste(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onDuplicate(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onToggle(Rml::DataModelHandle, Rml::Event& event, const Rml::VariantList& arguments);
    void onDragMove(Rml::DataModelHandle, Rml::Event& event, const Rml::VariantList& arguments);
    void onDragOut(Rml::DataModelHandle, Rml::Event& event, const Rml::VariantList& arguments);
    void onDrop(Rml::DataModelHandle, Rml::Event& event, const Rml::VariantList& arguments);
    void onClearMaterial(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onCreateMaterial(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);

    //! A copied object, held by value, so a paste still works after the original is removed.
    enum class DropPlace { None, Into, Before, After, End };

    struct DropTarget {
        int row = -1;
        DropPlace place = DropPlace::None;
    };

    using Clipboard = std::variant<std::monostate, Scene::Instance, Scene::Light, Scene::Camera, Scene::GaussianCloud, Scene::Volume>;

    Clipboard copySelection() const;
    void paste(const Clipboard& copied);
    void chooseFile(bool replace);
    void openEnvironment(const std::filesystem::path& path);
    void select(const Selection& selected);
    void selectGroup(Scene::NodeHandle group);
    void expandTo(Scene::NodeHandle handle);
    DropTarget findDropTarget(const Rml::Event& event) const;
    void showDropTarget(const DropTarget& target);
    void dropMaterial(int material, Scene::NodeHandle target);
    void moveNode(Scene::NodeHandle moved, Scene::NodeHandle target, DropPlace place);
    bool addRows(Scene::NodeHandle handle, int depth);
    void refresh();
    void sceneChanged();

    RmlKit::UI::Model& model;
    AppModel& app;
    SDL_Window* window;
    Session::Session& session;
    ViewportElementInstancer& viewports;
    InspectorController& inspector;
    MaterialController& materials;
    std::vector<Scene::NodeHandle> rowNodes;
    std::unordered_set<std::uint64_t> collapsedNodes;
    Scene::NodeHandle selectedGroup;
    std::filesystem::path scenePath;
    Clipboard clipboard;
};

}
