#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <span>
#include <unordered_set>
#include <variant>
#include <vector>

#include <RmlUi/Core/DataModelHandle.h>
#include <RmlUi/Core/Event.h>
#include <RmlUi/Core/Types.h>
#include <SDL3/SDL_video.h>

#include "NoorRay.App/UI/Controllers/Selection.h"
#include "NoorRay.App/UI/Models/OutlinerModel.h"
#include "NoorRay/Session/Session.h"
#include "RmlKit/UI/Controls/VirtualGrid.h"
#include "RmlKit/UI/Models/ListSelection.h"

namespace RmlKit::UI {
class Model;
}

namespace NoorRay::App {

class AppModel;
class InspectorController;
class MaterialController;
class NodeGraphController;
class ViewportElementInstancer;

class SceneController {
public:
    SceneController(RmlKit::UI::Model& model, AppModel& app, SDL_Window* window, Session::Session& session, ViewportElementInstancer& viewports,
                    InspectorController& inspector, MaterialController& materials, NodeGraphController& nodeGraph);
    SceneController(const SceneController&) = delete;
    SceneController& operator=(const SceneController&) = delete;

    enum class EditAction { Delete, Cut, Copy, Paste, Duplicate, SelectAll };

    void openFile(const std::filesystem::path& path, bool replace);
    void select(const Session::PickResult& picked);
    void sceneChanged();

    //! Called after a new node material is made, so the node editor opens on it.
    std::function<void(Scene::MaterialHandle material)> nodeMaterialCreated;

private:
    void onOpen(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onImport(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onImportEnvironment(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onSave(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onNew(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onAdd(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onSelect(Rml::DataModelHandle, Rml::Event& event, const Rml::VariantList& arguments);
    void onFilter(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onFrame(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onEdit(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onToggle(Rml::DataModelHandle, Rml::Event& event, const Rml::VariantList& arguments);
    void onDragMove(Rml::DataModelHandle, Rml::Event& event, const Rml::VariantList& arguments);
    void onDragOut(Rml::DataModelHandle, Rml::Event& event, const Rml::VariantList& arguments);
    void onDrop(Rml::DataModelHandle, Rml::Event& event, const Rml::VariantList& arguments);
    void onSlotDragEnd(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onClearMaterial(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onCreateMaterial(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onSelectMaterial(Rml::DataModelHandle, Rml::Event& event, const Rml::VariantList& arguments);
    void onEditMaterial(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);

    enum class DropPlace { None, Into, Before, After, End };

    struct DropTarget {
        int row = -1;
        DropPlace place = DropPlace::None;
        //! The material slot under a dragged material, or -1 for the whole object.
        int slot = -1;
    };

    //! The list that the edit hotkeys act on: the one the user clicked last.
    enum class EditList { Objects, Materials };

    //! A copied object, held by value, so a paste still works after the original is removed.
    using ClipObject = std::variant<Scene::Instance, Scene::Light, Scene::Camera, Scene::GaussianCloud, Scene::Volume>;

    void editObjects(EditAction action);
    void editMaterials(EditAction action);
    std::vector<ClipObject> copySelection() const;
    void paste(std::span<const ClipObject> copied);
    Selection pasteObject(const ClipObject& copied);
    void deleteSelection();
    void chooseFile(bool replace);
    void openEnvironment(const std::filesystem::path& path);
    void select(const Selection& selected);
    void selectGroup(Scene::NodeHandle group);
    void showSelection();
    void expandTo(Scene::NodeHandle handle);
    DropTarget findDropTarget(const Rml::Event& event) const;
    void showDropTarget(const DropTarget& target);
    void dropMaterial(int material, Scene::NodeHandle target, int slot);
    void setSlotMaterial(Scene::NodeHandle target, int slot, Scene::MaterialHandle material);
    Scene::MaterialHandle addMaterial(Scene::Material material);
    std::vector<Scene::MaterialHandle> materialOrder() const;
    void deleteSelectedMaterials();
    void pasteMaterials(std::span<const Scene::Material> copied);
    void showMaterials();
    void moveNode(Scene::NodeHandle moved, Scene::NodeHandle target, DropPlace place);
    bool addRows(Scene::NodeHandle handle, int depth);
    void refresh();

    RmlKit::UI::Model& model;
    AppModel& app;
    SDL_Window* window;
    Session::Session& session;
    ViewportElementInstancer& viewports;
    InspectorController& inspector;
    MaterialController& materials;
    NodeGraphController& nodeGraph;
    std::vector<Scene::NodeHandle> rowNodes;
    std::unordered_set<std::uint64_t> collapsedNodes;
    //! Set when a dragged slot material lands on a slot; a slot material dropped anywhere else leaves its slot empty.
    bool slotDropped = false;
    //! The selected nodes; the primary one is shown in the inspector and moved by the gizmo.
    RmlKit::UI::ListSelection<Scene::NodeHandle> selectedNodes;
    RmlKit::UI::ListSelection<Scene::MaterialHandle> selectedMaterials;
    EditList editList = EditList::Objects;
    std::filesystem::path scenePath;
    std::vector<ClipObject> clipboard;
    std::vector<Scene::Material> copiedMaterials;
    RmlKit::UI::VirtualGrid<MaterialCard> materialGrid;
};

}
