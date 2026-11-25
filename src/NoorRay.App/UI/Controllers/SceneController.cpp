#include "NoorRay.App/UI/Controllers/SceneController.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <format>
#include <limits>
#include <memory>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>

#include <RmlUi/Core/Element.h>

#include "NoorRay.App/UI/Controllers/InspectorController.h"
#include "NoorRay.App/UI/Controllers/MaterialController.h"
#include "NoorRay.App/UI/Controllers/NodeGraphController.h"
#include "NoorRay.App/UI/Controls/ViewportElement.h"
#include "NoorRay.App/UI/Models/AppModel.h"
#include "NoorRay/Import/EnvironmentImporter.h"
#include "NoorRay/Import/Primitives.h"
#include "NoorRay/Import/SceneImporter.h"
#include "NoorRay/Materials/NodeCatalog.h"
#include "NoorRay/Materials/NodeCompiler.h"
#include "RmlKit/UI/Backend/FileDialog.h"
#include "RmlKit/UI/Models/Model.h"

namespace NoorRay::App {

namespace {

using RmlKit::UI::ReportLevel;

//! The browser build has no USD. There the picker takes a scene together with the files it refers to (.bin, .mtl, textures, includes).
#if PLATFORM_WEB
constexpr std::array sceneFilters{
    RmlKit::UI::FileFilter{"All scenes", "nrs;gltf;glb;obj;ply;pbrt;hdr;exr"},
};
#else
constexpr std::array sceneFilters{
    RmlKit::UI::FileFilter{"All scenes", "nrs;usd;usda;usdc;usdz;gltf;glb;obj;ply;pbrt;hdr;exr"},
    RmlKit::UI::FileFilter{"NoorRay scene", "nrs"},
    RmlKit::UI::FileFilter{"USD", "usd;usda;usdc;usdz"},
    RmlKit::UI::FileFilter{"glTF", "gltf;glb"},
    RmlKit::UI::FileFilter{"Wavefront OBJ", "obj"},
    RmlKit::UI::FileFilter{"Gaussian splats", "ply"},
    RmlKit::UI::FileFilter{"PBRT", "pbrt"},
    RmlKit::UI::FileFilter{"HDRI environment", "hdr;exr"},
};
#endif
constexpr std::array environmentFilters{
    RmlKit::UI::FileFilter{"HDRI environment", "hdr;exr"},
};

bool contains(const std::string& text, const Rml::String& filter) {
    if (filter.empty())
        return true;
    const auto lower = [](const unsigned char character) { return std::tolower(character); };
    return !std::ranges::search(text, filter, {}, lower, lower).empty();
}

const char* lightIcon(const Scene::LightType type) {
    switch (type) {
    case Scene::LightType::Spot:
        return "light-spot";
    case Scene::LightType::Rect:
        return "light-area";
    case Scene::LightType::Directional:
        return "light-sun";
    default:
        return "light-point";
    }
}

//! An empty is a node, not an object, so its object is none.
Scene::NodeObject nodeObjectOf(const Selection& selection) {
    return std::visit(
        [](const auto& handle) -> Scene::NodeObject {
            if constexpr (std::is_same_v<std::decay_t<decltype(handle)>, Scene::NodeHandle>)
                return {};
            else
                return handle;
        },
        selection);
}

Scene::NodeHandle nodeOf(const Scene::Scene& scene, const Selection& selection) {
    if (const auto* node = std::get_if<Scene::NodeHandle>(&selection))
        return *node;
    if (std::holds_alternative<std::monostate>(selection))
        return {};
    return scene.findNode(nodeObjectOf(selection));
}

Selection selectionOf(const Scene::NodeObject& object) {
    return std::visit([](const auto& handle) -> Selection { return handle; }, object);
}

SceneController::EditAction editActionOf(const Rml::VariantList& arguments, const char* event) {
    if (arguments.size() != 1)
        RmlKit::UI::fatal(std::format("{} takes one argument: the action", event));
    const Rml::String action = arguments[0].Get<Rml::String>();
    using enum SceneController::EditAction;
    constexpr std::array<std::pair<std::string_view, SceneController::EditAction>, 6> actions{
        {{"delete", Delete}, {"cut", Cut}, {"copy", Copy}, {"paste", Paste}, {"duplicate", Duplicate}, {"selectAll", SelectAll}}};
    for (const auto& [name, value] : actions)
        if (action == name)
            return value;
    RmlKit::UI::fatal(std::format("{} does not know the action {}", event, action));
}

std::uint64_t nodeKey(const Scene::NodeHandle handle) {
    return std::uint64_t{handle.generation} << 32 | handle.index;
}

OutlinerRow describe(const Scene::Scene& scene, const Scene::Node& node) {
    OutlinerRow row{.name = node.name, .kind = "Empty", .icon = "coord-system"};
    if (const auto* instance = std::get_if<Scene::InstanceHandle>(&node.object)) {
        row.kind = "Mesh";
        const Scene::Instance* object = scene.find(*instance);
        const Scene::Mesh* mesh = object ? scene.find(object->mesh) : nullptr;
        //! The added primitives name their meshes, so the name tells the shape; other meshes show the cube.
        row.icon = mesh && mesh->name == "Sphere" ? "shape-sphere" : mesh && mesh->name == "Plane" ? "shape-plane" : "shape-cube";
        const auto effective = [&](const Scene::MaterialHandle material) {
            const std::uint32_t index = scene.getMaterials().denseIndex(object->material ? object->material : material);
            return index == Scene::MaterialHandle::invalidIndex ? -1 : static_cast<int>(index);
        };
        if (mesh && !mesh->subsets.empty())
            for (const Scene::MeshSubset& subset : mesh->subsets)
                row.slots.push_back(effective(subset.material));
        else if (object)
            row.slots.push_back(effective({}));
    } else if (const auto* light = std::get_if<Scene::LightHandle>(&node.object)) {
        const Scene::Light* object = scene.find(*light);
        row.kind = "Light";
        row.icon = object ? lightIcon(object->type) : "light-point";
    } else if (const auto* camera = std::get_if<Scene::CameraHandle>(&node.object)) {
        row.kind = "Camera";
        row.icon = "object-camera";
        row.active = *camera == scene.getActiveCamera();
    } else if (std::holds_alternative<Scene::GaussianCloudHandle>(node.object)) {
        row.kind = "Gaussians";
        row.icon = "object-array";
    } else if (std::holds_alternative<Scene::VolumeHandle>(node.object)) {
        row.kind = "Volume";
        row.icon = "poly";
    }
    return row;
}

//! The slot index of the material slot an element is in, or -1 when it is not in one.
int slotOf(Rml::Element* element) {
    for (; element && !element->HasAttribute("row"); element = element->GetParentNode())
        if (element->HasAttribute("slot"))
            return element->GetAttribute<int>("slot", -1);
    return -1;
}

constexpr RmlKit::UI::GridLayout materialLayout{.minItemWidth = 72, .heightPerWidth = 0, .fixedHeight = 84, .headerHeight = 0, .gap = 8};

Rml::Element* rowElementOf(Rml::Element* element) {
    while (element && !element->HasAttribute("row"))
        element = element->GetParentNode();
    return element;
}

const Rml::Element* draggedElement(const Rml::Event& event) {
    return static_cast<const Rml::Element*>(event.GetParameter<void*>("drag_element", nullptr));
}

Scene::LightHandle addLightNode(Scene::Scene& scene, Scene::Light light) {
    const Core::Transform transform = light.transform;
    std::string name = light.name;
    const Scene::LightHandle added = scene.addLight(std::move(light));
    scene.addNode({std::move(name), transform, {}, added});
    return added;
}

Scene::Light makeLight(const Scene::LightType type, const Core::Bounds& bounds) {
    Scene::Light light;
    light.type = type;
    const bool empty = bounds.min.x > bounds.max.x;
    const float radius = empty ? 1.0f : std::max(Core::length(bounds.max - bounds.min) * 0.5f, 0.1f);
    //! A new light sits at the origin like a new shape and points down and away from the default view.
    light.transform = Core::Transform::lookAt({0.0f, 0.0f, 0.0f}, {-1.0f, -2.0f, -1.0f}, {0.0f, 1.0f, 0.0f});
    switch (type) {
    case Scene::LightType::Point:
        light.name = "Point Light";
        break;
    case Scene::LightType::Spot:
        light.name = "Spot Light";
        break;
    case Scene::LightType::Rect:
        light.name = "Rect Light";
        light.width = radius;
        light.height = radius;
        break;
    case Scene::LightType::Directional:
        light.name = "Sun";
        light.power = 5.0f;
        break;
    default:
        light.name = "Light";
        break;
    }
    return light;
}

//! "Sun" becomes "Sun.001", and "Sun.001" becomes "Sun.002", up to the first name that no object of the kind has.
template <typename Objects>
std::string copyName(const std::string& name, const Objects& objects) {
    std::string_view base = name;
    if (const std::size_t dot = base.rfind('.');
        dot != std::string_view::npos && dot + 4 == base.size()
        && std::ranges::all_of(base.substr(dot + 1), [](const unsigned char character) { return std::isdigit(character) != 0; }))
        base = base.substr(0, dot);
    for (int number = 1;; ++number) {
        std::string candidate = std::format("{}.{:03}", base, number);
        if (std::ranges::none_of(objects.getValues(), [&](const auto& object) { return object.name == candidate; }))
            return candidate;
    }
}

}

SceneController::SceneController(RmlKit::UI::Model& model, AppModel& app, SDL_Window* window, Session::Session& session, ViewportElementInstancer& viewports,
                                 InspectorController& inspector, MaterialController& materials, NodeGraphController& nodeGraph)
    : model(model), app(app), window(window), session(session), viewports(viewports), inspector(inspector), materials(materials), nodeGraph(nodeGraph),
      materialGrid(model, "materialGridViewport", "outliner", app.outliner.materialGrid, materialLayout) {
    model.bindEvent("sceneOpen", &SceneController::onOpen, this);
    model.bindEvent("sceneImport", &SceneController::onImport, this);
    model.bindEvent("environmentImport", &SceneController::onImportEnvironment, this);
    model.bindEvent("sceneSave", &SceneController::onSave, this);
    model.bindEvent("sceneNew", &SceneController::onNew, this);
    model.bindEvent("sceneAdd", &SceneController::onAdd, this);
    model.bindEvent("outlinerSelect", &SceneController::onSelect, this);
    model.bindEvent("outlinerFilter", &SceneController::onFilter, this);
    model.bindEvent("viewportFrame", &SceneController::onFrame, this);
    model.bindEvent("edit", &SceneController::onEdit, this);
    model.bindEvent("outlinerToggle", &SceneController::onToggle, this);
    model.bindEvent("outlinerDragMove", &SceneController::onDragMove, this);
    model.bindEvent("outlinerDragOut", &SceneController::onDragOut, this);
    model.bindEvent("outlinerDrop", &SceneController::onDrop, this);
    model.bindEvent("outlinerSlotDragEnd", &SceneController::onSlotDragEnd, this);
    model.bindEvent("outlinerClearMaterial", &SceneController::onClearMaterial, this);
    model.bindEvent("materialCreate", &SceneController::onCreateMaterial, this);
    model.bindEvent("materialSelect", &SceneController::onSelectMaterial, this);
    model.bindEvent("materialEdit", &SceneController::onEditMaterial, this);
    materials.renamed = [this] { showMaterials(); };
    viewports.getEvents().fileDrop = [this](std::filesystem::path path) { openFile(path, false); };
    viewports.getEvents().pick = [this](const Session::PickResult& picked) { select(picked); };
    viewports.getEvents().selection = [this] { return this->inspector.getSelection(); };
    viewports.getEvents().select = [this](const Selection& selected) {
        editList = EditList::Objects;
        select(selected);
        refresh();
    };
    viewports.getEvents().moved = [this] { this->inspector.showSelectedTransform(); };
    viewports.getEvents().tool = [this] {
        return this->app.tool == "scale" ? GizmoMode::Scale : this->app.tool == "rotate" ? GizmoMode::Rotate : GizmoMode::Translate;
    };
    refresh();
}

void SceneController::chooseFile(const bool replace) {
    RmlKit::UI::FileDialog::openFile(window, sceneFilters, scenePath.parent_path(), [this, replace](std::filesystem::path path) {
        if (!path.empty())
            openFile(path, replace);
    });
}

void SceneController::openFile(const std::filesystem::path& path, const bool replace) {
    if (!Import::SceneImporter::isSupported(path)) {
        model.report(ReportLevel::Warning, "scene", std::format("NoorRay cannot open the file type of {}.", path.filename().string()));
        return;
    }
    if (Import::EnvironmentImporter::isSupported(path)) {
        openEnvironment(path);
        return;
    }
    if (replace)
        session.clear();
    if (auto imported = session.importFile(path); !imported) {
        model.report(ReportLevel::Error, "scene", std::format("NoorRay cannot open {}: {}", path.filename().string(), imported.error().message));
        sceneChanged();
        return;
    }
    if (replace || scenePath.empty())
        scenePath = path;
    app.outliner.fileName = scenePath.filename().string();
    model.report(ReportLevel::Done, "scene", std::format("Opened {}.", path.filename().string()));
    //! An animated file brings its own cameras, so the view looks through the active one instead of framing the scene.
    viewports.followActiveCamera();
    if (session.getAnimation().isEmpty())
        viewports.frameScene();
    select(Selection{});
    sceneChanged();
    if (viewports.getEvents().opened)
        viewports.getEvents().opened();
}

//! An HDRI lights the current scene instead of replacing it, so it never clears the scene or moves the view.
void SceneController::openEnvironment(const std::filesystem::path& path) {
    if (auto imported = Import::EnvironmentImporter::import(path, session.getScene()); !imported) {
        model.report(ReportLevel::Error, "scene", std::format("NoorRay cannot open {}: {}", path.filename().string(), imported.error().message));
        return;
    }
    model.report(ReportLevel::Done, "scene", std::format("The environment is now {}.", path.filename().string()));
    sceneChanged();
}

void SceneController::onImportEnvironment(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    RmlKit::UI::Model::requireNoArguments(arguments, "environmentImport");
    RmlKit::UI::FileDialog::openFile(window, environmentFilters, scenePath.parent_path(), [this](std::filesystem::path path) {
        if (!path.empty())
            openEnvironment(path);
    });
}

void SceneController::onOpen(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    RmlKit::UI::Model::requireNoArguments(arguments, "sceneOpen");
    chooseFile(true);
}

void SceneController::onImport(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    RmlKit::UI::Model::requireNoArguments(arguments, "sceneImport");
    chooseFile(false);
}

void SceneController::onSave(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    RmlKit::UI::Model::requireNoArguments(arguments, "sceneSave");
    RmlKit::UI::FileDialog::openFolder(window, RmlKit::UI::FolderKind::Export, scenePath.parent_path(), [this](const std::filesystem::path& folder) {
        const std::string stem = scenePath.empty() ? std::string("scene") : scenePath.stem().string();
        const std::filesystem::path target = folder / (stem + ".nrs");
        if (auto saved = session.saveFile(target); !saved) {
            model.report(ReportLevel::Error, "scene", std::format("NoorRay cannot save {}: {}", target.string(), saved.error().message));
            return;
        }
        scenePath = target;
        app.outliner.fileName = target.filename().string();
        model.dirty("outliner");
        model.report(ReportLevel::Done, "scene", std::format("Saved {}.", target.string()));
    });
}

void SceneController::onNew(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    RmlKit::UI::Model::requireNoArguments(arguments, "sceneNew");
    session.clear();
    scenePath.clear();
    app.outliner.fileName.clear();
    select(Selection{});
    sceneChanged();
}

void SceneController::onAdd(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    if (arguments.size() != 1)
        RmlKit::UI::fatal("sceneAdd takes one argument: what to add");
    const Rml::String what = arguments[0].Get<Rml::String>();
    Scene::Scene& scene = session.getScene();
    const Core::Bounds bounds = scene.computeBounds();
    Selection added;
    if (what == "cube")
        added = Import::Primitives::add(scene, Import::Primitive::Cube);
    else if (what == "plane")
        added = Import::Primitives::add(scene, Import::Primitive::Plane);
    else if (what == "sphere")
        added = Import::Primitives::add(scene, Import::Primitive::Sphere);
    else if (what == "point")
        added = addLightNode(scene, makeLight(Scene::LightType::Point, bounds));
    else if (what == "spot")
        added = addLightNode(scene, makeLight(Scene::LightType::Spot, bounds));
    else if (what == "rect")
        added = addLightNode(scene, makeLight(Scene::LightType::Rect, bounds));
    else if (what == "directional")
        added = addLightNode(scene, makeLight(Scene::LightType::Directional, bounds));
    else if (what == "camera") {
        Scene::Camera camera;
        camera.name = std::format("Camera {}", scene.getCameras().size() + 1);
        camera.transform = viewports.getCamera().getTransform();
        camera.previousTransform = camera.transform;
        const Scene::CameraHandle handle = scene.addCamera(camera);
        scene.addNode({camera.name, camera.transform, {}, handle});
        added = handle;
    } else if (what == "empty") {
        editList = EditList::Objects;
        selectGroup(scene.addNode({"Empty", {}, {}, {}}));
        sceneChanged();
        return;
    } else
        RmlKit::UI::fatal(std::format("sceneAdd does not know {}", what));
    select(added);
    sceneChanged();
}

void SceneController::onSelect(Rml::DataModelHandle, Rml::Event& event, const Rml::VariantList& arguments) {
    const std::size_t index = RmlKit::UI::Model::indexArgument(arguments, "outlinerSelect", rowNodes.size());
    editList = EditList::Objects;
    selectedNodes.click(rowNodes, index, RmlKit::UI::listClickOf(event));
    showSelection();
    refresh();
}

void SceneController::onFilter(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    RmlKit::UI::Model::requireNoArguments(arguments, "outlinerFilter");
    refresh();
}

void SceneController::onFrame(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    RmlKit::UI::Model::requireNoArguments(arguments, "viewportFrame");
    viewports.frameScene();
}

//! The edit hotkeys and the Edit menu act on the list the user clicked last, so one key set works for objects and materials.
void SceneController::onEdit(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    const EditAction action = editActionOf(arguments, "edit");
    if (editList == EditList::Materials)
        editMaterials(action);
    else
        editObjects(action);
}

void SceneController::onEditMaterial(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    editList = EditList::Materials;
    editMaterials(editActionOf(arguments, "materialEdit"));
}

void SceneController::editObjects(const EditAction action) {
    switch (action) {
    case EditAction::Delete:
        deleteSelection();
        break;
    case EditAction::Cut:
        if (std::vector<ClipObject> copied = copySelection(); !copied.empty()) {
            clipboard = std::move(copied);
            deleteSelection();
        }
        break;
    case EditAction::Copy:
        if (std::vector<ClipObject> copied = copySelection(); !copied.empty())
            clipboard = std::move(copied);
        break;
    case EditAction::Paste:
        paste(clipboard);
        break;
    case EditAction::Duplicate:
        paste(copySelection());
        break;
    case EditAction::SelectAll:
        selectedNodes.selectAll(rowNodes);
        showSelection();
        refresh();
        break;
    }
}

void SceneController::editMaterials(const EditAction action) {
    const Core::SlotMap<Scene::Material>& all = session.getScene().getMaterials();
    const auto copy = [&] {
        std::vector<Scene::Material> copied;
        for (const Scene::MaterialHandle handle : selectedMaterials.items())
            if (const Scene::Material* material = all.find(handle))
                copied.push_back(*material);
        return copied;
    };
    switch (action) {
    case EditAction::Delete:
        deleteSelectedMaterials();
        break;
    case EditAction::Cut:
        if (std::vector<Scene::Material> copied = copy(); !copied.empty()) {
            copiedMaterials = std::move(copied);
            deleteSelectedMaterials();
        }
        break;
    case EditAction::Copy:
        if (std::vector<Scene::Material> copied = copy(); !copied.empty())
            copiedMaterials = std::move(copied);
        break;
    case EditAction::Paste:
        pasteMaterials(copiedMaterials);
        break;
    case EditAction::Duplicate:
        pasteMaterials(copy());
        break;
    case EditAction::SelectAll:
        selectedMaterials.selectAll(materialOrder());
        showMaterials();
        break;
    }
}

void SceneController::onToggle(Rml::DataModelHandle, Rml::Event& event, const Rml::VariantList& arguments) {
    const std::size_t index = RmlKit::UI::Model::indexArgument(arguments, "outlinerToggle", rowNodes.size());
    event.StopPropagation();
    if (const std::uint64_t key = nodeKey(rowNodes[index]); !collapsedNodes.erase(key))
        collapsedNodes.insert(key);
    refresh();
}

void SceneController::onDragMove(Rml::DataModelHandle, Rml::Event& event, const Rml::VariantList& arguments) {
    RmlKit::UI::Model::requireNoArguments(arguments, "outlinerDragMove");
    showDropTarget(findDropTarget(event));
}

void SceneController::onDragOut(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    RmlKit::UI::Model::requireNoArguments(arguments, "outlinerDragOut");
    showDropTarget({});
}

void SceneController::onDrop(Rml::DataModelHandle, Rml::Event& event, const Rml::VariantList& arguments) {
    RmlKit::UI::Model::requireNoArguments(arguments, "outlinerDrop");
    const DropTarget target = findDropTarget(event);
    showDropTarget({});
    if (target.place == DropPlace::None)
        return;
    const Rml::Element* dragged = draggedElement(event);
    const Scene::NodeHandle targetNode = target.row < 0 ? Scene::NodeHandle{} : rowNodes[static_cast<std::size_t>(target.row)];
    slotDropped = dragged->HasAttribute("slot-row");
    if (const int material = dragged->GetAttribute<int>("material-index", -1); material >= 0)
        dropMaterial(material, targetNode, target.slot);
    else
        moveNode(rowNodes[static_cast<std::size_t>(dragged->GetAttribute<int>("row", -1))], targetNode, target.place);
}

//! RmlUi sends the dragged element itself, not its clone, as drag_element.
SceneController::DropTarget SceneController::findDropTarget(const Rml::Event& event) const {
    const Rml::Element* dragged = draggedElement(event);
    if (!dragged)
        return {};
    Rml::Element* rowElement = rowElementOf(event.GetTargetElement());
    const int row = rowElement ? rowElement->GetAttribute<int>("row", -1) : -1;
    const Scene::Scene& scene = session.getScene();
    if (const int material = dragged->GetAttribute<int>("material-index", -1); material >= 0) {
        if (row < 0 || static_cast<std::size_t>(material) >= scene.getMaterials().size())
            return {};
        const Scene::Node* node = scene.find(rowNodes[static_cast<std::size_t>(row)]);
        if (!node || !std::holds_alternative<Scene::InstanceHandle>(node->object))
            return {};
        //! A material dragged out of a slot lands only on a slot, so a drop on a row name empties its slot.
        const int slot = slotOf(event.GetTargetElement());
        if (dragged->HasAttribute("slot-row") && slot < 0)
            return {};
        return {row, DropPlace::Into, slot};
    }
    const int draggedRow = dragged->GetAttribute<int>("row", -1);
    if (draggedRow < 0 || static_cast<std::size_t>(draggedRow) >= rowNodes.size())
        return {};
    if (row < 0)
        return {-1, DropPlace::End};
    const Scene::NodeHandle moved = rowNodes[static_cast<std::size_t>(draggedRow)];
    for (Scene::NodeHandle above = rowNodes[static_cast<std::size_t>(row)]; above; above = scene.find(above)->parent)
        if (above == moved)
            return {};
    const float height = rowElement->GetOffsetHeight();
    const float y = event.GetParameter<float>("mouse_y", 0.0f) - rowElement->GetAbsoluteOffset(Rml::BoxArea::Border).y;
    const DropPlace place = y < height * 0.25f ? DropPlace::Before : y > height * 0.75f ? DropPlace::After : DropPlace::Into;
    return {row, place};
}

void SceneController::showDropTarget(const DropTarget& target) {
    const int row = target.place == DropPlace::End ? -1 : target.row;
    const char* place = target.slot >= 0 ? "slot" : target.place == DropPlace::Before ? "before" : target.place == DropPlace::After ? "after" : "into";
    if (app.outliner.dropRow == row && app.outliner.dropSlot == target.slot && (row < 0 || app.outliner.dropPlace == place))
        return;
    app.outliner.dropRow = row;
    app.outliner.dropSlot = target.slot;
    app.outliner.dropPlace = place;
    model.dirty("outliner");
}

void SceneController::onSlotDragEnd(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    if (arguments.size() != 2)
        RmlKit::UI::fatal("outlinerSlotDragEnd takes two arguments: the row and the slot");
    const int row = arguments[0].Get<int>(-1), slot = arguments[1].Get<int>(-1);
    showDropTarget({});
    if (std::exchange(slotDropped, false) || row < 0 || static_cast<std::size_t>(row) >= rowNodes.size() || slot < 0)
        return;
    setSlotMaterial(rowNodes[static_cast<std::size_t>(row)], slot, {});
}

void SceneController::dropMaterial(const int material, const Scene::NodeHandle target, const int slot) {
    setSlotMaterial(target, slot, session.getScene().getMaterials().keyAt(static_cast<std::uint32_t>(material)));
}

//! Slots belong to the mesh, as in Blender, so a mesh shared by several objects changes in all of them; an object material is first spread over the slots. Slot
//! -1 is every slot, and an empty material empties the slot.
void SceneController::setSlotMaterial(const Scene::NodeHandle target, const int slot, const Scene::MaterialHandle material) {
    Scene::Scene& scene = session.getScene();
    const Scene::Node* node = scene.find(target);
    const auto* instance = node ? std::get_if<Scene::InstanceHandle>(&node->object) : nullptr;
    const Scene::Instance* object = instance ? scene.find(*instance) : nullptr;
    const Scene::Mesh* mesh = object ? scene.find(object->mesh) : nullptr;
    if (!mesh)
        return;
    const Scene::MaterialHandle spread = object->material;
    const Scene::MeshHandle meshHandle = object->mesh;
    const std::size_t slots = std::max<std::size_t>(mesh->subsets.size(), 1);
    scene.setInstanceMaterial(*instance, {});
    for (std::size_t index = 0; index < slots; ++index)
        if (slot < 0 || static_cast<std::size_t>(slot) == index)
            scene.setMeshSubsetMaterial(meshHandle, index, material);
        else if (spread)
            scene.setMeshSubsetMaterial(meshHandle, index, spread);
    select(Selection{*instance});
    sceneChanged();
}

void SceneController::moveNode(const Scene::NodeHandle moved, const Scene::NodeHandle target, const DropPlace place) {
    Scene::Scene& scene = session.getScene();
    if (moved == target)
        return;
    if (place == DropPlace::End)
        scene.setParent(moved, {}, std::numeric_limits<std::size_t>::max());
    else if (place == DropPlace::Into) {
        collapsedNodes.erase(nodeKey(target));
        scene.setParent(moved, target, std::numeric_limits<std::size_t>::max());
    } else {
        const Scene::NodeHandle parent = scene.find(target)->parent;
        const std::span<const Scene::NodeHandle> siblings = parent ? std::span<const Scene::NodeHandle>(scene.find(parent)->children) : scene.getRootNodes();
        std::size_t index = place == DropPlace::After ? 1 : 0;
        for (const Scene::NodeHandle sibling : siblings) {
            if (sibling == target)
                break;
            if (sibling != moved)
                ++index;
        }
        scene.setParent(moved, parent, index);
    }
    sceneChanged();
}

void SceneController::onClearMaterial(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    RmlKit::UI::Model::requireNoArguments(arguments, "outlinerClearMaterial");
    const auto* instance = std::get_if<Scene::InstanceHandle>(&inspector.getSelection());
    if (!instance)
        return;
    session.getScene().setInstanceMaterial(*instance, {});
    select(Selection{*instance});
    sceneChanged();
}

//! The argument is true for a node material, whose node tree is the one Blender gives a new material.
void SceneController::onCreateMaterial(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    if (arguments.size() != 1)
        RmlKit::UI::fatal("materialCreate takes one argument: whether the material uses nodes");
    const bool nodes = arguments[0].Get<bool>(false);
    Scene::Scene& scene = session.getScene();
    Scene::Material material;
    material.name = std::ranges::none_of(scene.getMaterials().getValues(), [](const Scene::Material& other) { return other.name == "Material"; })
                        ? std::string("Material")
                        : copyName("Material", scene.getMaterials());
    if (nodes) {
        auto tree = std::make_shared<Materials::NodeTree>();
        Materials::NodeCatalog::addDefaultNodes(*tree, Materials::NodeTreeKind::Material);
        material.nodeTree = std::move(tree);
    }
    const Scene::MaterialHandle added = addMaterial(std::move(material));
    if (nodes)
        if (auto compiled = Materials::NodeCompiler::compileMaterial(scene, added, false); !compiled)
            model.report(ReportLevel::Error, "material", std::format("NoorRay cannot compile the new node material: {}", compiled.error().message));
    editList = EditList::Materials;
    selectedMaterials.selectOnly(added);
    materials.edit(added);
    sceneChanged();
    if (nodes && nodeMaterialCreated)
        nodeMaterialCreated(added);
}

//! A pasted material keeps its name when the scene no longer has it, as after a cut, and gets a numbered name otherwise.
Scene::MaterialHandle SceneController::addMaterial(Scene::Material material) {
    Scene::Scene& scene = session.getScene();
    const Core::SlotMap<Scene::Material>& all = scene.getMaterials();
    if (std::ranges::any_of(all.getValues(), [&](const Scene::Material& other) { return other.name == material.name; }))
        material.name = copyName(material.name, all);
    return scene.addMaterial(std::move(material));
}

std::vector<Scene::MaterialHandle> SceneController::materialOrder() const {
    const Core::SlotMap<Scene::Material>& all = session.getScene().getMaterials();
    std::vector<Scene::MaterialHandle> order;
    order.reserve(all.size());
    for (std::uint32_t index = 0; index < all.size(); ++index)
        order.push_back(all.keyAt(index));
    return order;
}

void SceneController::pasteMaterials(const std::span<const Scene::Material> copied) {
    if (copied.empty())
        return;
    std::vector<Scene::MaterialHandle> added;
    for (const Scene::Material& material : copied)
        added.push_back(addMaterial(material));
    selectedMaterials.set(std::move(added));
    sceneChanged();
}

void SceneController::onSelectMaterial(Rml::DataModelHandle, Rml::Event& event, const Rml::VariantList& arguments) {
    const std::vector<Scene::MaterialHandle> order = materialOrder();
    const std::size_t index = RmlKit::UI::Model::indexArgument(arguments, "materialSelect", order.size());
    editList = EditList::Materials;
    selectedMaterials.click(order, index, RmlKit::UI::listClickOf(event));
    if (const std::optional<Scene::MaterialHandle> primary = selectedMaterials.primary())
        nodeGraph.showMaterial(*primary);
    showMaterials();
}

//! The slots that used the materials become empty; the inspector is shown again, since its material links name dense indices.
void SceneController::deleteSelectedMaterials() {
    if (selectedMaterials.empty())
        return;
    Scene::Scene& scene = session.getScene();
    for (const Scene::MaterialHandle handle : selectedMaterials.items())
        scene.remove(handle);
    selectedMaterials.clear();
    inspector.show(inspector.getSelection());
    sceneChanged();
}

//! Groups hold no object, so they are not copied.
std::vector<SceneController::ClipObject> SceneController::copySelection() const {
    const Scene::Scene& scene = session.getScene();
    std::vector<ClipObject> copied;
    for (const Scene::NodeHandle handle : selectedNodes.items())
        if (const Scene::Node* node = scene.find(handle))
            std::visit(
                [&](const auto& object) {
                    if constexpr (!std::is_same_v<std::decay_t<decltype(object)>, std::monostate>)
                        if (const auto* value = scene.find(object))
                            copied.emplace_back(*value);
                },
                node->object);
    return copied;
}

//! The copies land where the originals are, selected, so the gizmo moves them off the originals.
void SceneController::paste(const std::span<const ClipObject> copied) {
    Scene::Scene& scene = session.getScene();
    std::vector<Scene::NodeHandle> added;
    for (const ClipObject& object : copied)
        if (const Selection handle = pasteObject(object); !std::holds_alternative<std::monostate>(handle))
            added.push_back(scene.findNode(nodeObjectOf(handle)));
    if (added.empty())
        return;
    editList = EditList::Objects;
    selectedNodes.set(std::move(added));
    showSelection();
    sceneChanged();
}

Selection SceneController::pasteObject(const ClipObject& copied) {
    Scene::Scene& scene = session.getScene();
    return std::visit(
        [&](const auto& object) -> Selection {
            using Object = std::decay_t<decltype(object)>;
            if constexpr (std::is_same_v<Object, Scene::Instance>) {
                const Scene::Mesh* mesh = scene.find(object.mesh);
                if (!mesh)
                    return {};
                const Scene::InstanceHandle handle = scene.addInstance(object);
                scene.addNode({mesh->name, object.transform, {}, handle});
                return handle;
            } else {
                Object copy = object;
                if constexpr (std::is_same_v<Object, Scene::Light>)
                    copy.name = copyName(object.name, scene.getLights());
                else if constexpr (std::is_same_v<Object, Scene::Camera>)
                    copy.name = copyName(object.name, scene.getCameras());
                else if constexpr (std::is_same_v<Object, Scene::GaussianCloud>)
                    copy.name = copyName(object.name, scene.getGaussianClouds());
                else
                    copy.name = copyName(object.name, scene.getVolumes());
                const Core::Transform transform = copy.transform;
                std::string name = copy.name;
                Selection handle;
                if constexpr (std::is_same_v<Object, Scene::Light>)
                    handle = scene.addLight(std::move(copy));
                else if constexpr (std::is_same_v<Object, Scene::Camera>)
                    handle = scene.addCamera(std::move(copy));
                else if constexpr (std::is_same_v<Object, Scene::GaussianCloud>)
                    handle = scene.addGaussianCloud(std::move(copy));
                else
                    handle = scene.addVolume(std::move(copy));
                scene.addNode({std::move(name), transform, {}, nodeObjectOf(handle)});
                return handle;
            }
        },
        copied);
}

//! Removing a node removes its children, so a selected child of a removed node is already gone.
void SceneController::deleteSelection() {
    Scene::Scene& scene = session.getScene();
    for (const Scene::NodeHandle handle : selectedNodes.items())
        scene.remove(handle);
    select(Selection{});
    sceneChanged();
}

void SceneController::select(const Session::PickResult& picked) {
    editList = EditList::Objects;
    if (const auto* instance = std::get_if<Scene::InstanceHandle>(&picked))
        select(Selection{*instance});
    else if (const auto* cloud = std::get_if<Scene::GaussianCloudHandle>(&picked))
        select(Selection{*cloud});
    else
        select(Selection{});
    refresh();
}

void SceneController::select(const Selection& selected) {
    selectedNodes.clear();
    if (!std::holds_alternative<std::monostate>(selected))
        if (const Scene::NodeHandle node = nodeOf(session.getScene(), selected))
            selectedNodes.selectOnly(node);
    showSelection();
}

void SceneController::selectGroup(const Scene::NodeHandle group) {
    selectedNodes.selectOnly(group);
    showSelection();
}

//! The inspector shows the primary node; a node without an object is an empty, shown by its node.
void SceneController::showSelection() {
    const Scene::Scene& scene = session.getScene();
    const std::optional<Scene::NodeHandle> primary = selectedNodes.primary();
    const Scene::Node* node = primary ? scene.find(*primary) : nullptr;
    Selection shown = node ? selectionOf(node->object) : Selection{};
    if (node && std::holds_alternative<std::monostate>(shown))
        shown = *primary;
    inspector.show(shown);
    nodeGraph.showSelection(shown);
    for (const Scene::NodeHandle handle : selectedNodes.items())
        expandTo(handle);
    if (!std::holds_alternative<std::monostate>(shown))
        app.inspector.tab = "object";
}

void SceneController::expandTo(const Scene::NodeHandle handle) {
    const Scene::Scene& scene = session.getScene();
    for (const Scene::Node* node = scene.find(handle); node && node->parent; node = scene.find(node->parent))
        collapsedNodes.erase(nodeKey(node->parent));
}

void SceneController::sceneChanged() {
    viewports.invalidate();
    inspector.showEnvironment();
    materials.refresh();
    nodeGraph.refresh();
    refresh();
}

bool SceneController::addRows(const Scene::NodeHandle handle, const int depth) {
    const Scene::Scene& scene = session.getScene();
    const Scene::Node* node = scene.find(handle);
    if (!node)
        return false;
    std::vector<OutlinerRow>& rows = app.outliner.rows;
    const std::size_t mark = rows.size();
    const bool filtering = !app.outliner.filter.empty();
    OutlinerRow row = describe(scene, *node);
    row.depth = depth;
    row.parent = !node->children.empty();
    row.expanded = filtering || !collapsedNodes.contains(nodeKey(handle));
    row.selected = selectedNodes.contains(handle);
    rows.push_back(std::move(row));
    rowNodes.push_back(handle);
    bool matches = contains(node->name, app.outliner.filter);
    if (rows[mark].expanded)
        for (const Scene::NodeHandle child : node->children)
            if (addRows(child, depth + 1))
                matches = true;
    if (matches)
        return true;
    rows.resize(mark);
    rowNodes.resize(mark);
    return false;
}

void SceneController::refresh() {
    const Scene::Scene& scene = session.getScene();
    selectedNodes.keepIf([&](const Scene::NodeHandle handle) { return scene.find(handle) != nullptr; });
    app.outliner.rows.clear();
    rowNodes.clear();
    for (const Scene::NodeHandle root : scene.getRootNodes())
        addRows(root, 0);
    showMaterials();
}

void SceneController::showMaterials() {
    const Core::SlotMap<Scene::Material>& all = session.getScene().getMaterials();
    selectedMaterials.keepIf([&](const Scene::MaterialHandle handle) { return all.contains(handle); });
    app.outliner.selectedMaterials = static_cast<int>(selectedMaterials.size());
    RmlKit::UI::GridSection<MaterialCard> section;
    for (std::uint32_t index = 0; index < all.size(); ++index)
        section.items.push_back({.index = static_cast<int>(index),
                                 .name = all.getValues()[index].name,
                                 .selected = selectedMaterials.contains(all.keyAt(index)),
                                 .nodes = all.getValues()[index].nodeTree != nullptr});
    materialGrid.setSections({std::move(section)});
    model.dirty("outliner");
}
}
