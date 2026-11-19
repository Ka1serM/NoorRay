#include "NoorRay.App/UI/Controllers/SceneController.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <format>
#include <limits>
#include <string_view>
#include <type_traits>
#include <variant>

#include <RmlUi/Core/Element.h>

#include "NoorRay.App/UI/Controllers/InspectorController.h"
#include "NoorRay.App/UI/Controllers/MaterialController.h"
#include "NoorRay.App/UI/Controls/ViewportElement.h"
#include "NoorRay.App/UI/Models/AppModel.h"
#include "NoorRay/Import/EnvironmentImporter.h"
#include "NoorRay/Import/Primitives.h"
#include "NoorRay/Import/SceneImporter.h"
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

Scene::NodeObject nodeObjectOf(const Selection& selection) {
    return std::visit([](const auto& handle) -> Scene::NodeObject { return handle; }, selection);
}

Selection selectionOf(const Scene::NodeObject& object) {
    return std::visit([](const auto& handle) -> Selection { return handle; }, object);
}

std::uint64_t nodeKey(const Scene::NodeHandle handle) {
    return std::uint64_t{handle.generation} << 32 | handle.index;
}

OutlinerRow describe(const Scene::Scene& scene, const Scene::Node& node) {
    OutlinerRow row{.name = node.name, .kind = "Group", .icon = "coord-system"};
    if (const auto* instance = std::get_if<Scene::InstanceHandle>(&node.object)) {
        row.kind = "Mesh";
        row.icon = "shape-cube";
        const Scene::Instance* object = scene.find(*instance);
        const std::uint32_t material = object ? scene.getMaterials().denseIndex(object->material) : Scene::MaterialHandle::invalidIndex;
        row.material = material == Scene::MaterialHandle::invalidIndex ? -1 : static_cast<int>(material);
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
                                 InspectorController& inspector, MaterialController& materials)
    : model(model), app(app), window(window), session(session), viewports(viewports), inspector(inspector), materials(materials) {
    model.bindEvent("sceneOpen", &SceneController::onOpen, this);
    model.bindEvent("sceneImport", &SceneController::onImport, this);
    model.bindEvent("environmentImport", &SceneController::onImportEnvironment, this);
    model.bindEvent("sceneSave", &SceneController::onSave, this);
    model.bindEvent("sceneNew", &SceneController::onNew, this);
    model.bindEvent("sceneAdd", &SceneController::onAdd, this);
    model.bindEvent("outlinerSelect", &SceneController::onSelect, this);
    model.bindEvent("outlinerRemove", &SceneController::onRemove, this);
    model.bindEvent("outlinerFilter", &SceneController::onFilter, this);
    model.bindEvent("viewportFrame", &SceneController::onFrame, this);
    model.bindEvent("sceneCopy", &SceneController::onCopy, this);
    model.bindEvent("scenePaste", &SceneController::onPaste, this);
    model.bindEvent("sceneDuplicate", &SceneController::onDuplicate, this);
    model.bindEvent("outlinerToggle", &SceneController::onToggle, this);
    model.bindEvent("outlinerDragMove", &SceneController::onDragMove, this);
    model.bindEvent("outlinerDragOut", &SceneController::onDragOut, this);
    model.bindEvent("outlinerDrop", &SceneController::onDrop, this);
    model.bindEvent("outlinerClearMaterial", &SceneController::onClearMaterial, this);
    model.bindEvent("materialCreate", &SceneController::onCreateMaterial, this);
    viewports.getEvents().fileDrop = [this](std::filesystem::path path) { openFile(path, false); };
    viewports.getEvents().pick = [this](const Session::PickResult& picked) { select(picked); };
    viewports.getEvents().selection = [this] { return this->inspector.getSelection(); };
    viewports.getEvents().select = [this](const Selection& selected) {
        select(selected);
        refresh();
    };
    viewports.getEvents().moved = [this] { this->inspector.showSelectedTransform(); };
    viewports.getEvents().tool = [this] {
        return this->app.tool == "scale" ? GizmoMode::Scale : this->app.tool == "rotate" ? GizmoMode::Rotate : GizmoMode::Move;
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
    viewports.followActiveCamera();
    viewports.frameScene();
    select(Selection{});
    sceneChanged();
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
    } else if (what == "null") {
        selectGroup(scene.addNode({"Null", {}, {}, {}}));
        sceneChanged();
        return;
    } else
        RmlKit::UI::fatal(std::format("sceneAdd does not know {}", what));
    select(added);
    sceneChanged();
}

void SceneController::onSelect(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    const std::size_t index = RmlKit::UI::Model::indexArgument(arguments, "outlinerSelect", rowNodes.size());
    const Scene::Node* node = session.getScene().find(rowNodes[index]);
    if (!node)
        return;
    if (std::holds_alternative<std::monostate>(node->object))
        selectGroup(rowNodes[index]);
    else
        select(selectionOf(node->object));
    refresh();
}

void SceneController::onRemove(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    RmlKit::UI::Model::requireNoArguments(arguments, "outlinerRemove");
    Scene::Scene& scene = session.getScene();
    scene.remove(selectedGroup ? selectedGroup : scene.findNode(nodeObjectOf(inspector.getSelection())));
    select(Selection{});
    sceneChanged();
}

void SceneController::onFilter(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    RmlKit::UI::Model::requireNoArguments(arguments, "outlinerFilter");
    refresh();
}

void SceneController::onFrame(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    RmlKit::UI::Model::requireNoArguments(arguments, "viewportFrame");
    viewports.frameScene();
}

void SceneController::onCopy(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    RmlKit::UI::Model::requireNoArguments(arguments, "sceneCopy");
    if (Clipboard copied = copySelection(); !std::holds_alternative<std::monostate>(copied))
        clipboard = std::move(copied);
}

void SceneController::onPaste(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    RmlKit::UI::Model::requireNoArguments(arguments, "scenePaste");
    paste(clipboard);
}

void SceneController::onDuplicate(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    RmlKit::UI::Model::requireNoArguments(arguments, "sceneDuplicate");
    paste(copySelection());
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
    if (const int material = dragged->GetAttribute<int>("material-index", -1); material >= 0)
        dropMaterial(material, targetNode);
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
        return node && std::holds_alternative<Scene::InstanceHandle>(node->object) ? DropTarget{row, DropPlace::Into} : DropTarget{};
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
    const char* place = target.place == DropPlace::Before ? "before" : target.place == DropPlace::After ? "after" : "into";
    if (app.outliner.dropRow == row && (row < 0 || app.outliner.dropPlace == place))
        return;
    app.outliner.dropRow = row;
    app.outliner.dropPlace = place;
    model.dirty("outliner");
}

void SceneController::dropMaterial(const int material, const Scene::NodeHandle target) {
    Scene::Scene& scene = session.getScene();
    const Scene::Node* node = scene.find(target);
    const auto* instance = node ? std::get_if<Scene::InstanceHandle>(&node->object) : nullptr;
    if (!instance)
        return;
    scene.setInstanceMaterial(*instance, scene.getMaterials().keyAt(static_cast<std::uint32_t>(material)));
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

void SceneController::onCreateMaterial(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    RmlKit::UI::Model::requireNoArguments(arguments, "materialCreate");
    Scene::Scene& scene = session.getScene();
    Scene::Material material;
    material.name = std::ranges::none_of(scene.getMaterials().getValues(), [](const Scene::Material& other) { return other.name == "Material"; })
                        ? std::string("Material")
                        : copyName("Material", scene.getMaterials());
    materials.edit(scene.addMaterial(std::move(material)));
    sceneChanged();
}

SceneController::Clipboard SceneController::copySelection() const {
    const Scene::Scene& scene = session.getScene();
    return std::visit(
        [&](const auto& handle) -> Clipboard {
            if constexpr (std::is_same_v<std::decay_t<decltype(handle)>, std::monostate>)
                return {};
            else if (const auto* object = scene.find(handle))
                return *object;
            else
                return {};
        },
        inspector.getSelection());
}

//! The copy lands where the original is, selected, so the gizmo moves it off the original.
void SceneController::paste(const Clipboard& copied) {
    Scene::Scene& scene = session.getScene();
    const Selection added = std::visit(
        [&](const auto& object) -> Selection {
            using Object = std::decay_t<decltype(object)>;
            if constexpr (std::is_same_v<Object, std::monostate>)
                return {};
            else if constexpr (std::is_same_v<Object, Scene::Instance>) {
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
    if (std::holds_alternative<std::monostate>(added))
        return;
    select(added);
    sceneChanged();
}

void SceneController::select(const Session::PickResult& picked) {
    if (const auto* instance = std::get_if<Scene::InstanceHandle>(&picked))
        select(Selection{*instance});
    else if (const auto* cloud = std::get_if<Scene::GaussianCloudHandle>(&picked))
        select(Selection{*cloud});
    else
        select(Selection{});
    refresh();
}

void SceneController::select(const Selection& selected) {
    selectedGroup = {};
    inspector.show(selected);
    if (!std::holds_alternative<std::monostate>(selected))
        expandTo(session.getScene().findNode(nodeObjectOf(selected)));
    if (!std::holds_alternative<std::monostate>(selected))
        app.inspector.tab = "object";
}

void SceneController::selectGroup(const Scene::NodeHandle group) {
    select(Selection{});
    selectedGroup = group;
    expandTo(group);
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
    const Selection selection = selectionOf(node->object);
    row.depth = depth;
    row.parent = !node->children.empty();
    row.expanded = filtering || !collapsedNodes.contains(nodeKey(handle));
    row.selected = handle == selectedGroup || (!std::holds_alternative<std::monostate>(selection) && selection == inspector.getSelection());
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
    if (selectedGroup && !scene.find(selectedGroup))
        selectedGroup = {};
    app.outliner.rows.clear();
    rowNodes.clear();
    for (const Scene::NodeHandle root : scene.getRootNodes())
        addRows(root, 0);
    app.outliner.materials.clear();
    for (const Scene::Material& material : scene.getMaterials().getValues())
        app.outliner.materials.push_back(material.name);
    model.dirty("outliner");
}

}
