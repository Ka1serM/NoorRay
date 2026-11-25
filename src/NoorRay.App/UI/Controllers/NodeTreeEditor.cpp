#include "NoorRay.App/UI/Controllers/NodeTreeEditor.h"

#include <utility>

#include "NoorRay.App/UI/Controls/ViewportElement.h"
#include "NoorRay/Materials/NodeCompiler.h"

namespace NoorRay::App {

namespace {

using Materials::NodeTree;

constexpr std::size_t maxUndoSteps = 128;

std::expected<std::shared_ptr<const NodeTree>, Core::Error> changedCopy(const NodeTree& tree, const std::span<const std::uint32_t> path,
                                                                        const NodeTreeEditor::Change& change) {
    auto copy = std::make_shared<NodeTree>(tree);
    if (path.empty()) {
        if (auto changed = change(*copy); !changed)
            return std::unexpected(changed.error());
        return copy;
    }
    auto group = changedCopy(*tree.groups[path.front()], path.subspan(1), change);
    if (!group)
        return group;
    copy->groups[path.front()] = std::move(*group);
    return copy;
}

}

NodeTreeEditor::NodeTreeEditor(Session::Session& session, ViewportElementInstancer& viewports) : session(session), viewports(viewports) {}

void NodeTreeEditor::show(NodeTreeOwner shown) {
    if (shown == owner)
        return;
    if (compilePending)
        compile();
    owner = shown;
    path.clear();
    undoSteps.clear();
    redoSteps.clear();
    lastMerge.clear();
    warnings.clear();
    error.clear();
    stored = findRoot();
    notify(true);
}

void NodeTreeEditor::refresh() {
    const Scene::Scene& scene = session.getScene();
    const auto* material = std::get_if<Scene::MaterialHandle>(&owner);
    const auto* light = std::get_if<Scene::LightHandle>(&owner);
    if ((material && !scene.find(*material)) || (light && !scene.find(*light)))
        return show(std::monostate{});
    const std::shared_ptr<const NodeTree> root = findRoot();
    if (root == stored)
        return notify(false);
    stored = root;
    undoSteps.clear();
    redoSteps.clear();
    lastMerge.clear();
    notify(findShown() == nullptr);
}

std::string NodeTreeEditor::getOwnerName() const {
    const Scene::Scene& scene = session.getScene();
    if (const auto* material = std::get_if<Scene::MaterialHandle>(&owner); material && scene.find(*material))
        return scene.find(*material)->name;
    if (const auto* light = std::get_if<Scene::LightHandle>(&owner); light && scene.find(*light))
        return scene.find(*light)->name;
    if (std::holds_alternative<NodeTreeWorld>(owner))
        return "World";
    return {};
}

Materials::NodeTreeKind NodeTreeEditor::getKind() const {
    if (std::holds_alternative<Scene::LightHandle>(owner))
        return Materials::NodeTreeKind::Light;
    if (std::holds_alternative<NodeTreeWorld>(owner))
        return Materials::NodeTreeKind::World;
    return Materials::NodeTreeKind::Material;
}

std::shared_ptr<const NodeTree> NodeTreeEditor::findRoot() const {
    const Scene::Scene& scene = session.getScene();
    if (const auto* material = std::get_if<Scene::MaterialHandle>(&owner)) {
        const Scene::Material* found = scene.find(*material);
        return found ? found->nodeTree : nullptr;
    }
    if (const auto* light = std::get_if<Scene::LightHandle>(&owner)) {
        const Scene::Light* found = scene.find(*light);
        return found ? found->nodeTree : nullptr;
    }
    if (std::holds_alternative<NodeTreeWorld>(owner))
        return scene.getEnvironment().nodeTree;
    return nullptr;
}

const NodeTree* NodeTreeEditor::findShown() const {
    const NodeTree* tree = stored.get();
    for (const std::uint32_t group : path) {
        if (!tree || group >= tree->groups.size())
            return nullptr;
        tree = tree->groups[group].get();
    }
    return tree;
}

void NodeTreeEditor::enter(const std::uint32_t group) {
    const NodeTree* tree = findShown();
    if (!tree || group >= tree->groups.size() || !tree->groups[group])
        return;
    path.push_back(group);
    notify(true);
}

void NodeTreeEditor::leave(const std::size_t depth) {
    if (depth >= path.size())
        return;
    path.resize(depth);
    notify(true);
}

std::expected<void, Core::Error> NodeTreeEditor::edit(const Change& change, const NodeEdit effect, const std::string_view merge) {
    const NodeTree* shown = findShown();
    if (!stored || !shown)
        return Core::fail(Core::ErrorCode::InvalidState, "the node editor shows no node tree");
    auto root = changedCopy(*stored, path, change);
    if (!root)
        return std::unexpected(root.error());
    if (merge.empty() || merge != lastMerge) {
        undoSteps.push_back(stored);
        if (undoSteps.size() > maxUndoSteps)
            undoSteps.erase(undoSteps.begin());
    }
    lastMerge = merge;
    redoSteps.clear();
    store(std::move(*root));
    if (effect == NodeEdit::Result)
        requestCompile();
    notify(false);
    return {};
}

std::expected<void, Core::Error> NodeTreeEditor::useNodes() {
    if (!hasOwner())
        return Core::fail(Core::ErrorCode::InvalidState, "the node editor shows no owner");
    if (stored)
        return {};
    auto tree = std::make_shared<NodeTree>();
    const auto* material = std::get_if<Scene::MaterialHandle>(&owner);
    if (const Scene::Material* found = material ? session.getScene().find(*material) : nullptr)
        Materials::NodeCatalog::addMaterialNodes(*tree, *found);
    else
        Materials::NodeCatalog::addDefaultNodes(*tree, getKind());
    path.clear();
    undoSteps.clear();
    redoSteps.clear();
    lastMerge.clear();
    store(std::move(tree));
    requestCompile();
    notify(true);
    return {};
}

bool NodeTreeEditor::undo() {
    if (undoSteps.empty())
        return false;
    restore(undoSteps, redoSteps);
    return true;
}

bool NodeTreeEditor::redo() {
    if (redoSteps.empty())
        return false;
    restore(redoSteps, undoSteps);
    return true;
}

void NodeTreeEditor::restore(std::vector<std::shared_ptr<const NodeTree>>& from, std::vector<std::shared_ptr<const NodeTree>>& to) {
    to.push_back(stored);
    std::shared_ptr<const NodeTree> root = std::move(from.back());
    from.pop_back();
    lastMerge.clear();
    store(std::move(root));
    requestCompile();
    notify(findShown() == nullptr);
}

void NodeTreeEditor::store(std::shared_ptr<const NodeTree> root) {
    Scene::Scene& scene = session.getScene();
    stored = root;
    if (const auto* material = std::get_if<Scene::MaterialHandle>(&owner)) {
        const Scene::Material* current = scene.find(*material);
        if (!current)
            return;
        Scene::Material changed = *current;
        changed.nodeTree = std::move(root);
        scene.setMaterial(*material, changed);
    } else if (const auto* light = std::get_if<Scene::LightHandle>(&owner)) {
        const Scene::Light* current = scene.find(*light);
        if (!current)
            return;
        Scene::Light changed = *current;
        changed.nodeTree = std::move(root);
        scene.setLight(*light, changed);
    } else if (std::holds_alternative<NodeTreeWorld>(owner)) {
        Scene::Environment changed = scene.getEnvironment();
        changed.nodeTree = std::move(root);
        scene.setEnvironment(changed);
    }
    viewports.invalidate();
}

void NodeTreeEditor::requestCompile() {
    if (compiledThisFrame) {
        compilePending = true;
        viewports.invalidate();
        return;
    }
    compile();
    compiledThisFrame = true;
}

bool NodeTreeEditor::tick() {
    compiledThisFrame = false;
    if (!compilePending)
        return false;
    compile();
    compiledThisFrame = true;
    notify(false);
    return false;
}

void NodeTreeEditor::compile() {
    compilePending = false;
    Scene::Scene& scene = session.getScene();
    std::expected<std::vector<std::string>, Core::Error> compiled;
    if (const auto* material = std::get_if<Scene::MaterialHandle>(&owner))
        compiled = Materials::NodeCompiler::compileMaterial(scene, *material, false);
    else if (const auto* light = std::get_if<Scene::LightHandle>(&owner))
        compiled = Materials::NodeCompiler::compileLight(scene, *light, false);
    else if (std::holds_alternative<NodeTreeWorld>(owner))
        compiled = Materials::NodeCompiler::compileEnvironment(scene, false);
    else
        return;
    viewports.invalidate();
    if (!compiled) {
        error = compiled.error().message;
        warnings.clear();
        return;
    }
    error.clear();
    warnings = std::move(*compiled);
}

void NodeTreeEditor::notify(const bool newView) {
    if (changed)
        changed(newView);
}

}
