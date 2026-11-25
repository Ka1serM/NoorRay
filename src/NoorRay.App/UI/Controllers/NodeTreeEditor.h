#pragma once

#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "NoorRay/Core/Error.h"
#include "NoorRay/Materials/NodeCatalog.h"
#include "NoorRay/Materials/NodeTree.h"
#include "NoorRay/Session/Session.h"

namespace NoorRay::App {

class ViewportElementInstancer;

struct NodeTreeWorld {
    bool operator==(const NodeTreeWorld&) const = default;
};

using NodeTreeOwner = std::variant<std::monostate, Scene::MaterialHandle, Scene::LightHandle, NodeTreeWorld>;

//! Layout edits change only how the tree looks, so the owner does not compile again.
enum class NodeEdit : std::uint8_t { Layout, Result };

//! The node tree of a material, a light or the world that the node editor edits: the shown group, the undo history and the compile of the owner.
class NodeTreeEditor {
public:
    using Change = std::function<std::expected<void, Core::Error>(Materials::NodeTree&)>;

    NodeTreeEditor(Session::Session& session, ViewportElementInstancer& viewports);
    NodeTreeEditor(const NodeTreeEditor&) = delete;
    NodeTreeEditor& operator=(const NodeTreeEditor&) = delete;

    void show(NodeTreeOwner shown);
    //! The scene changed: forgets the owner when the scene no longer has it, and the history when another edit replaced the tree.
    void refresh();
    const NodeTreeOwner& getOwner() const { return owner; }
    bool hasOwner() const { return !std::holds_alternative<std::monostate>(owner); }
    std::string getOwnerName() const;
    Materials::NodeTreeKind getKind() const;
    std::shared_ptr<const Materials::NodeTree> findRoot() const;
    //! The tree of the entered group, or the root.
    const Materials::NodeTree* findShown() const;
    std::span<const std::uint32_t> getPath() const { return path; }
    void enter(std::uint32_t group);
    void leave(std::size_t depth);

    //! Changes a copy of the shown tree and stores it in the owner. Edits with the same merge key in a row keep one undo step, as a scrub does.
    std::expected<void, Core::Error> edit(const Change& change, NodeEdit effect, std::string_view merge = {});
    //! Gives an owner without a tree a tree, as Blender's Use Nodes does: a material gets nodes that keep its look, a light or the world the default tree.
    std::expected<void, Core::Error> useNodes();
    bool undo();
    bool redo();
    bool canUndo() const { return !undoSteps.empty(); }
    bool canRedo() const { return !redoSteps.empty(); }
    //! Runs once a frame and compiles the owner when an edit asked for it.
    bool tick();

    const std::vector<std::string>& getWarnings() const { return warnings; }
    const std::string& getError() const { return error; }

    //! Called after the tree or the shown group changed; newView is true when another tree is shown.
    std::function<void(bool newView)> changed;

private:
    void store(std::shared_ptr<const Materials::NodeTree> root);
    void requestCompile();
    void compile();
    void restore(std::vector<std::shared_ptr<const Materials::NodeTree>>& from, std::vector<std::shared_ptr<const Materials::NodeTree>>& to);
    void notify(bool newView);

    Session::Session& session;
    ViewportElementInstancer& viewports;
    NodeTreeOwner owner;
    //! The group indices from the owner's tree to the shown tree.
    std::vector<std::uint32_t> path;
    std::shared_ptr<const Materials::NodeTree> stored;
    std::vector<std::shared_ptr<const Materials::NodeTree>> undoSteps;
    std::vector<std::shared_ptr<const Materials::NodeTree>> redoSteps;
    std::string lastMerge;
    bool compilePending = false;
    bool compiledThisFrame = false;
    std::vector<std::string> warnings;
    std::string error;
};

}
