#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <RmlUi/Core/DataModelHandle.h>
#include <RmlUi/Core/Event.h>
#include <RmlUi/Core/Types.h>

#include "NoorRay.App/UI/Controllers/NodePanelController.h"
#include "NoorRay.App/UI/Controllers/NodeTreeEditor.h"
#include "NoorRay.App/UI/Controllers/Selection.h"
#include "NoorRay.App/UI/Controls/NodeGraphElement.h"
#include "NoorRay/Materials/NodeTree.h"
#include "NoorRay/Session/Session.h"

namespace RmlKit::UI {
class Model;
}

namespace NoorRay::App {

class AppModel;
class ViewportElementInstancer;

//! The node editor: shows the node tree of a material, a light or the world in the node graph, and edits it through the node catalog.
class NodeGraphController {
public:
    NodeGraphController(RmlKit::UI::Model& model, AppModel& app, Session::Session& session, ViewportElementInstancer& viewports, NodeGraphInstancer& graphs);
    NodeGraphController(const NodeGraphController&) = delete;
    NodeGraphController& operator=(const NodeGraphController&) = delete;

    void showMaterial(Scene::MaterialHandle material);
    //! A light shows its tree and an instance the tree of its first material; other objects keep the shown tree.
    void showSelection(const Selection& selection);
    void showEnvironment();
    //! The scene changed: shows the tree of the owner again, or nothing when the scene no longer has the owner.
    void refresh();
    std::string getOwnerName() const { return editor.getOwnerName(); }

    //! Called when the shown owner or its name changes.
    std::function<void()> retitled;

private:
    //! A node in canvas dp; the sockets have the y offset of their centre, or a negative one when they are not shown.
    struct NodeBox {
        Rml::Vector2f position;
        float width = 0.0f;
        float height = 0.0f;
        bool shown = false;
        std::vector<float> inputY;
        std::vector<float> outputY;
    };

    struct SocketRef {
        std::uint32_t node = 0;
        std::uint32_t socket = 0;
        bool output = false;
    };

    void onValue(Rml::DataModelHandle, Rml::Event& event, const Rml::VariantList& arguments);
    void onPick(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onEnter(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onUp(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onWorld(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onUseNodes(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onFrame(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onUndo(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onRedo(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onMenuSearch(Rml::DataModelHandle, Rml::Event& event, const Rml::VariantList& arguments);
    void onMenuPick(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onMenuBack(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);

    void bindGraph();
    void rebuild(bool newView);
    void buildRows(const Materials::NodeTree& tree);
    void layoutShapes();
    void showSelectionRows();
    void select(std::uint32_t node, bool extend);
    void selectArea(NodeGraphShapes::Rectangle area, bool extend);
    std::vector<std::uint32_t> selectedNodes() const;
    std::span<const NodeGraphMoved> drag(Rml::Vector2f moved);
    void drop(bool keep);
    std::optional<NodeGraphLinkStart> beginLink(NodeGraphSocket socket);
    //! Returns true when the add menu opened for the link.
    bool endLink(std::optional<NodeGraphSocket> target, Rml::Vector2f canvas, Rml::Vector2f area, Rml::Vector2f size);
    void markLinkable(bool dragging);
    void cut(std::span<const std::size_t> links);
    bool command(NodeGraphCommand command, Rml::Vector2f canvas);
    void copy();
    void paste(Rml::Vector2f canvas);
    void toggle(bool Materials::Node::* flag, NodeEdit effect);
    void openMenu(Rml::Vector2f canvas, Rml::Vector2f area, Rml::Vector2f size);
    void showMenuItems();
    bool menuKey(NodeMenuKey key);
    bool closeMenu();
    void pickMenuItem(std::size_t item);
    void addNode(const std::string& type);
    void apply(const NodeTreeEditor::Change& change, NodeEdit effect, std::string_view merge = {});
    std::pair<std::size_t, std::size_t> socketArguments(const Rml::VariantList& arguments, const char* event) const;
    Rml::Vector2f socketPosition(const SocketRef& socket) const;

    RmlKit::UI::Model& model;
    AppModel& app;
    Session::Session& session;
    NodeGraphInstancer& graphs;
    NodeTreeEditor editor;
    NodePanelController panel;
    //! The tree node of each node row, and the row of each tree node or SIZE_MAX.
    std::vector<std::uint32_t> rowNodes;
    std::vector<std::size_t> nodeRows;
    std::vector<NodeBox> boxes;
    std::vector<bool> selected;
    std::uint32_t active = Materials::noNode;
    std::vector<std::pair<std::uint32_t, Rml::Vector2f>> dragStarts;
    std::vector<NodeGraphMoved> moved;
    std::optional<SocketRef> linkFrom;
    //! The link that the user picked up from an input; layout does not draw it.
    std::optional<std::uint32_t> detached;
    std::optional<SocketRef> menuLink;
    Rml::Vector2f menuCanvas;
    std::optional<Materials::NodeTree> clipboard;
    Materials::NodeTreeKind clipboardKind = Materials::NodeTreeKind::Material;
    std::string shownOwner;
};

}
