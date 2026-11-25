#include "NoorRay.App/UI/Controllers/NodeGraphController.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <format>
#include <numeric>
#include <utility>

#include "NoorRay.App/UI/Controllers/NodeSockets.h"
#include "NoorRay.App/UI/Controls/FieldCommit.h"
#include "NoorRay.App/UI/Controls/ViewportElement.h"
#include "NoorRay.App/UI/Models/AppModel.h"
#include "NoorRay/Materials/NodeCatalog.h"
#include "RmlKit/UI/Backend/Fatal.h"
#include "RmlKit/UI/Models/Model.h"

namespace NoorRay::App {

namespace {

using Materials::Node;
using Materials::NodeCatalog;
using Materials::NodeLink;
using Materials::NodeSocket;
using Materials::NodeSocketType;
using Materials::NodeTree;
using Materials::noNode;
using Vec2 = Rml::Vector2f;

//! The heights match .node-header, .node-body and .node-row in noorray.rcss.
constexpr float headerHeight = 24.0f;
constexpr float bodyTop = 4.0f;
constexpr float bodyBottom = 6.0f;
constexpr float rowHeight = 22.0f;
constexpr float rerouteSize = 12.0f;
constexpr float framePadding = 12.0f;
constexpr float frameTitle = 24.0f;
constexpr float emptyFrameHeight = 100.0f;
//! The size of .node-menu in noorray.rcss.
constexpr float menuWidth = 240.0f;
constexpr float menuHeight = 320.0f;
constexpr Rml::byte mutedLinkAlpha = 70;
const Rml::Colourb invalidLinkColour{230, 60, 60};
constexpr std::array<std::string_view, 11> menuCategories{"Input", "Output",    "Shader", "Displacement", "Color", "Texture",
                                                          "Math",  "Converter", "Vector", "Group",        "Layout"};

Rml::String lowercase(const std::string_view text) {
    Rml::String lower(text);
    for (char& letter : lower)
        letter = static_cast<char>(std::tolower(static_cast<unsigned char>(letter)));
    return lower;
}

//! The header class of a node, after the categories of Blender's add menu.
Rml::String categoryOf(const Node& node) {
    if (node.group != noNode || node.type == "ShaderNodeGroup")
        return "group";
    if (node.type == "NodeGroupInput")
        return "input";
    if (node.type == "NodeGroupOutput")
        return "output";
    const Materials::NodeType* type = NodeCatalog::find(node.type);
    return type ? lowercase(type->category) : Rml::String("input");
}

bool isShown(const NodeSocket& socket) {
    return socket.enabled && !socket.hidden;
}

//! The first shown socket that takes the type, with the same type before a converted one.
std::optional<std::uint32_t> findSocket(const std::vector<NodeSocket>& sockets, const NodeSocketType type, const bool inputs) {
    const auto fits = [&](const NodeSocket& socket) { return inputs ? canLink(type, socket.type) : canLink(socket.type, type); };
    for (const bool exact : {true, false})
        for (std::uint32_t index = 0; index < sockets.size(); ++index)
            if (isShown(sockets[index]) && fits(sockets[index]) && (!exact || sockets[index].type == type))
                return index;
    return std::nullopt;
}

struct Bounds {
    Vec2 low{1e30f, 1e30f};
    Vec2 high{-1e30f, -1e30f};

    void grow(const Vec2 position, const Vec2 size) {
        low = {std::min(low.x, position.x), std::min(low.y, position.y)};
        high = {std::max(high.x, position.x + size.x), std::max(high.y, position.y + size.y)};
    }
    bool isEmpty() const { return low.x > high.x; }
};

bool overlaps(const NodeGraphShapes::Rectangle& a, const Vec2 position, const Vec2 size) {
    return a.position.x <= position.x + size.x && position.x <= a.position.x + a.size.x && a.position.y <= position.y + size.y
           && position.y <= a.position.y + a.size.y;
}

}

NodeGraphController::NodeGraphController(RmlKit::UI::Model& model, AppModel& app, Session::Session& session, ViewportElementInstancer& viewports,
                                         NodeGraphInstancer& graphs)
    : model(model), app(app), session(session), graphs(graphs), editor(session, viewports), panel(model, app, session, editor) {
    model.bindEvent("nodeGraphValue", &NodeGraphController::onValue, this);
    model.bindEvent("nodeGraphPick", &NodeGraphController::onPick, this);
    model.bindEvent("nodeGraphEnter", &NodeGraphController::onEnter, this);
    model.bindEvent("nodeGraphUp", &NodeGraphController::onUp, this);
    model.bindEvent("nodeGraphWorld", &NodeGraphController::onWorld, this);
    model.bindEvent("nodeGraphUseNodes", &NodeGraphController::onUseNodes, this);
    model.bindEvent("nodeGraphFrame", &NodeGraphController::onFrame, this);
    model.bindEvent("nodeGraphUndo", &NodeGraphController::onUndo, this);
    model.bindEvent("nodeGraphRedo", &NodeGraphController::onRedo, this);
    model.bindEvent("nodeMenuSearch", &NodeGraphController::onMenuSearch, this);
    model.bindEvent("nodeMenuPick", &NodeGraphController::onMenuPick, this);
    model.bindEvent("nodeMenuBack", &NodeGraphController::onMenuBack, this);
    editor.changed = [this](const bool newView) { rebuild(newView); };
    viewports.getEvents().ticks.push_back([this] { return editor.tick(); });
    bindGraph();
}

void NodeGraphController::bindGraph() {
    NodeGraphEvents& events = graphs.getEvents();
    events.press = [this](const std::size_t row, const bool extend) {
        select(rowNodes.at(row), extend);
        showSelectionRows();
    };
    events.click = [this](const std::size_t row, const bool extend) {
        if (extend)
            return;
        std::ranges::fill(selected, false);
        select(rowNodes.at(row), false);
        showSelectionRows();
    };
    events.boxSelect = [this](const NodeGraphShapes::Rectangle area, const bool extend) { selectArea(area, extend); };
    events.drag = [this](const Vec2 offset) { return drag(offset); };
    events.drop = [this](const bool keep) { drop(keep); };
    events.beginLink = [this](const NodeGraphSocket socket) { return beginLink(socket); };
    events.endLink = [this](const std::optional<NodeGraphSocket> target, const Vec2 canvas, const Vec2 area, const Vec2 size) {
        return endLink(target, canvas, area, size);
    };
    events.menu = [this](const Vec2 canvas, const Vec2 area, const Vec2 size) { openMenu(canvas, area, size); };
    events.menuKey = [this](const NodeMenuKey key) { return menuKey(key); };
    events.closeMenu = [this] { return closeMenu(); };
    events.command = [this](const NodeGraphCommand command, const Vec2 canvas) { return this->command(command, canvas); };
    events.cut = [this](const std::span<const std::size_t> links) { cut(links); };
}

void NodeGraphController::showMaterial(const Scene::MaterialHandle material) {
    editor.show(material);
}

void NodeGraphController::showSelection(const Selection& selection) {
    const Scene::Scene& scene = session.getScene();
    if (const auto* light = std::get_if<Scene::LightHandle>(&selection))
        return editor.show(*light);
    const auto* instanceHandle = std::get_if<Scene::InstanceHandle>(&selection);
    const Scene::Instance* instance = instanceHandle ? scene.find(*instanceHandle) : nullptr;
    if (!instance)
        return;
    if (scene.find(instance->material))
        return editor.show(instance->material);
    const Scene::Mesh* mesh = scene.find(instance->mesh);
    if (mesh && !mesh->subsets.empty() && scene.find(mesh->subsets.front().material))
        editor.show(mesh->subsets.front().material);
}

void NodeGraphController::showEnvironment() {
    editor.show(NodeTreeWorld{});
}

void NodeGraphController::refresh() {
    editor.refresh();
}

void NodeGraphController::rebuild(const bool newView) {
    NodeGraphModel& graph = app.nodeGraph;
    if (newView) {
        std::ranges::fill(selected, false);
        active = noNode;
        graph.menu.open = false;
        graph.notice.clear();
        ++graphs.getShapes().view;
    }
    dragStarts.clear();
    linkFrom.reset();
    detached.reset();
    const std::string owner = editor.getOwnerName();
    graph.owner = owner.empty() ? Rml::String("Nothing selected") : Rml::String(owner);
    graph.hasOwner = editor.hasOwner();
    graph.hasTree = editor.findRoot() != nullptr;
    graph.path.clear();
    const NodeTree* tree = editor.findRoot().get();
    for (const std::uint32_t group : editor.getPath()) {
        if (!tree || group >= tree->groups.size())
            break;
        const auto named = std::ranges::find(tree->nodes, group, &Node::group);
        graph.path.push_back(named == tree->nodes.end() ? Rml::String("Group") : Rml::String(named->label.empty() ? named->name : named->label));
        tree = tree->groups[group].get();
    }
    graph.error = editor.getError();
    graph.warnings.assign(editor.getWarnings().begin(), editor.getWarnings().end());
    graph.canUndo = editor.canUndo();
    graph.canRedo = editor.canRedo();
    graph.nodes.clear();
    rowNodes.clear();
    boxes.clear();
    const NodeTree* shown = editor.findShown();
    selected.resize(shown ? shown->nodes.size() : 0, false);
    if (shown && active >= shown->nodes.size())
        active = noNode;
    if (shown)
        buildRows(*shown);
    layoutShapes();
    showSelectionRows();
    if (owner != shownOwner) {
        shownOwner = owner;
        if (retitled)
            retitled();
    }
}

void NodeGraphController::buildRows(const NodeTree& tree) {
    NodeGraphModel& graph = app.nodeGraph;
    const std::size_t count = tree.nodes.size();
    std::vector<std::uint32_t> firstInput(count + 1, 0);
    for (std::size_t node = 0; node < count; ++node)
        firstInput[node + 1] = firstInput[node] + static_cast<std::uint32_t>(tree.nodes[node].inputs.size());
    std::vector<bool> linked(firstInput.back(), false);
    for (const NodeLink& link : tree.links)
        if (link.toNode < count && link.toSocket < tree.nodes[link.toNode].inputs.size())
            linked[firstInput[link.toNode] + link.toSocket] = true;
    boxes.resize(count);
    nodeRows.assign(count, SIZE_MAX);
    const std::function<Vec2(std::uint32_t, std::size_t)> originOf = [&](const std::uint32_t node, const std::size_t depth) {
        const Node& current = tree.nodes[node];
        const Vec2 own(current.location.x, -current.location.y);
        if (current.parent >= count || depth > count)
            return own;
        return own + originOf(current.parent, depth + 1);
    };
    for (std::uint32_t index = 0; index < count; ++index) {
        const Node& node = tree.nodes[index];
        NodeBox& box = boxes[index];
        box.position = originOf(index, 0);
        box.inputY.assign(node.inputs.size(), -1.0f);
        box.outputY.assign(node.outputs.size(), -1.0f);
        if (node.type == "NodeFrame")
            continue;
        box.shown = true;
        NodeGraphNodeRow row{.title = !node.label.empty() ? node.label : node.name,
                             .category = categoryOf(node),
                             .muted = node.muted,
                             .hidden = node.hidden,
                             .reroute = node.type == "NodeReroute",
                             .group = node.group != noNode};
        nodeRows[index] = rowNodes.size();
        if (row.reroute) {
            std::ranges::fill(box.inputY, 0.0f);
            std::ranges::fill(box.outputY, 0.0f);
            row.x = box.position.x - rerouteSize * 0.5f;
            row.y = box.position.y - rerouteSize * 0.5f;
            row.width = rerouteSize;
            row.height = rerouteSize;
            rowNodes.push_back(index);
            graph.nodes.push_back(std::move(row));
            continue;
        }
        box.width = node.width;
        float cursor = headerHeight + bodyTop;
        for (std::uint32_t socket = 0; socket < node.outputs.size(); ++socket) {
            if (!isShown(node.outputs[socket]))
                continue;
            box.outputY[socket] = node.hidden ? headerHeight * 0.5f : cursor + rowHeight * 0.5f;
            row.outputs.push_back(socketRow(node.outputs[socket], socket, false));
            cursor += rowHeight;
        }
        for (std::uint32_t socket = 0; socket < node.inputs.size(); ++socket) {
            if (!isShown(node.inputs[socket]))
                continue;
            box.inputY[socket] = node.hidden ? headerHeight * 0.5f : cursor + rowHeight * 0.5f;
            row.inputs.push_back(socketRow(node.inputs[socket], socket, !linked[firstInput[index] + socket] && !hidesValue(node, socket)));
            const NodeGraphSocketRow& shown = row.inputs.back();
            cursor += rowHeight * static_cast<float>(shown.editor == "vector" ? 1 + shown.components : 1);
        }
        box.height = node.hidden ? headerHeight : cursor + bodyBottom;
        row.x = box.position.x;
        row.y = box.position.y;
        row.width = box.width;
        row.height = box.height;
        rowNodes.push_back(index);
        graph.nodes.push_back(std::move(row));
    }
}

void NodeGraphController::layoutShapes() {
    NodeGraphShapes& shapes = graphs.getShapes();
    shapes.frames.clear();
    shapes.links.clear();
    app.nodeGraph.frames.clear();
    ++shapes.revision;
    const NodeTree* tree = editor.findShown();
    Bounds all;
    if (!tree || boxes.size() != tree->nodes.size()) {
        shapes.bounds = {};
        return;
    }
    const std::size_t count = tree->nodes.size();
    for (const NodeBox& box : boxes)
        if (box.shown)
            all.grow(box.position, {box.width, box.height});
    std::vector<std::optional<Bounds>> frames(count);
    const std::function<Bounds(std::uint32_t, std::size_t)> frameOf = [&](const std::uint32_t frame, const std::size_t depth) {
        if (frames[frame])
            return *frames[frame];
        Bounds bounds;
        for (std::uint32_t child = 0; child < count && depth <= count; ++child) {
            if (tree->nodes[child].parent != frame)
                continue;
            if (boxes[child].shown) {
                bounds.grow(boxes[child].position, {boxes[child].width, boxes[child].height});
                continue;
            }
            const Bounds inner = frameOf(child, depth + 1);
            if (!inner.isEmpty())
                bounds.grow(inner.low, inner.high - inner.low);
        }
        if (bounds.isEmpty())
            bounds.grow(boxes[frame].position, {tree->nodes[frame].width, emptyFrameHeight});
        else
            bounds = Bounds{bounds.low - Vec2(framePadding, framePadding + frameTitle), bounds.high + Vec2(framePadding, framePadding)};
        frames[frame] = bounds;
        return bounds;
    };
    for (std::uint32_t index = 0; index < count; ++index) {
        const Node& node = tree->nodes[index];
        if (node.type != "NodeFrame")
            continue;
        const Bounds bounds = frameOf(index, 0);
        shapes.frames.push_back({bounds.low, bounds.high - bounds.low});
        app.nodeGraph.frames.push_back({node.label, bounds.low.x, bounds.low.y, bounds.high.x - bounds.low.x});
        all.grow(bounds.low, bounds.high - bounds.low);
    }
    for (std::uint32_t index = 0; index < tree->links.size(); ++index) {
        const NodeLink& link = tree->links[index];
        if (link.fromNode >= count || link.toNode >= count || detached == index)
            continue;
        const NodeBox& from = boxes[link.fromNode];
        const NodeBox& to = boxes[link.toNode];
        if (link.fromSocket >= from.outputY.size() || link.toSocket >= to.inputY.size() || from.outputY[link.fromSocket] < 0.0f
            || to.inputY[link.toSocket] < 0.0f)
            continue;
        const NodeSocketType fromType = tree->nodes[link.fromNode].outputs[link.fromSocket].type;
        Rml::Colourb colour = canLink(fromType, tree->nodes[link.toNode].inputs[link.toSocket].type) ? socketColour(fromType) : invalidLinkColour;
        if (link.muted)
            colour.alpha = mutedLinkAlpha;
        shapes.links.push_back({socketPosition({link.fromNode, link.fromSocket, true}), socketPosition({link.toNode, link.toSocket, false}), colour, index});
    }
    shapes.bounds = all.isEmpty() ? NodeGraphShapes::Rectangle{} : NodeGraphShapes::Rectangle{all.low, all.high - all.low};
}

Vec2 NodeGraphController::socketPosition(const SocketRef& socket) const {
    const NodeBox& box = boxes[socket.node];
    if (socket.output)
        return box.position + Vec2(box.width, box.outputY[socket.socket]);
    return box.position + Vec2(0.0f, box.inputY[socket.socket]);
}

void NodeGraphController::showSelectionRows() {
    for (std::size_t row = 0; row < rowNodes.size(); ++row) {
        app.nodeGraph.nodes[row].selected = selected[rowNodes[row]];
        app.nodeGraph.nodes[row].active = rowNodes[row] == active;
    }
    panel.show(active);
    model.dirty("nodeGraph");
}

void NodeGraphController::select(const std::uint32_t node, const bool extend) {
    if (node >= selected.size())
        return;
    if (!extend) {
        if (!selected[node])
            std::ranges::fill(selected, false);
        selected[node] = true;
        active = node;
        return;
    }
    if (selected[node] && active == node) {
        selected[node] = false;
        active = noNode;
        return;
    }
    selected[node] = true;
    active = node;
}

void NodeGraphController::selectArea(const NodeGraphShapes::Rectangle area, const bool extend) {
    if (!extend) {
        std::ranges::fill(selected, false);
        active = noNode;
    }
    const bool empty = area.size.x <= 0.0f && area.size.y <= 0.0f;
    for (std::uint32_t node = 0; node < boxes.size() && !empty; ++node) {
        const NodeBox& box = boxes[node];
        if (!box.shown)
            continue;
        const bool reroute = nodeRows[node] < app.nodeGraph.nodes.size() && app.nodeGraph.nodes[nodeRows[node]].reroute;
        const Vec2 corner = reroute ? box.position - Vec2(rerouteSize, rerouteSize) * 0.5f : box.position;
        const Vec2 size = reroute ? Vec2(rerouteSize, rerouteSize) : Vec2(box.width, box.height);
        if (overlaps(area, corner, size))
            selected[node] = true;
    }
    showSelectionRows();
}

std::vector<std::uint32_t> NodeGraphController::selectedNodes() const {
    std::vector<std::uint32_t> nodes;
    for (std::uint32_t node = 0; node < selected.size(); ++node)
        if (selected[node])
            nodes.push_back(node);
    return nodes;
}

std::span<const NodeGraphMoved> NodeGraphController::drag(const Vec2 offset) {
    if (dragStarts.empty())
        for (const std::uint32_t node : selectedNodes())
            if (boxes[node].shown)
                dragStarts.emplace_back(node, boxes[node].position);
    moved.clear();
    for (const auto& [node, start] : dragStarts) {
        NodeBox& box = boxes[node];
        box.position = start + offset;
        const std::size_t row = nodeRows[node];
        NodeGraphNodeRow& shown = app.nodeGraph.nodes[row];
        const Vec2 corner = shown.reroute ? box.position - Vec2(rerouteSize, rerouteSize) * 0.5f : box.position;
        shown.x = corner.x;
        shown.y = corner.y;
        moved.push_back({row, corner});
    }
    layoutShapes();
    return moved;
}

void NodeGraphController::drop(const bool keep) {
    if (dragStarts.empty())
        return;
    const std::vector<std::pair<std::uint32_t, Vec2>> starts = std::exchange(dragStarts, {});
    const Vec2 offset = boxes[starts.front().first].position - starts.front().second;
    if (!keep || (offset.x == 0.0f && offset.y == 0.0f))
        return rebuild(false);
    apply(
        [&](NodeTree& tree) -> std::expected<void, Core::Error> {
            for (const auto& [node, start] : starts)
                tree.nodes[node].location = {tree.nodes[node].location.x + offset.x, tree.nodes[node].location.y - offset.y};
            return {};
        },
        NodeEdit::Layout);
}

std::optional<NodeGraphLinkStart> NodeGraphController::beginLink(const NodeGraphSocket socket) {
    const NodeTree* tree = editor.findShown();
    if (!tree || socket.row >= rowNodes.size())
        return std::nullopt;
    const std::uint32_t node = rowNodes[socket.row];
    const Node& owner = tree->nodes[node];
    if (socket.output) {
        if (socket.socket >= owner.outputs.size())
            return std::nullopt;
        linkFrom = SocketRef{node, socket.socket, true};
    } else {
        if (socket.socket >= owner.inputs.size())
            return std::nullopt;
        linkFrom = SocketRef{node, socket.socket, false};
        const auto found = std::ranges::find_if(tree->links, [&](const NodeLink& link) { return link.toNode == node && link.toSocket == socket.socket; });
        if (found != tree->links.end()) {
            detached = static_cast<std::uint32_t>(found - tree->links.begin());
            linkFrom = SocketRef{found->fromNode, found->fromSocket, true};
        }
    }
    const NodeSocket& from = linkFrom->output ? tree->nodes[linkFrom->node].outputs[linkFrom->socket] : tree->nodes[linkFrom->node].inputs[linkFrom->socket];
    markLinkable(true);
    layoutShapes();
    return NodeGraphLinkStart{socketPosition(*linkFrom), linkFrom->output, socketColour(from.type)};
}

void NodeGraphController::markLinkable(const bool dragging) {
    const NodeTree* tree = editor.findShown();
    if (!tree)
        return;
    const NodeSocketType type = linkFrom && dragging ? (linkFrom->output ? tree->nodes[linkFrom->node].outputs[linkFrom->socket].type
                                                                         : tree->nodes[linkFrom->node].inputs[linkFrom->socket].type)
                                                     : NodeSocketType::Float;
    for (std::size_t row = 0; row < rowNodes.size(); ++row) {
        const Node& node = tree->nodes[rowNodes[row]];
        const bool other = !linkFrom || rowNodes[row] != linkFrom->node;
        for (NodeGraphSocketRow& socket : app.nodeGraph.nodes[row].outputs)
            socket.dim = dragging && (linkFrom->output || !other || !canLink(node.outputs[socket.socket].type, type));
        for (NodeGraphSocketRow& socket : app.nodeGraph.nodes[row].inputs)
            socket.dim = dragging && (!linkFrom->output || !other || !canLink(type, node.inputs[socket.socket].type));
    }
    model.dirty("nodeGraph");
}

bool NodeGraphController::endLink(const std::optional<NodeGraphSocket> target, const Vec2 canvas, const Vec2 area, const Vec2 size) {
    const std::optional<SocketRef> from = linkFrom;
    const NodeTree* tree = editor.findShown();
    const std::optional<NodeLink> old = detached && tree ? std::optional(tree->links[*detached]) : std::nullopt;
    markLinkable(false);
    linkFrom.reset();
    detached.reset();
    layoutShapes();
    if (!from || !tree)
        return false;
    const auto disconnectOld = [&](NodeTree& changed) {
        if (old)
            NodeCatalog::disconnect(changed, old->toNode, old->toSocket);
    };
    std::optional<NodeLink> made;
    if (target && target->row < rowNodes.size() && target->output != from->output) {
        const SocketRef to{rowNodes[target->row], target->socket, target->output};
        made = from->output ? NodeLink{.fromNode = from->node, .fromSocket = from->socket, .toNode = to.node, .toSocket = to.socket}
                            : NodeLink{.fromNode = to.node, .fromSocket = to.socket, .toNode = from->node, .toSocket = from->socket};
    }
    if (made && old && made->toNode == old->toNode && made->toSocket == old->toSocket && made->fromNode == old->fromNode && made->fromSocket == old->fromSocket)
        return false;
    if (made) {
        apply(
            [&](NodeTree& changed) -> std::expected<void, Core::Error> {
                disconnectOld(changed);
                return NodeCatalog::connect(changed, *made);
            },
            NodeEdit::Result);
        return false;
    }
    if (old) {
        apply(
            [&](NodeTree& changed) -> std::expected<void, Core::Error> {
                disconnectOld(changed);
                return {};
            },
            NodeEdit::Result);
        return false;
    }
    if (target)
        return false;
    openMenu(canvas, area, size);
    menuLink = from;
    return true;
}

void NodeGraphController::cut(const std::span<const std::size_t> links) {
    const NodeTree* tree = editor.findShown();
    if (!tree)
        return;
    std::vector<NodeLink> removed;
    for (const std::size_t shape : links)
        removed.push_back(tree->links[graphs.getShapes().links[shape].link]);
    apply(
        [&](NodeTree& changed) -> std::expected<void, Core::Error> {
            for (const NodeLink& link : removed)
                NodeCatalog::disconnect(changed, link.toNode, link.toSocket);
            return {};
        },
        NodeEdit::Result);
}

bool NodeGraphController::command(const NodeGraphCommand command, const Vec2 canvas) {
    const NodeTree* tree = editor.findShown();
    if (!tree)
        return false;
    const std::vector<std::uint32_t> nodes = selectedNodes();
    switch (command) {
    case NodeGraphCommand::Delete:
        if (nodes.empty())
            return false;
        std::ranges::fill(selected, false);
        active = noNode;
        apply(
            [&](NodeTree& changed) -> std::expected<void, Core::Error> {
                NodeCatalog::removeNodes(changed, nodes);
                return {};
            },
            NodeEdit::Result);
        return true;
    case NodeGraphCommand::Duplicate: {
        if (nodes.empty())
            return false;
        std::vector<std::uint32_t> copies;
        apply(
            [&](NodeTree& changed) -> std::expected<void, Core::Error> {
                copies = NodeCatalog::duplicateNodes(changed, nodes, {0.0f, 0.0f});
                return {};
            },
            NodeEdit::Result);
        std::ranges::fill(selected, false);
        for (const std::uint32_t copy : copies)
            if (copy < selected.size())
                selected[copy] = true;
        active = copies.empty() ? noNode : copies.back();
        showSelectionRows();
        return true;
    }
    case NodeGraphCommand::Copy:
        copy();
        return !nodes.empty();
    case NodeGraphCommand::Cut:
        copy();
        return !nodes.empty() && this->command(NodeGraphCommand::Delete, canvas);
    case NodeGraphCommand::Paste:
        paste(canvas);
        return true;
    case NodeGraphCommand::Mute:
        toggle(&Node::muted, NodeEdit::Result);
        return !nodes.empty();
    case NodeGraphCommand::Collapse:
        toggle(&Node::hidden, NodeEdit::Layout);
        return !nodes.empty();
    case NodeGraphCommand::Undo:
    case NodeGraphCommand::Redo:
        std::ranges::fill(selected, false);
        active = noNode;
        if (!(command == NodeGraphCommand::Undo ? editor.undo() : editor.redo()))
            showSelectionRows();
        return true;
    case NodeGraphCommand::Enter:
        if (active >= tree->nodes.size() || tree->nodes[active].group == noNode)
            return false;
        editor.enter(tree->nodes[active].group);
        return true;
    case NodeGraphCommand::Exit:
        if (editor.getPath().empty())
            return false;
        editor.leave(editor.getPath().size() - 1);
        return true;
    case NodeGraphCommand::SelectAll:
    case NodeGraphCommand::SelectNone:
        std::ranges::fill(selected, command == NodeGraphCommand::SelectAll);
        for (std::uint32_t node = 0; node < boxes.size(); ++node)
            selected[node] = selected[node] && boxes[node].shown;
        if (command == NodeGraphCommand::SelectNone)
            active = noNode;
        showSelectionRows();
        return true;
    }
    return false;
}

void NodeGraphController::toggle(bool Node::* flag, const NodeEdit effect) {
    const std::vector<std::uint32_t> nodes = selectedNodes();
    const NodeTree* tree = editor.findShown();
    if (nodes.empty() || !tree)
        return;
    const bool set = !(tree->nodes[nodes.front()].*flag);
    apply(
        [&](NodeTree& changed) -> std::expected<void, Core::Error> {
            for (const std::uint32_t node : nodes)
                changed.nodes[node].*flag = set;
            return {};
        },
        effect);
}

void NodeGraphController::copy() {
    const NodeTree* tree = editor.findShown();
    const std::vector<std::uint32_t> nodes = selectedNodes();
    if (!tree || nodes.empty())
        return;
    NodeTree copied;
    std::vector<std::uint32_t> copiedIndex(tree->nodes.size(), noNode);
    for (std::uint32_t index = 0; index < nodes.size(); ++index)
        copiedIndex[nodes[index]] = index;
    for (const std::uint32_t node : nodes) {
        Node entry = tree->nodes[node];
        if (entry.parent < copiedIndex.size() && copiedIndex[entry.parent] != noNode) {
            entry.parent = copiedIndex[entry.parent];
        } else {
            entry.parent = noNode;
            entry.location = {boxes[node].position.x, -boxes[node].position.y};
        }
        if (entry.group < tree->groups.size()) {
            copied.groups.push_back(tree->groups[entry.group]);
            entry.group = static_cast<std::uint32_t>(copied.groups.size() - 1);
        }
        copied.nodes.push_back(std::move(entry));
    }
    for (const NodeLink& link : tree->links) {
        if (copiedIndex[link.fromNode] == noNode || copiedIndex[link.toNode] == noNode)
            continue;
        NodeLink entry = link;
        entry.fromNode = copiedIndex[link.fromNode];
        entry.toNode = copiedIndex[link.toNode];
        copied.links.push_back(entry);
    }
    clipboard = std::move(copied);
    clipboardKind = editor.getKind();
}

//! The copied nodes go into the tree as they are, then the catalog duplicates them with unique names and the originals go away again.
void NodeGraphController::paste(const Vec2 canvas) {
    if (!clipboard)
        return;
    if (clipboardKind != editor.getKind()) {
        app.nodeGraph.notice = "The copied nodes belong to another kind of node tree.";
        model.dirty("nodeGraph");
        return;
    }
    Bounds bounds;
    for (const Node& node : clipboard->nodes)
        if (node.parent == noNode)
            bounds.grow({node.location.x, node.location.y}, {0.0f, 0.0f});
    const Vec2 centre = bounds.isEmpty() ? Vec2(0.0f, 0.0f) : (bounds.low + bounds.high) * 0.5f;
    const Core::float2 offset{canvas.x - centre.x, -canvas.y - centre.y};
    std::size_t pasted = 0;
    apply(
        [&](NodeTree& tree) -> std::expected<void, Core::Error> {
            const auto first = static_cast<std::uint32_t>(tree.nodes.size());
            const auto groupBase = static_cast<std::uint32_t>(tree.groups.size());
            for (Node node : clipboard->nodes) {
                if (node.group != noNode)
                    node.group += groupBase;
                if (node.parent != noNode)
                    node.parent += first;
                tree.nodes.push_back(std::move(node));
            }
            tree.groups.insert(tree.groups.end(), clipboard->groups.begin(), clipboard->groups.end());
            for (NodeLink link : clipboard->links) {
                link.fromNode += first;
                link.toNode += first;
                tree.links.push_back(link);
            }
            std::vector<std::uint32_t> originals(clipboard->nodes.size());
            std::iota(originals.begin(), originals.end(), first);
            pasted = NodeCatalog::duplicateNodes(tree, originals, offset).size();
            NodeCatalog::removeNodes(tree, originals);
            return {};
        },
        NodeEdit::Result);
    std::ranges::fill(selected, false);
    for (std::size_t node = selected.size() - std::min(pasted, selected.size()); node < selected.size(); ++node)
        selected[node] = boxes[node].shown;
    active = noNode;
    showSelectionRows();
}

void NodeGraphController::openMenu(const Vec2 canvas, const Vec2 area, const Vec2 size) {
    if (!editor.findShown())
        return;
    NodeMenuModel& menu = app.nodeGraph.menu;
    menu.open = true;
    menu.x = std::clamp(area.x, 0.0f, std::max(size.x - menuWidth, 0.0f));
    menu.y = std::clamp(area.y, 0.0f, std::max(size.y - menuHeight, 0.0f));
    menu.search.clear();
    menu.category.clear();
    menuCanvas = canvas;
    menuLink.reset();
    showMenuItems();
}

void NodeGraphController::showMenuItems() {
    NodeMenuModel& menu = app.nodeGraph.menu;
    menu.items.clear();
    menu.active = 0;
    const std::vector<const Materials::NodeType*> types = NodeCatalog::search(menu.search, editor.getKind());
    if (!menu.search.empty()) {
        for (const Materials::NodeType* type : types)
            menu.items.push_back({type->name, type->type, type->category});
    } else if (!menu.category.empty()) {
        for (const Materials::NodeType* type : types)
            if (type->category == menu.category)
                menu.items.push_back({type->name, type->type, type->category});
        std::ranges::sort(menu.items, {}, &NodeMenuItem::name);
    } else {
        std::vector<std::string> categories(menuCategories.begin(), menuCategories.end());
        for (const Materials::NodeType* type : types)
            if (std::ranges::find(categories, type->category) == categories.end())
                categories.push_back(type->category);
        for (const std::string& category : categories)
            if (std::ranges::any_of(types, [&](const Materials::NodeType* type) { return type->category == category; }))
                menu.items.push_back({category, {}, category});
    }
    model.dirty("nodeGraph");
}

bool NodeGraphController::menuKey(const NodeMenuKey key) {
    NodeMenuModel& menu = app.nodeGraph.menu;
    if (!menu.open)
        return false;
    const int count = static_cast<int>(menu.items.size());
    switch (key) {
    case NodeMenuKey::Up:
        menu.active = count == 0 ? 0 : (menu.active + count - 1) % count;
        break;
    case NodeMenuKey::Down:
        menu.active = count == 0 ? 0 : (menu.active + 1) % count;
        break;
    case NodeMenuKey::Enter:
        if (menu.active < count)
            pickMenuItem(static_cast<std::size_t>(menu.active));
        return true;
    case NodeMenuKey::Escape:
        if (!menu.category.empty() && menu.search.empty()) {
            menu.category.clear();
            showMenuItems();
            return true;
        }
        closeMenu();
        return true;
    }
    model.dirty("nodeGraph");
    return true;
}

bool NodeGraphController::closeMenu() {
    if (!app.nodeGraph.menu.open)
        return false;
    app.nodeGraph.menu.open = false;
    menuLink.reset();
    model.dirty("nodeGraph");
    return true;
}

void NodeGraphController::onMenuSearch(Rml::DataModelHandle, Rml::Event& event, const Rml::VariantList& arguments) {
    RmlKit::UI::Model::requireNoArguments(arguments, "nodeMenuSearch");
    NodeMenuModel& menu = app.nodeGraph.menu;
    const Rml::String search = event.GetParameter<Rml::String>("value", menu.search);
    if (search == menu.search && !menu.items.empty())
        return;
    menu.search = search;
    showMenuItems();
}

void NodeGraphController::onMenuPick(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    pickMenuItem(RmlKit::UI::Model::indexArgument(arguments, "nodeMenuPick", app.nodeGraph.menu.items.size()));
}

void NodeGraphController::onMenuBack(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    RmlKit::UI::Model::requireNoArguments(arguments, "nodeMenuBack");
    app.nodeGraph.menu.category.clear();
    showMenuItems();
}

void NodeGraphController::pickMenuItem(const std::size_t item) {
    NodeMenuModel& menu = app.nodeGraph.menu;
    const NodeMenuItem picked = menu.items[item];
    if (picked.type.empty()) {
        menu.category = picked.category;
        showMenuItems();
        return;
    }
    addNode(picked.type);
}

void NodeGraphController::addNode(const std::string& type) {
    const std::optional<SocketRef> link = menuLink;
    closeMenu();
    const Materials::NodeType* entry = NodeCatalog::find(type);
    Vec2 corner = menuCanvas;
    if (link && !link->output && entry)
        corner.x -= entry->width;
    std::uint32_t added = noNode;
    Rml::String linkError;
    apply(
        [&](NodeTree& tree) -> std::expected<void, Core::Error> {
            const std::expected<std::uint32_t, Core::Error> index = NodeCatalog::addNode(tree, type, {corner.x, -corner.y});
            if (!index)
                return std::unexpected(index.error());
            added = *index;
            if (!link)
                return {};
            const Node& other = tree.nodes[link->node];
            const Node& node = tree.nodes[added];
            std::optional<NodeLink> made;
            if (link->output) {
                if (const std::optional<std::uint32_t> input = findSocket(node.inputs, other.outputs[link->socket].type, true))
                    made = NodeLink{.fromNode = link->node, .fromSocket = link->socket, .toNode = added, .toSocket = *input};
            } else if (const std::optional<std::uint32_t> output = findSocket(node.outputs, other.inputs[link->socket].type, false)) {
                made = NodeLink{.fromNode = added, .fromSocket = *output, .toNode = link->node, .toSocket = link->socket};
            }
            if (!made)
                return {};
            if (auto connected = NodeCatalog::connect(tree, *made); !connected)
                linkError = connected.error().message;
            return {};
        },
        NodeEdit::Result);
    if (added == noNode || added >= selected.size())
        return;
    std::ranges::fill(selected, false);
    selected[added] = true;
    active = added;
    if (!linkError.empty())
        app.nodeGraph.notice = linkError;
    showSelectionRows();
}

void NodeGraphController::apply(const NodeTreeEditor::Change& change, const NodeEdit effect, const std::string_view merge) {
    const std::expected<void, Core::Error> applied = editor.edit(change, effect, merge);
    app.nodeGraph.notice = applied ? Rml::String() : Rml::String(applied.error().message);
    model.dirty("nodeGraph");
}

std::pair<std::size_t, std::size_t> NodeGraphController::socketArguments(const Rml::VariantList& arguments, const char* event) const {
    if (arguments.size() != 2)
        RmlKit::UI::fatal(std::format("{} takes two arguments: the node row and the input row", event));
    const std::size_t row = RmlKit::UI::Model::indexOf(arguments[0], event);
    const std::size_t input = RmlKit::UI::Model::indexOf(arguments[1], event);
    if (row >= app.nodeGraph.nodes.size() || input >= app.nodeGraph.nodes[row].inputs.size())
        RmlKit::UI::fatal(std::format("{}: the node graph has no input {} of node row {}", event, input, row));
    return {row, input};
}

void NodeGraphController::onValue(Rml::DataModelHandle, Rml::Event& event, const Rml::VariantList& arguments) {
    if (!FieldCommit::isCommit(event))
        return;
    const auto [row, input] = socketArguments(arguments, "nodeGraphValue");
    const NodeGraphSocketRow& shown = app.nodeGraph.nodes[row].inputs[input];
    const NodeTree* tree = editor.findShown();
    if (!tree || !isEditorEvent(shown, event))
        return;
    const std::uint32_t node = rowNodes[row];
    const std::uint32_t socket = shown.socket;
    const Core::float4 old = tree->nodes[node].inputs[socket].value;
    const Core::float4 value = valueOf(shown, old);
    if (value == old)
        return;
    apply(
        [&](NodeTree& changed) -> std::expected<void, Core::Error> {
            changed.nodes[node].inputs[socket].value = value;
            return {};
        },
        NodeEdit::Result, std::format("input {} {}", node, socket));
}

void NodeGraphController::onPick(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    const auto [row, input] = socketArguments(arguments, "nodeGraphPick");
    std::ranges::fill(selected, false);
    select(rowNodes[row], false);
    showSelectionRows();
    panel.pick(app.nodeGraph.nodes[row].inputs[input].socket);
}

void NodeGraphController::onEnter(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    const std::size_t row = RmlKit::UI::Model::indexArgument(arguments, "nodeGraphEnter", rowNodes.size());
    const NodeTree* tree = editor.findShown();
    if (tree && tree->nodes[rowNodes[row]].group != noNode)
        editor.enter(tree->nodes[rowNodes[row]].group);
}

void NodeGraphController::onUp(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    editor.leave(RmlKit::UI::Model::indexArgument(arguments, "nodeGraphUp", editor.getPath().size() + 1));
}

void NodeGraphController::onWorld(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    RmlKit::UI::Model::requireNoArguments(arguments, "nodeGraphWorld");
    showEnvironment();
}

void NodeGraphController::onUseNodes(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    RmlKit::UI::Model::requireNoArguments(arguments, "nodeGraphUseNodes");
    const auto used = editor.useNodes();
    app.nodeGraph.notice = used ? Rml::String() : Rml::String(used.error().message);
    model.dirty("nodeGraph");
}

void NodeGraphController::onFrame(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    RmlKit::UI::Model::requireNoArguments(arguments, "nodeGraphFrame");
    ++graphs.getShapes().view;
    model.dirty("nodeGraph");
}

void NodeGraphController::onUndo(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    RmlKit::UI::Model::requireNoArguments(arguments, "nodeGraphUndo");
    command(NodeGraphCommand::Undo, {});
}

void NodeGraphController::onRedo(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    RmlKit::UI::Model::requireNoArguments(arguments, "nodeGraphRedo");
    command(NodeGraphCommand::Redo, {});
}

}
