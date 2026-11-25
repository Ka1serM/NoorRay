#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <vector>

#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/ElementInstancer.h>
#include <RmlUi/Core/EventListener.h>
#include <RmlUi/Core/Geometry.h>

namespace NoorRay::App {

//! What a <nodegraph> draws behind its nodes, in dp of the graph canvas, whose y axis points down.
struct NodeGraphShapes {
    struct Rectangle {
        Rml::Vector2f position;
        Rml::Vector2f size;
    };
    struct Curve {
        Rml::Vector2f from;
        Rml::Vector2f to;
        Rml::Colourb colour;
        //! The index of the link in the shown tree.
        std::uint32_t link = 0;
    };
    std::vector<Rectangle> frames;
    std::vector<Curve> links;
    Rectangle bounds;
    //! Changes when the shapes change.
    std::uint64_t revision = 0;
    //! Changes when the graph must frame all nodes, as when another tree is shown.
    std::uint64_t view = 0;
};

//! A socket of a node row; socket indexes the inputs or the outputs of the tree node.
struct NodeGraphSocket {
    std::size_t row = 0;
    std::uint32_t socket = 0;
    bool output = false;
};

//! The fixed end of a link that the user drags, in canvas dp.
struct NodeGraphLinkStart {
    Rml::Vector2f anchor;
    bool fromOutput = true;
    Rml::Colourb colour;
};

struct NodeGraphMoved {
    std::size_t row = 0;
    Rml::Vector2f corner;
};

enum class NodeGraphCommand : std::uint8_t { Delete, Duplicate, Copy, Cut, Paste, Mute, Collapse, Undo, Redo, Enter, Exit, SelectAll, SelectNone };

enum class NodeMenuKey : std::uint8_t { Up, Down, Enter, Escape };

struct NodeGraphEvents {
    //! A press on a node; extend adds it to the selection.
    std::function<void(std::size_t row, bool extend)> press;
    //! A press and release on a node without a drag.
    std::function<void(std::size_t row, bool extend)> click;
    //! A box drag on empty canvas; a click on empty canvas gives an empty area.
    std::function<void(NodeGraphShapes::Rectangle area, bool extend)> boxSelect;
    //! Moves the selected nodes by this many canvas dp since the drag started; returns the new canvas corners of the moved rows.
    std::function<std::span<const NodeGraphMoved>(Rml::Vector2f moved)> drag;
    //! Ends a drag; a cancelled drag puts the nodes back.
    std::function<void(bool keep)> drop;
    std::function<std::optional<NodeGraphLinkStart>(NodeGraphSocket socket)> beginLink;
    //! Ends a link drag on a socket or on empty canvas (canvas point, area point and area size in dp); true when the add menu opened.
    std::function<bool(std::optional<NodeGraphSocket> target, Rml::Vector2f canvas, Rml::Vector2f area, Rml::Vector2f size)> endLink;
    //! Opens the add menu at a canvas point and a point of the graph area in dp, which has the given size in dp.
    std::function<void(Rml::Vector2f canvas, Rml::Vector2f area, Rml::Vector2f size)> menu;
    //! Returns false when the add menu is not open.
    std::function<bool(NodeMenuKey key)> menuKey;
    std::function<bool()> closeMenu;
    std::function<bool(NodeGraphCommand command, Rml::Vector2f canvas)> command;
    //! The links, as indices into NodeGraphShapes::links, that a cut line crosses.
    std::function<void(std::span<const std::size_t> links)> cut;
};

class NodeGraphInstancer;

//! The .node-canvas child holds node elements with a node attribute, and their .node-socket elements have a socket attribute; links and frames draw below them.
class NodeGraphElement final : public Rml::Element, public Rml::EventListener {
public:
    NodeGraphElement(const Rml::String& tag, NodeGraphInstancer& instancer);
    ~NodeGraphElement() override;

    void ProcessEvent(Rml::Event& event) override;
    //! Draws the link, the box or the cut line that the user drags, above the nodes.
    void renderOverlay(Rml::RenderManager& renderManager);

protected:
    void OnUpdate() override;
    void OnRender() override;

private:
    enum class Drag { None, Pan, Node, Grab, Box, Link, Cut };

    struct GeometryKey {
        std::uint64_t revision = 0;
        Rml::Vector2f pan;
        float zoom = 1.0f;
        float density = 1.0f;
        bool operator==(const GeometryKey&) const = default;
    };

    void press(Rml::Event& event);
    void pressLeft(Rml::Event& event, Rml::Element* target);
    void move(Rml::Vector2f mouse);
    void release(Rml::Vector2f mouse);
    void startDrag(Drag started);
    void startGrab();
    void moveNodes(Rml::Vector2f mouse);
    void endDrag(bool keep);
    void key(Rml::Event& event);
    bool menuKey(int key);
    bool runCommand(int key, bool control, bool shift, bool alt);
    void zoomAt(Rml::Vector2f mouse, float wheel);
    void frameAll();
    void applyTransform();
    void watchPointer(bool watched);
    float getDensity() const;
    Rml::Vector2f toLocal(Rml::Vector2f mouse);
    Rml::Vector2f toCanvas(Rml::Vector2f mouse);
    void cutLinks();
    void buildGeometry(Rml::RenderManager& renderManager, const GeometryKey& key);
    void requestDraw();

    NodeGraphInstancer& instancer;
    Rml::Element* canvas = nullptr;
    Rml::Element* overlay = nullptr;
    Rml::Vector2f pan;
    float zoom = 1.0f;
    std::uint64_t shownView = 0;
    bool framePending = false;
    bool menuFocusPending = false;
    Drag drag = Drag::None;
    bool moved = false;
    bool extend = false;
    Rml::Vector2f pressedAt;
    Rml::Vector2f mouseAt;
    Rml::Vector2f panAtPress;
    std::size_t pressedRow = 0;
    std::vector<Rml::Element*> rowElements;
    NodeGraphLinkStart link;
    Rml::Geometry geometry;
    std::optional<GeometryKey> geometryKey;
    Rml::Context* pointerContext = nullptr;
};

//! Creates the <nodegraph> elements and keeps what the controller gives them.
class NodeGraphInstancer final : public Rml::ElementInstancer {
public:
    NodeGraphShapes& getShapes() { return shapes; }
    NodeGraphEvents& getEvents() { return events; }

    Rml::ElementPtr InstanceElement(Rml::Element* parent, const Rml::String& tag, const Rml::XMLAttributes& attributes) override;
    void ReleaseElement(Rml::Element* element) override;

private:
    NodeGraphShapes shapes;
    NodeGraphEvents events;
};

}
