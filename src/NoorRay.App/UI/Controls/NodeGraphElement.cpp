#include "NoorRay.App/UI/Controls/NodeGraphElement.h"

#include <algorithm>
#include <cmath>
#include <format>

#include <RmlUi/Core/Context.h>
#include <RmlUi/Core/Event.h>
#include <RmlUi/Core/Input.h>
#include <RmlUi/Core/Mesh.h>
#include <RmlUi/Core/RenderManager.h>

namespace NoorRay::App {

namespace {

using Vec2 = Rml::Vector2f;

constexpr int leftButton = 0;
constexpr int rightButton = 1;
constexpr int middleButton = 2;
constexpr float minZoom = 0.1f;
constexpr float maxZoom = 3.0f;
constexpr float frameZoom = 1.0f;
//! The font size of .node in noorray.rcss, in dp; the canvas lengths are em of it, so the canvas font size zooms the nodes.
constexpr float nodeFontSize = 11.0f;
constexpr float wheelStep = 1.15f;
constexpr float frameMargin = 32.0f;
constexpr float linkWidth = 2.0f;
constexpr float handleLength = 40.0f;
constexpr float dragSlop = 3.0f;
constexpr int linkSegments = 24;
const Rml::Colourb frameColour{255, 255, 255, 13};
const Rml::Colourb boxFill{255, 255, 255, 16};
const Rml::Colourb boxEdge{255, 255, 255, 110};
const Rml::Colourb cutColour{255, 107, 107, 220};

void addQuad(Rml::Mesh& mesh, const Vec2 a, const Vec2 b, const Vec2 c, const Vec2 d, const Rml::ColourbPremultiplied colour) {
    const int first = static_cast<int>(mesh.vertices.size());
    for (const Vec2 point : {a, b, c, d})
        mesh.vertices.push_back({point, colour, {0.0f, 0.0f}});
    mesh.indices.insert(mesh.indices.end(), {first, first + 1, first + 2, first, first + 2, first + 3});
}

void addLine(Rml::Mesh& mesh, const Vec2 from, const Vec2 to, const float width, const Rml::ColourbPremultiplied colour) {
    const Vec2 along = to - from;
    const float length = along.Magnitude();
    if (length <= 0.0f)
        return;
    const Vec2 side = Vec2(-along.y, along.x) * (0.5f * width / length);
    addQuad(mesh, from + side, to + side, to - side, from - side, colour);
}

//! A link leaves its output and enters its input horizontally, as in Blender.
Vec2 curvePoint(const Vec2 from, const Vec2 to, const float scale, const float t) {
    const float handle = std::max(std::abs(to.x - from.x) * 0.5f, handleLength * scale);
    const Vec2 c1 = from + Vec2(handle, 0.0f);
    const Vec2 c2 = to - Vec2(handle, 0.0f);
    const float u = 1.0f - t;
    return from * (u * u * u) + c1 * (3.0f * u * u * t) + c2 * (3.0f * u * t * t) + to * (t * t * t);
}

void addCurve(Rml::Mesh& mesh, const Vec2 from, const Vec2 to, const float scale, const float width, const Rml::ColourbPremultiplied colour) {
    Vec2 previous = from;
    for (int i = 1; i <= linkSegments; ++i) {
        const Vec2 next = curvePoint(from, to, scale, static_cast<float>(i) / linkSegments);
        addLine(mesh, previous, next, width, colour);
        previous = next;
    }
}

float cross(const Vec2 a, const Vec2 b) {
    return a.x * b.y - a.y * b.x;
}

bool crosses(const Vec2 a, const Vec2 b, const Vec2 c, const Vec2 d) {
    const Vec2 ab = b - a;
    const Vec2 cd = d - c;
    const float denominator = cross(ab, cd);
    if (denominator == 0.0f)
        return false;
    const float t = cross(c - a, cd) / denominator;
    const float u = cross(c - a, ab) / denominator;
    return t >= 0.0f && t <= 1.0f && u >= 0.0f && u <= 1.0f;
}

Rml::Element* findAncestor(Rml::Element* target, const Rml::Element* graph, const char* attribute) {
    for (Rml::Element* element = target; element && element != graph; element = element->GetParentNode())
        if (element->HasAttribute(attribute))
            return element;
    return nullptr;
}

bool isInClass(Rml::Element* target, const Rml::Element* graph, const char* name) {
    for (Rml::Element* element = target; element && element != graph; element = element->GetParentNode())
        if (element->IsClassSet(name))
            return true;
    return false;
}

//! Presses on the fields of a node go to the field, not to the node.
bool isWidget(Rml::Element* target, const Rml::Element* node) {
    for (Rml::Element* element = target; element && element != node; element = element->GetParentNode()) {
        const Rml::String& tag = element->GetTagName();
        if (tag == "input" || tag == "select" || tag == "button" || element->IsClassSet("num") || element->IsClassSet("toggle")
            || element->IsClassSet("node-swatch"))
            return true;
    }
    return false;
}

bool isTyping(Rml::Context* context) {
    const Rml::Element* focus = context ? context->GetFocusElement() : nullptr;
    if (!focus || !focus->IsVisible(true))
        return false;
    const Rml::String& tag = focus->GetTagName();
    if (tag == "input")
        return focus->GetAttribute<Rml::String>("type", "text") == "text";
    return tag == "select" || tag == "textarea";
}

std::optional<NodeGraphSocket> socketOf(Rml::Element* target, const Rml::Element* graph) {
    Rml::Element* socket = findAncestor(target, graph, "socket");
    if (!socket)
        return std::nullopt;
    Rml::Element* node = findAncestor(socket, graph, "node");
    if (!node)
        return std::nullopt;
    return NodeGraphSocket{static_cast<std::size_t>(node->GetAttribute<int>("node", 0)), static_cast<std::uint32_t>(socket->GetAttribute<int>("socket", 0)),
                           socket->HasAttribute("output")};
}

}

//! The last child of a <nodegraph>; it renders after the nodes, so a dragged link shows above them.
class NodeGraphOverlay final : public Rml::Element {
public:
    explicit NodeGraphOverlay(NodeGraphElement& graph) : Rml::Element("div"), graph(graph) {}

protected:
    void OnRender() override {
        if (Rml::RenderManager* renderManager = GetRenderManager())
            graph.renderOverlay(*renderManager);
    }

private:
    NodeGraphElement& graph;
};

NodeGraphElement::NodeGraphElement(const Rml::String& tag, NodeGraphInstancer& instancer) : Rml::Element(tag), instancer(instancer) {
    AddEventListener(Rml::EventId::Mousedown, this);
    AddEventListener(Rml::EventId::Mousemove, this);
    AddEventListener(Rml::EventId::Mousescroll, this);
}

NodeGraphElement::~NodeGraphElement() {
    RemoveEventListener(Rml::EventId::Mousedown, this);
    RemoveEventListener(Rml::EventId::Mousemove, this);
    RemoveEventListener(Rml::EventId::Mousescroll, this);
    if (!pointerContext)
        return;
    pointerContext->RemoveEventListener("keydown", this, true);
    watchPointer(false);
}

void NodeGraphElement::ProcessEvent(Rml::Event& event) {
    const Vec2 mouse(event.GetParameter<float>("mouse_x", 0), event.GetParameter<float>("mouse_y", 0));
    switch (event.GetId()) {
    case Rml::EventId::Mousedown:
        press(event);
        return;
    case Rml::EventId::Mousemove:
        if (event.GetCurrentElement() == this && drag != Drag::None)
            return;
        move(mouse);
        return;
    case Rml::EventId::Mouseup:
        release(mouse);
        return;
    case Rml::EventId::Mousescroll:
        if (isInClass(event.GetTargetElement(), this, "node-menu"))
            return;
        zoomAt(mouse, event.GetParameter<float>("wheel_delta_y", 0));
        event.StopPropagation();
        return;
    case Rml::EventId::Keydown:
        key(event);
        return;
    default:
        return;
    }
}

void NodeGraphElement::press(Rml::Event& event) {
    const int button = event.GetParameter<int>("button", 0);
    Rml::Element* target = event.GetTargetElement();
    mouseAt = Vec2(event.GetParameter<float>("mouse_x", 0), event.GetParameter<float>("mouse_y", 0));
    const NodeGraphEvents& events = instancer.getEvents();
    if (drag == Drag::Grab) {
        endDrag(button == leftButton);
        return;
    }
    if (drag != Drag::None)
        return;
    if (isInClass(target, this, "node-menu"))
        return;
    menuFocusPending = false;
    if (events.closeMenu && events.closeMenu())
        return;
    pressedAt = mouseAt;
    panAtPress = pan;
    extend = event.GetParameter<int>("shift_key", 0) != 0;
    if (button == middleButton)
        return startDrag(Drag::Pan);
    if (button == rightButton && event.GetParameter<int>("ctrl_key", 0) != 0)
        return startDrag(Drag::Cut);
    if (button == rightButton && events.menu) {
        const float density = getDensity();
        menuFocusPending = true;
        events.menu(toCanvas(mouseAt), toLocal(mouseAt) / density, GetBox().GetSize(Rml::BoxArea::Content) / density);
        return;
    }
    if (button == leftButton)
        pressLeft(event, target);
}

void NodeGraphElement::pressLeft(Rml::Event& event, Rml::Element* target) {
    const NodeGraphEvents& events = instancer.getEvents();
    if (const std::optional<NodeGraphSocket> socket = socketOf(target, this)) {
        const std::optional<NodeGraphLinkStart> started = events.beginLink ? events.beginLink(*socket) : std::nullopt;
        if (!started)
            return;
        link = *started;
        event.StopPropagation();
        return startDrag(Drag::Link);
    }
    Rml::Element* node = findAncestor(target, this, "node");
    if (!node)
        return startDrag(Drag::Box);
    if (isWidget(target, node) || !events.press)
        return;
    pressedRow = static_cast<std::size_t>(node->GetAttribute<int>("node", 0));
    events.press(pressedRow, extend);
    startDrag(Drag::Node);
}

void NodeGraphElement::startDrag(const Drag started) {
    drag = started;
    moved = false;
    watchPointer(true);
}

void NodeGraphElement::startGrab() {
    pressedAt = mouseAt;
    startDrag(Drag::Grab);
    moved = true;
}

void NodeGraphElement::move(const Vec2 mouse) {
    mouseAt = mouse;
    const Vec2 offset = mouse - pressedAt;
    if (!moved && std::max(std::abs(offset.x), std::abs(offset.y)) > dragSlop)
        moved = true;
    switch (drag) {
    case Drag::Pan:
        pan = panAtPress + offset;
        applyTransform();
        return;
    case Drag::Node:
    case Drag::Grab:
        if (moved)
            moveNodes(mouse);
        return;
    case Drag::Box:
    case Drag::Link:
    case Drag::Cut:
        requestDraw();
        return;
    case Drag::None:
        return;
    }
}

void NodeGraphElement::moveNodes(const Vec2 mouse) {
    const NodeGraphEvents& events = instancer.getEvents();
    if (!events.drag || !canvas)
        return;
    if (rowElements.empty()) {
        for (int child = 0; child < canvas->GetNumChildren(); ++child) {
            Rml::Element* element = canvas->GetChild(child);
            if (!element->HasAttribute("node"))
                continue;
            const auto row = static_cast<std::size_t>(element->GetAttribute<int>("node", 0));
            if (row >= rowElements.size())
                rowElements.resize(row + 1, nullptr);
            rowElements[row] = element;
        }
    }
    for (const NodeGraphMoved& shown : events.drag((mouse - pressedAt) / (getDensity() * zoom))) {
        if (shown.row >= rowElements.size() || !rowElements[shown.row])
            continue;
        rowElements[shown.row]->SetProperty(Rml::PropertyId::Left, Rml::Property(shown.corner.x / nodeFontSize, Rml::Unit::EM));
        rowElements[shown.row]->SetProperty(Rml::PropertyId::Top, Rml::Property(shown.corner.y / nodeFontSize, Rml::Unit::EM));
    }
    requestDraw();
}

void NodeGraphElement::release(const Vec2 mouse) {
    mouseAt = mouse;
    const NodeGraphEvents& events = instancer.getEvents();
    const Drag ended = drag;
    if (ended == Drag::Grab)
        return;
    drag = Drag::None;
    watchPointer(false);
    rowElements.clear();
    requestDraw();
    switch (ended) {
    case Drag::Node:
        if (!moved && events.click)
            events.click(pressedRow, extend);
        else if (moved && events.drop)
            events.drop(true);
        return;
    case Drag::Box: {
        if (!events.boxSelect)
            return;
        const Vec2 a = toCanvas(pressedAt);
        const Vec2 b = toCanvas(mouse);
        const Vec2 low(std::min(a.x, b.x), std::min(a.y, b.y));
        const Vec2 size = moved ? Vec2(std::abs(b.x - a.x), std::abs(b.y - a.y)) : Vec2(0.0f, 0.0f);
        events.boxSelect({low, size}, extend);
        return;
    }
    case Drag::Link: {
        if (!events.endLink)
            return;
        Rml::Context* context = GetContext();
        const std::optional<NodeGraphSocket> target = context ? socketOf(context->GetHoverElement(), this) : std::nullopt;
        const float density = getDensity();
        menuFocusPending = events.endLink(target, toCanvas(mouse), toLocal(mouse) / density, GetBox().GetSize(Rml::BoxArea::Content) / density);
        return;
    }
    case Drag::Cut:
        cutLinks();
        return;
    case Drag::Pan:
    case Drag::Grab:
    case Drag::None:
        return;
    }
}

void NodeGraphElement::endDrag(const bool keep) {
    if (drag != Drag::Grab)
        return;
    drag = Drag::None;
    watchPointer(false);
    rowElements.clear();
    if (instancer.getEvents().drop)
        instancer.getEvents().drop(keep);
}

void NodeGraphElement::cutLinks() {
    const NodeGraphEvents& events = instancer.getEvents();
    if (!events.cut || !moved)
        return;
    const NodeGraphShapes& shapes = instancer.getShapes();
    const float scale = getDensity() * zoom;
    const Vec2 a = toLocal(pressedAt);
    const Vec2 b = toLocal(mouseAt);
    std::vector<std::size_t> cut;
    for (std::size_t index = 0; index < shapes.links.size(); ++index) {
        const Vec2 from = pan + shapes.links[index].from * scale;
        const Vec2 to = pan + shapes.links[index].to * scale;
        Vec2 previous = from;
        for (int i = 1; i <= linkSegments; ++i) {
            const Vec2 next = curvePoint(from, to, scale, static_cast<float>(i) / linkSegments);
            if (crosses(a, b, previous, next)) {
                cut.push_back(index);
                break;
            }
            previous = next;
        }
    }
    if (!cut.empty())
        events.cut(cut);
}

void NodeGraphElement::key(Rml::Event& event) {
    const int pressed = event.GetParameter<int>("key_identifier", Rml::Input::KI_UNKNOWN);
    if (menuKey(pressed)) {
        event.StopPropagation();
        return;
    }
    if (drag == Drag::Grab && pressed == Rml::Input::KI_ESCAPE)
        return endDrag(false);
    if (drag == Drag::Grab && (pressed == Rml::Input::KI_RETURN || pressed == Rml::Input::KI_NUMPADENTER))
        return endDrag(true);
    if (drag != Drag::None || isTyping(GetContext()))
        return;
    const bool control = event.GetParameter<int>("ctrl_key", 0) != 0 || event.GetParameter<int>("meta_key", 0) != 0;
    const bool shift = event.GetParameter<int>("shift_key", 0) != 0;
    const bool alt = event.GetParameter<int>("alt_key", 0) != 0;
    if (runCommand(pressed, control, shift, alt))
        event.StopPropagation();
}

bool NodeGraphElement::menuKey(const int pressed) {
    const NodeGraphEvents& events = instancer.getEvents();
    if (!events.menuKey)
        return false;
    switch (pressed) {
    case Rml::Input::KI_UP:
        return events.menuKey(NodeMenuKey::Up);
    case Rml::Input::KI_DOWN:
        return events.menuKey(NodeMenuKey::Down);
    case Rml::Input::KI_RETURN:
    case Rml::Input::KI_NUMPADENTER:
        return events.menuKey(NodeMenuKey::Enter);
    case Rml::Input::KI_ESCAPE:
        return events.menuKey(NodeMenuKey::Escape);
    default:
        return false;
    }
}

bool NodeGraphElement::runCommand(const int pressed, const bool control, const bool shift, const bool alt) {
    const NodeGraphEvents& events = instancer.getEvents();
    const auto run = [&](const NodeGraphCommand command) { return events.command && events.command(command, toCanvas(mouseAt)); };
    switch (pressed) {
    case Rml::Input::KI_HOME:
        frameAll();
        return true;
    case Rml::Input::KI_A:
        if (shift && !control && events.menu) {
            const float density = getDensity();
            menuFocusPending = true;
            events.menu(toCanvas(mouseAt), toLocal(mouseAt) / density, GetBox().GetSize(Rml::BoxArea::Content) / density);
            return true;
        }
        return run(alt ? NodeGraphCommand::SelectNone : NodeGraphCommand::SelectAll);
    case Rml::Input::KI_X:
        return run(control ? NodeGraphCommand::Cut : NodeGraphCommand::Delete);
    case Rml::Input::KI_DELETE:
        return run(NodeGraphCommand::Delete);
    case Rml::Input::KI_D:
        if (!shift || !run(NodeGraphCommand::Duplicate))
            return false;
        startGrab();
        return true;
    case Rml::Input::KI_G:
        startGrab();
        return true;
    case Rml::Input::KI_C:
        return control && run(NodeGraphCommand::Copy);
    case Rml::Input::KI_V:
        return control && run(NodeGraphCommand::Paste);
    case Rml::Input::KI_M:
        return run(NodeGraphCommand::Mute);
    case Rml::Input::KI_H:
        return run(NodeGraphCommand::Collapse);
    case Rml::Input::KI_Z:
        return control && run(shift ? NodeGraphCommand::Redo : NodeGraphCommand::Undo);
    case Rml::Input::KI_Y:
        return control && run(NodeGraphCommand::Redo);
    case Rml::Input::KI_TAB:
        return run(control ? NodeGraphCommand::Exit : NodeGraphCommand::Enter);
    default:
        return false;
    }
}

void NodeGraphElement::zoomAt(const Vec2 mouse, const float wheel) {
    const float zoomed = std::clamp(zoom * std::pow(wheelStep, -wheel), minZoom, maxZoom);
    const Vec2 local = toLocal(mouse);
    pan = local - (local - pan) * (zoomed / zoom);
    zoom = zoomed;
    applyTransform();
}

void NodeGraphElement::frameAll() {
    const NodeGraphShapes::Rectangle& bounds = instancer.getShapes().bounds;
    const Vec2 size = GetBox().GetSize(Rml::BoxArea::Content);
    const float density = getDensity();
    const Vec2 area = size - Vec2(2.0f * frameMargin, 2.0f * frameMargin);
    if (size.x <= 0.0f || size.y <= 0.0f || bounds.size.x <= 0.0f || bounds.size.y <= 0.0f) {
        zoom = frameZoom;
        pan = size * 0.5f - (bounds.position + bounds.size * 0.5f) * density;
        applyTransform();
        return;
    }
    zoom = std::clamp(std::min(area.x / (bounds.size.x * density), area.y / (bounds.size.y * density)), minZoom, frameZoom);
    pan = size * 0.5f - (bounds.position + bounds.size * 0.5f) * (density * zoom);
    applyTransform();
}

void NodeGraphElement::applyTransform() {
    if (!canvas)
        return;
    //! A whole pixel pan keeps the text on the pixel grid, and the font size lays the nodes out at the zoom, so the text is drawn at its shown size.
    pan = Vec2(std::round(pan.x), std::round(pan.y));
    canvas->SetProperty("transform", std::format("translate({}px, {}px)", pan.x, pan.y));
    canvas->SetProperty(Rml::PropertyId::FontSize, Rml::Property(nodeFontSize * zoom, Rml::Unit::DP));
    requestDraw();
}

void NodeGraphElement::watchPointer(const bool watched) {
    if (!pointerContext)
        return;
    if (!watched) {
        pointerContext->RemoveEventListener("mousemove", this);
        pointerContext->RemoveEventListener("mouseup", this);
        return;
    }
    pointerContext->AddEventListener("mousemove", this);
    pointerContext->AddEventListener("mouseup", this);
}

float NodeGraphElement::getDensity() const {
    const Rml::Context* context = GetContext();
    return context ? context->GetDensityIndependentPixelRatio() : 1.0f;
}

Vec2 NodeGraphElement::toLocal(const Vec2 mouse) {
    return mouse - GetAbsoluteOffset(Rml::BoxArea::Content);
}

Vec2 NodeGraphElement::toCanvas(const Vec2 mouse) {
    return (toLocal(mouse) - pan) / (getDensity() * zoom);
}

void NodeGraphElement::requestDraw() {
    if (Rml::Context* context = GetContext())
        context->RequestNextUpdate(0);
}

void NodeGraphElement::OnUpdate() {
    for (int child = 0; !canvas && child < GetNumChildren(); ++child) {
        if (!GetChild(child)->IsClassSet("node-canvas"))
            continue;
        canvas = GetChild(child);
        applyTransform();
    }
    if (canvas && !overlay) {
        Rml::ElementPtr added(new NodeGraphOverlay(*this));
        added->SetInstancer(&instancer);
        added->SetClass("node-overlay", true);
        overlay = AppendChild(std::move(added));
    }
    if (Rml::Context* context = GetContext(); context && !pointerContext) {
        pointerContext = context;
        //! The capture phase sees Enter and the arrow keys before the search field of the add menu takes them.
        context->AddEventListener("keydown", this, true);
    }
    //! A closed menu keeps its search field in the document, hidden, so the field gives up the focus here.
    if (Rml::Element* focus = GetContext() ? GetContext()->GetFocusElement() : nullptr;
        focus && focus->IsClassSet("node-menu-search") && !focus->IsVisible(true))
        focus->Blur();
    if (menuFocusPending) {
        if (Rml::Element* search = QuerySelector(".node-menu-search")) {
            menuFocusPending = false;
            search->Focus();
        }
    }
    const NodeGraphShapes& shapes = instancer.getShapes();
    if (shapes.view != shownView) {
        shownView = shapes.view;
        framePending = true;
    }
    if (!framePending || GetBox().GetSize(Rml::BoxArea::Content).x <= 0.0f)
        return;
    framePending = false;
    frameAll();
}

void NodeGraphElement::OnRender() {
    Rml::RenderManager* renderManager = GetRenderManager();
    if (!renderManager)
        return;
    const GeometryKey key{instancer.getShapes().revision, pan, zoom, getDensity()};
    if (geometryKey != key)
        buildGeometry(*renderManager, key);
    const Vec2 offset = GetAbsoluteOffset(Rml::BoxArea::Content);
    const Rml::Rectanglei previous = renderManager->GetScissorRegion();
    const Rml::Rectanglei content = Rml::Rectanglei::FromPositionSize(Rml::Vector2i(offset), Rml::Vector2i(GetBox().GetSize(Rml::BoxArea::Content)));
    renderManager->SetScissorRegion(content.IntersectIfValid(previous));
    if (geometry)
        geometry.Render(offset);
    if (previous.Valid())
        renderManager->SetScissorRegion(previous);
    else
        renderManager->DisableScissorRegion();
}

void NodeGraphElement::renderOverlay(Rml::RenderManager& renderManager) {
    if (drag != Drag::Link && drag != Drag::Box && drag != Drag::Cut)
        return;
    const Vec2 offset = GetAbsoluteOffset(Rml::BoxArea::Content);
    const float scale = getDensity() * zoom;
    const Vec2 mouse = toLocal(mouseAt);
    const Vec2 start = toLocal(pressedAt);
    Rml::Mesh mesh;
    if (drag == Drag::Link) {
        const Vec2 anchor = pan + link.anchor * scale;
        const float width = std::max(linkWidth * scale, 1.0f);
        if (link.fromOutput)
            addCurve(mesh, anchor, mouse, scale, width, link.colour.ToPremultiplied());
        else
            addCurve(mesh, mouse, anchor, scale, width, link.colour.ToPremultiplied());
    } else if (drag == Drag::Cut) {
        addLine(mesh, start, mouse, std::max(getDensity(), 1.0f) * 1.5f, cutColour.ToPremultiplied());
    } else if (moved) {
        const Vec2 low(std::min(start.x, mouse.x), std::min(start.y, mouse.y));
        const Vec2 high(std::max(start.x, mouse.x), std::max(start.y, mouse.y));
        const float edge = std::max(getDensity(), 1.0f);
        addQuad(mesh, low, {high.x, low.y}, high, {low.x, high.y}, boxFill.ToPremultiplied());
        addLine(mesh, low, {high.x, low.y}, edge, boxEdge.ToPremultiplied());
        addLine(mesh, {high.x, low.y}, high, edge, boxEdge.ToPremultiplied());
        addLine(mesh, high, {low.x, high.y}, edge, boxEdge.ToPremultiplied());
        addLine(mesh, {low.x, high.y}, low, edge, boxEdge.ToPremultiplied());
    }
    if (mesh.indices.empty())
        return;
    Rml::Geometry overlay = renderManager.MakeGeometry(std::move(mesh));
    overlay.Render(offset);
}

void NodeGraphElement::buildGeometry(Rml::RenderManager& renderManager, const GeometryKey& key) {
    const NodeGraphShapes& shapes = instancer.getShapes();
    const float scale = key.density * key.zoom;
    const auto toLocal = [&](const Vec2 point) { return key.pan + point * scale; };
    Rml::Mesh mesh;
    for (const NodeGraphShapes::Rectangle& frame : shapes.frames) {
        const Vec2 low = toLocal(frame.position);
        const Vec2 high = toLocal(frame.position + frame.size);
        addQuad(mesh, low, {high.x, low.y}, high, {low.x, high.y}, frameColour.ToPremultiplied());
    }
    const float width = std::max(linkWidth * scale, 1.0f);
    for (const NodeGraphShapes::Curve& link : shapes.links)
        addCurve(mesh, toLocal(link.from), toLocal(link.to), scale, width, link.colour.ToPremultiplied());
    geometry = mesh.indices.empty() ? Rml::Geometry() : renderManager.MakeGeometry(std::move(mesh));
    geometryKey = key;
}

Rml::ElementPtr NodeGraphInstancer::InstanceElement(Rml::Element*, const Rml::String& tag, const Rml::XMLAttributes&) {
    return Rml::ElementPtr(new NodeGraphElement(tag, *this));
}

void NodeGraphInstancer::ReleaseElement(Rml::Element* element) {
    delete element;
}

}
