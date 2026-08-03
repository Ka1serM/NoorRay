#include "MaterialXNodeEditorPanel.h"

#include "Materials/MaterialX/MaterialXDocument.h"
#include "Geometry/Mesh/Assets/MeshAsset.h"
#include "Scene/Objects/MeshInstance.h"
#include "UI/MaterialXNodeCatalog.h"
#include "UI/MathInput.h"
#include "UI/MaterialXNodes/MaterialXGraphNodeRegistry.h"
#include "UI/MaterialXNodes/MaterialXGraphLayout.h"

#include <MaterialXCore/Definition.h>
#include <MaterialXCore/Node.h>
#include <MaterialXFormat/Util.h>
#include <MaterialXFormat/XmlIo.h>
#include <imgui.h>
#include <imgui_internal.h>
#include <ImNodeFlow.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <cmath>
#include <string>
#include <unordered_set>
#include <vector>

namespace mx = MaterialX;

namespace
{
constexpr const char* PositionXAttribute = "xpos";
constexpr const char* PositionYAttribute = "ypos";

bool readPosition(const mx::Node& node, ImVec2& position)
{
    if (!node.hasAttribute(PositionXAttribute) || !node.hasAttribute(PositionYAttribute))
        return false;
    position = ImVec2(
        std::strtof(node.getAttribute(PositionXAttribute).c_str(), nullptr),
        std::strtof(node.getAttribute(PositionYAttribute).c_str(), nullptr));
    return true;
}

void writePosition(mx::Node& node, const ImVec2& position)
{
    node.setAttribute(PositionXAttribute, std::to_string(position.x));
    node.setAttribute(PositionYAttribute, std::to_string(position.y));
}

bool positionsDiffer(const ImVec2& a, const ImVec2& b)
{
    return std::fabs(a.x - b.x) > 0.5f || std::fabs(a.y - b.y) > 0.5f;
}

std::vector<std::string> splitCommaSeparated(const std::string& value)
{
    std::vector<std::string> result;
    size_t start = 0;
    while (start < value.size()) {
        const size_t end = value.find(',', start);
        const size_t tokenEnd = end == std::string::npos ? value.size() : end;
        const size_t first = value.find_first_not_of(" \t\n\r", start);
        if (first != std::string::npos && first < tokenEnd) {
            const size_t last = value.find_last_not_of(" \t\n\r", tokenEnd - 1);
            result.emplace_back(value.substr(first, last - first + 1));
        }
        if (end == std::string::npos)
            break;
        start = end + 1;
    }
    return result;
}

template <size_t Count>
std::array<float, Count> parseFloatComponents(const std::string& value, const float fallback)
{
    std::array<float, Count> result;
    result.fill(fallback);

    const char* cursor = value.c_str();
    for (float& component : result) {
        char* end = nullptr;
        const float parsed = std::strtof(cursor, &end);
        if (end == cursor)
            break;
        component = parsed;
        cursor = end;
        while (*cursor == ',' || *cursor == ' ' || *cursor == '\t')
            ++cursor;
    }
    return result;
}

template <size_t Count>
std::string serializeFloatComponents(const std::array<float, Count>& components)
{
    std::string result;
    for (size_t index = 0; index < Count; ++index) {
        if (index != 0)
            result += ", ";
        result += std::to_string(components[index]);
    }
    return result;
}
}

MaterialXNodeEditorPanel::MaterialXNodeEditorPanel(std::string name, Scene& scene)
    : ImGuiComponent(std::move(name)), scene(scene)
{
}

// ── Material resolution and persistence ───────────────────────────────────────

MaterialXNodeEditorPanel::MaterialTarget MaterialXNodeEditorPanel::resolveTarget() const
{
    MaterialTarget target;
    const auto object = scene.getActiveObjectPtr();
    const auto* mesh = object ? dynamic_cast<const MeshInstance*>(object.get()) : nullptr;
    if (!mesh || !mesh->hasMeshAsset() || mesh->getMeshAsset().getMaterialCount() == 0)
        return target;

    const uint32_t slot = std::min(scene.getSelectedMaterialSlot(),
        static_cast<uint32_t>(mesh->getMeshAsset().getMaterialCount() - 1));
    target.handle = mesh->getMeshAsset().getMaterialHandle(slot);
    target.valid = true;

    const uint32_t index = target.handle.index();
    if (const auto& paths = scene.getMaterialXSourcePaths(); index < paths.size())
        target.sourcePath = paths[index];
    if (const auto& documents = scene.getMaterialXDocuments(); index < documents.size())
        target.document = documents[index];

    target.key = target.sourcePath.empty()
        ? "material:" + std::to_string(index)
        : "file:" + target.sourcePath;
    return target;
}

void MaterialXNodeEditorPanel::loadDocument(const MaterialTarget& target)
{
    loadedKey = target.key;
    document.reset();
    loadError.clear();
    // The node objects survive the reload; only the pointer-free selection
    // name is kept, and syncGraph() rebinds everything to the new document.
    syncedDocument = target.document;
    ++documentRevision;

    try {
        if (!target.sourcePath.empty()) {
            document = mx::createDocument();
            const mx::FileSearchPath searchPath(
                std::filesystem::path(target.sourcePath).parent_path().string());
            mx::readFromXmlFile(document, target.sourcePath, searchPath);
        } else if (target.document) {
            document = target.document->copy();
        } else {
            // A material slot with no authored document at all edits as the
            // default MaterialX material.
            document = nr::materialx::defaultMaterial();
        }
    }
    catch (const mx::Exception& error) {
        loadError = error.what();
        document.reset();
    }
}

void MaterialXNodeEditorPanel::persistDocument(
    const MaterialTarget& target, const bool invalidateMaterial)
{
    if (!document || !target.valid)
        return;
    try {
        if (!target.sourcePath.empty()) {
            mx::writeToXmlFile(document, target.sourcePath);
        } else {
            auto& documents = scene.getMaterialXDocuments();
            const uint32_t index = target.handle.index();
            if (documents.size() <= index)
                documents.resize(index + 1);
            documents[index] = document->copy();
            // Remember what we wrote so the next frame does not mistake our own
            // edit for an external one and reload the graph underneath the user.
            syncedDocument = documents[index];
        }
        if (invalidateMaterial)
            scene.invalidateMaterial(target.handle);
    }
    catch (const mx::Exception& error) {
        loadError = error.what();
    }
}

// ── Frame ─────────────────────────────────────────────────────────────────────

void MaterialXNodeEditorPanel::renderUi()
{
    ImGui::Begin(name.c_str());

    const MaterialTarget target = resolveTarget();
    if (!target.valid) {
        ImGui::TextDisabled("Select a mesh with a material to edit its MaterialX graph.");
        ImGui::End();
        return;
    }

    // Reload only when the selected material changes, or when the document was
    // rewritten by something other than this panel. Editing a value here must
    // never reload, or the graph view would reset on every change.
    const bool externallyChanged = target.sourcePath.empty()
        && target.document && target.document.get() != syncedDocument.get();
    // A document that failed to parse is not retried every frame; the error is
    // shown until the material changes or is rewritten from outside.
    const bool neverLoaded = !document && loadError.empty();
    if (target.key != loadedKey || externallyChanged || neverLoaded)
        loadDocument(target);

    if (!loadError.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1, 0.35f, 0.3f, 1));
        ImGui::TextWrapped("MaterialX load error: %s", loadError.c_str());
        ImGui::PopStyleColor();
        ImGui::End();
        return;
    }
    if (!document) {
        ImGui::End();
        return;
    }

    // The graph view is intentionally compact, but the properties view should
    // retain the same padding and spacing as the regular Details panel.
    const ImVec2 panelWindowPadding = ImGui::GetStyle().WindowPadding;
    const ImVec2 panelItemSpacing = ImGui::GetStyle().ItemSpacing;
    const ImVec2 panelFramePadding = ImGui::GetStyle().FramePadding;
    const ImVec4 panelBackground = ImGui::GetStyle().Colors[ImGuiCol_WindowBg];

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(2.0f, 2.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(2.0f, 2.0f));
    const float panelWidth = ImGui::GetContentRegionAvail().x;
    constexpr float splitterWidth = 2.0f;
    const float graphWidth = std::clamp(panelWidth * graphRatio, 180.0f,
        std::max(180.0f, panelWidth - splitterWidth - 220.0f));
    ImGui::BeginChild("MaterialXGraphPane", ImVec2(graphWidth, 0.0f), true);
    drawGraph(target);
    ImGui::EndChild();
    ImGui::SameLine();
    ImGui::InvisibleButton("MaterialXPaneSplitter", ImVec2(splitterWidth, -1.0f));
    if (ImGui::IsItemActive()) {
        graphRatio = std::clamp(graphRatio
            + ImGui::GetIO().MouseDelta.x / std::max(panelWidth, 1.0f), 0.2f, 0.8f);
    }
    if (ImGui::IsItemHovered() || ImGui::IsItemActive())
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
    ImGui::SameLine();

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, panelWindowPadding);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, panelItemSpacing);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, panelFramePadding);
    ImGui::PushStyleColor(ImGuiCol_ChildBg, panelBackground);
    ImGui::BeginChild("MaterialXParameterPane", ImVec2(0.0f, 0.0f), true);
    drawParameterPane(target);
    ImGui::EndChild();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(3);
    ImGui::PopStyleVar(3);
    ImGui::End();
}

// ── Graph ─────────────────────────────────────────────────────────────────────

std::vector<mx::NodePtr> MaterialXNodeEditorPanel::collectNodes() const
{
    std::vector<mx::NodePtr> nodes;
    std::unordered_set<const mx::Node*> seen;

    mx::NodePtr root;
    for (const mx::NodePtr& node : document->getNodes()) {
        if (node->getCategory() == "surfacematerial") {
            root = node;
            break;
        }
    }
    std::function<void(const mx::NodePtr&)> collect = [&](const mx::NodePtr& node) {
        if (!node || !seen.insert(node.get()).second || nodes.size() >= 256)
            return;
        nodes.push_back(node);
        for (const mx::InputPtr& input : node->getInputs())
            if (input)
                collect(input->getConnectedNode());
    };
    collect(root);
    // Nodes that are not reachable from the material root still belong to the
    // document: a node added from the menu has no connections yet, and would
    // otherwise vanish the moment the graph was rebuilt.
    for (const mx::NodePtr& node : document->getNodes())
        collect(node);
    return nodes;
}

void MaterialXNodeEditorPanel::syncGraph()
{
    if (!flow) {
        flow = std::make_unique<ImFlow::ImNodeFlow>("MaterialXGraph");
        // Tab opens the add menu (see drawGraph), which frees the right button
        // for panning.
        flow->getGrid().config().scroll_button = ImGuiMouseButton_Right;
        flow->droppedLinkPopUpContent([this](ImFlow::Pin* dragged) {
            if (!dragged)
                return;
            // ImNodeFlow invokes this while its internal drop popup is open.
            // Defer opening our ordinary, unscaled add popup until after
            // flow->update() returns.
            droppedLinkWantsAddMenu = true;
            ImGui::CloseCurrentPopup();
        });
    }

    const std::vector<mx::NodePtr> nodes = collectNodes();
    std::unordered_map<std::string, mx::NodePtr> wanted;
    for (const mx::NodePtr& node : nodes)
        wanted.emplace(node->getName(), node);
    pendingLayoutNodes.clear();

    // Retire the node objects whose MaterialX node is gone.
    for (auto entry = uiNodes.begin(); entry != uiNodes.end();) {
        if (wanted.contains(entry->first)) {
            ++entry;
            continue;
        }
        entry->second->destroy();
        entry = uiNodes.erase(entry);
    }

    for (const mx::NodePtr& node : nodes) {
        // A node that is already on screen keeps its object, and with it the
        // position the user dragged it to. Only the MaterialX node behind it is
        // swapped for the one from the newly parsed document.
        if (const auto existing = uiNodes.find(node->getName()); existing != uiNodes.end()) {
            existing->second->setMaterialNode(node);
            existing->second->rebuildPins();
            continue;
        }
        ImVec2 position(0.0f, 0.0f);
        // Nodes whose file records an xpos/ypos keep that arrangement; a saved
        // graph reopens with the layout it was left in.
        if (readPosition(*node, position)) {
            uiNodes.emplace(node->getName(), addMaterialXGraphNode(*flow, position, node));
            continue;
        }
        // Everything else is parked off-canvas and handed to applyAutoLayout
        // once this frame's draw pass has measured its size. The placement is
        // recorded on the node so a later drag can be told apart from it, but
        // it is not treated as an edit: merely opening a material must not
        // rewrite the user's .mtlx file.
        pendingLayoutNodes.push_back(node->getName());
        uiNodes.emplace(node->getName(),
            addMaterialXGraphNode(*flow, ImVec2(-10000.0f, -10000.0f), node));
    }

    // Links are rebuilt wholesale because rebuildPins() just replaced the pins
    // they pointed at.
    for (const auto& [nodeName, uiNode] : uiNodes)
        for (const std::shared_ptr<ImFlow::Pin>& pin : uiNode->getIns())
            pin->deleteLink();
    for (const mx::NodePtr& node : nodes) {
        const auto target = uiNodes.find(node->getName());
        if (target == uiNodes.end())
            continue;
        for (const mx::InputPtr& input : node->getInputs()) {
            if (!input || !input->getConnectedNode())
                continue;
            const auto source = uiNodes.find(input->getConnectedNode()->getName());
            if (source == uiNodes.end())
                continue;
            if (ImFlow::Pin* pin = target->second->inPin(input->getName()))
                source->second->outPin("out")->createLink(pin);
        }
    }

    // connectionState is left empty: the first sync pass after this seeds it
    // from the pins themselves, which is the only way to record exactly the
    // inputs that pass observes. Seeding it from the document here would list
    // inline inputs too and read as an edit on the very next frame.
    connectionState.clear();
}

bool MaterialXNodeEditorPanel::captureMovedPositions()
{
    bool moved = false;
    for (const auto& [nodeName, uiNode] : uiNodes) {
        ImVec2 stored;
        const ImVec2 current = uiNode->getPos();
        if (!readPosition(*uiNode->materialNode(), stored) || positionsDiffer(stored, current)) {
            writePosition(*uiNode->materialNode(), current);
            moved = true;
        }
    }
    return moved;
}

void MaterialXNodeEditorPanel::applyAutoLayout()
{
    const std::vector<mx::NodePtr> nodes = collectNodes();
    std::vector<MaterialXGraphLayout::LayoutNode> layoutNodes;
    std::vector<MaterialXGraphLayout::Link> links;
    layoutNodes.reserve(nodes.size());
    for (const mx::NodePtr& node : nodes) {
        MaterialXGraphLayout::LayoutNode layout;
        layout.name = node->getName();
        ImVec2 stored(0.0f, 0.0f);
        if (readPosition(*node, stored)) {
            layout.hasPosition = true;
            layout.position = {stored.x, stored.y};
        }
        if (const auto ui = uiNodes.find(node->getName()); ui != uiNodes.end()) {
            const ImVec2& size = ui->second->getSize();
            if (size.x > 1.0f)
                layout.width = size.x;
            if (size.y > 1.0f)
                layout.height = size.y;
        }
        layoutNodes.push_back(std::move(layout));
        for (const mx::InputPtr& input : node->getInputs()) {
            if (!input || !input->getConnectedNode())
                continue;
            links.emplace_back(input->getConnectedNode()->getName(), node->getName());
        }
    }

    const std::unordered_map<std::string, MaterialXGraphLayout::Vec2> positions =
        MaterialXGraphLayout::autoLayout(layoutNodes, links);
    for (const std::string& name : pendingLayoutNodes) {
        const auto ui = uiNodes.find(name);
        if (ui == uiNodes.end())
            continue;
        const auto placed = positions.find(name);
        // Defense in depth: every pending node is part of the collected graph,
        // so autoLayout always places it, but a parked off-canvas node is
        // strictly worse than a visible fallback.
        const MaterialXGraphLayout::Vec2 where = placed != positions.end()
            ? placed->second
            : MaterialXGraphLayout::Vec2{40.0f, 40.0f + static_cast<float>(layoutNodes.size()) * 100.0f};
        const ImVec2 position(where.x, where.y);
        ui->second->setPos(position);
        // Auto-placement is recorded so a later drag can be told apart from it
        // and the layout survives reopening the material, but it is not a
        // shading change.
        writePosition(*ui->second->materialNode(), position);
    }
    pendingLayoutNodes.clear();
}

void MaterialXNodeEditorPanel::copySelectedNode()
{
    const auto entry = uiNodes.find(selectedNodeName);
    if (entry == uiNodes.end() || !entry->second->materialNode())
        return;

    const mx::NodePtr& source = entry->second->materialNode();
    clipboardDocument = mx::createDocument();
    clipboardNode = clipboardDocument->addNode(
        source->getCategory(), source->getName(), source->getType());
    // copyContentFrom includes authored inputs, values, connection names and
    // editor attributes while keeping the clipboard independent of the live
    // document.
    clipboardNode->copyContentFrom(source);
}

void MaterialXNodeEditorPanel::pasteNode(const MaterialTarget& target)
{
    if (!document || !flow || !clipboardNode)
        return;

    const std::string baseName = clipboardNode->getName().empty()
        ? clipboardNode->getCategory() : clipboardNode->getName();
    const std::string name = document->createValidChildName(baseName + "_copy");
    const mx::NodePtr pasted = document->addNode(
        clipboardNode->getCategory(), name, clipboardNode->getType());
    pasted->copyContentFrom(clipboardNode);

    ImVec2 position;
    if (ImGui::IsMouseHoveringRect(canvasMin, canvasMax)) {
        position = flow->screen2grid(ImGui::GetMousePos());
    } else {
        ImVec2 sourcePosition(0.0f, 0.0f);
        if (readPosition(*clipboardNode, sourcePosition))
            position = sourcePosition + ImVec2(32.0f, 32.0f);
        else
            position = ImVec2(32.0f, 32.0f);
    }
    writePosition(*pasted, position);

    const std::shared_ptr<MaterialXGraphNode> created =
        addMaterialXGraphNode(*flow, position, pasted);
    for (auto& [nodeName, node] : uiNodes)
        node->selected(nodeName == name);
    uiNodes.emplace(name, created);
    selectedNodeName = name;
    connectionState.clear();

    // Rebuild only the live flow topology on the next frame. This reconnects
    // copied input links without resetting the viewport or the existing node
    // objects, and follows the same deferred-edit policy as the add menu.
    ++documentRevision;
    (void)target;
}

void MaterialXNodeEditorPanel::cutSelectedNode(const MaterialTarget& target)
{
    const auto entry = uiNodes.find(selectedNodeName);
    if (entry == uiNodes.end())
        return;

    copySelectedNode();
    const std::string name = entry->first;
    entry->second->destroy();
    uiNodes.erase(entry);
    document->removeNode(name);
    selectedNodeName.clear();
    connectionState.clear();
    persistDocument(target, true);
}

void MaterialXNodeEditorPanel::drawGraph(const MaterialTarget& target)
{
    if (graphRevision != documentRevision) {
        graphRevision = documentRevision;
        syncGraph();
    }

    // Keyboard operations are handled in the outer ImGui context after the
    // flow has finished drawing. This keeps shortcuts out of node-body editors
    // while still allowing the graph to use the normal Ctrl+C/V/X gestures.
    canvasMin = ImGui::GetWindowPos();
    canvasMax = canvasMin + ImGui::GetWindowSize();

    // ImNodeFlow renders through a contained ImGui context and copies that
    // context's IME state back to the outer one when update() returns. If an
    // outer text field is active (for example a transform value in Details),
    // an idle graph would otherwise clear its WantTextInput request and SDL3
    // would stop delivering character events. Give the already-active outer
    // editor priority while still allowing text editors inside the graph to
    // publish their IME state when the outer UI is not accepting text.
    ImGuiContext* const outerContext = ImGui::GetCurrentContext();
    const ImGuiPlatformImeData outerImeData = outerContext->PlatformImeData;
    flow->update();
    if (outerImeData.WantTextInput || outerImeData.WantVisible)
        outerContext->PlatformImeData = outerImeData;

    if (droppedLinkWantsAddMenu) {
        droppedLinkWantsAddMenu = false;
        addNodePosition = flow->screen2grid(ImGui::GetMousePos());
        addMenuScreenPosition = ImGui::GetMousePos();
        addMenuOpen = false;
        focusNodeSearch = true;
        ImGui::OpenPopup(AddNodePopupId);
    }

    // The add menu is deliberately not ImNodeFlow's right-click popup. That
    // callback runs inside the canvas's own ImGui context, whose draw data is
    // multiplied by the graph zoom on the way out (see context_wrapper.h's
    // AppendDrawData), so the search box grew and shrank with the view and was
    // unreadable at low zoom. Opened here instead, in the outer context after
    // update() has returned, it is an ordinary popup at UI size.
    // drawGraph runs inside the graph pane's child window, so this is the
    // canvas rect the menu has to stay inside of.
    canvasMin = ImGui::GetWindowPos();
    canvasMax = canvasMin + ImGui::GetWindowSize();

    const bool canvasActive = ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows)
        || ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows);
    if (canvasActive && !ImGui::IsPopupOpen(AddNodePopupId)
        && ImGui::IsKeyPressed(ImGuiKey_Tab, false)) {
        // Where the new node lands. screen2grid handles being called from
        // outside the canvas context, so the outer mouse position is correct
        // here.
        addNodePosition = flow->screen2grid(ImGui::GetMousePos());
        addMenuScreenPosition = ImGui::GetMousePos();
        addMenuOpen = false;
        ImGui::OpenPopup(AddNodePopupId);
    }

    // A popup is a top-level window, so it is not clipped by the child it was
    // opened over: left alone it spills across the parameter pane and out of
    // the panel entirely. Constrain it to the canvas instead -- cap the size,
    // then place it so the whole menu lands inside. addMenuSize is last
    // frame's measurement, which is exact after the first frame and only ever
    // used to nudge the position.
    const ImVec2 canvasSize = canvasMax - canvasMin;
    constexpr float canvasMargin = 8.0f;
    const ImVec2 maxMenuSize(
        std::max(canvasSize.x - 2.0f * canvasMargin, 120.0f),
        std::max(canvasSize.y - 2.0f * canvasMargin, 120.0f));
    ImGui::SetNextWindowSizeConstraints(ImVec2(0.0f, 0.0f), maxMenuSize);
    const ImVec2 menuSize(
        std::min(addMenuSize.x > 0.0f ? addMenuSize.x : maxMenuSize.x, maxMenuSize.x),
        std::min(addMenuSize.y > 0.0f ? addMenuSize.y : maxMenuSize.y, maxMenuSize.y));
    const ImVec2 clampedPosition(
        std::clamp(addMenuScreenPosition.x,
            canvasMin.x + canvasMargin,
            std::max(canvasMax.x - canvasMargin - menuSize.x, canvasMin.x + canvasMargin)),
        std::clamp(addMenuScreenPosition.y,
            canvasMin.y + canvasMargin,
            std::max(canvasMax.y - canvasMargin - menuSize.y, canvasMin.y + canvasMargin)));
    ImGui::SetNextWindowPos(clampedPosition);
    if (ImGui::BeginPopup(AddNodePopupId)) {
        drawAddNodeMenu(target);
        addMenuSize = ImGui::GetWindowSize();
        ImGui::EndPopup();
    } else {
        addMenuOpen = false;
    }

    // Nodes created this frame without a stored position were parked off-canvas
    // by syncGraph; now that the flow has measured them, give them the layout
    // the MaterialX graph editor uses.
    if (!pendingLayoutNodes.empty())
        applyAutoLayout();

    selectedNodeName.clear();
    for (const auto& [nodeName, uiNode] : uiNodes)
        if (uiNode->isSelected())
            selectedNodeName = nodeName;

    const bool graphFocused = ImGui::IsWindowFocused() && !ImGui::IsAnyItemActive();
    if (graphFocused && ImGui::IsKeyDown(ImGuiMod_Ctrl)) {
        if (ImGui::IsKeyPressed(ImGuiKey_C, false))
            copySelectedNode();
        else if (ImGui::IsKeyPressed(ImGuiKey_X, false))
            cutSelectedNode(target);
        else if (ImGui::IsKeyPressed(ImGuiKey_V, false))
            pasteNode(target);
    }

    // Dragging a node is a layout change only: record it, but do not recompile
    // the material and do not write the document on every mouse-move frame.
    if (captureMovedPositions())
        layoutDirty = true;

    bool changed = false;
    // ImNodeFlow removes a node from its registry after Delete is pressed.
    // Mirror that operation into the MaterialX document and let the normal
    // persistence path below serialize/recompile the edited graph.
    for (auto entry = uiNodes.begin(); entry != uiNodes.end();) {
        bool stillPresent = false;
        for (const auto& [uid, candidate] : flow->getNodes()) {
            if (candidate.get() == entry->second.get()) {
                stillPresent = true;
                break;
            }
        }
        if (stillPresent) {
            ++entry;
            continue;
        }
        if (document->getNode(entry->first)) {
            document->removeNode(entry->first);
            changed = true;
        }
        entry = uiNodes.erase(entry);
    }

    std::string newConnectionState;
    for (const auto& [nodeName, uiNode] : uiNodes) {
        const mx::NodePtr& materialNode = uiNode->materialNode();
        // Pins are built from exposedInputs(), which includes library defaults
        // that are not authored on the node yet. Iterate the same set here:
        // otherwise a link to a default input such as Disney's `ior` is drawn
        // but never serialized because materialNode->getInputs() is empty for
        // that port.
        for (const mx::InputPtr& declared : exposedInputs(materialNode)) {
            if (!declared)
                continue;
            const std::string& inputName = declared->getName();
            // Inputs drawn inline by the node body have no pin, so their
            // connection state cannot be read back here and must be left alone.
            ImFlow::Pin* pin = uiNode->findInputPin(inputName);
            if (!pin)
                continue;
            mx::NodePtr connected;
            if (const auto link = pin->getLink().lock()) {
                if (auto* source = dynamic_cast<MaterialXGraphNode*>(link->left()->getParent()))
                    connected = source->materialNode();
            }
            const mx::InputPtr authored = materialNode->getInput(inputName);
            const mx::NodePtr old = authored ? authored->getConnectedNode() : nullptr;
            if ((old ? old->getName() : "") != (connected ? connected->getName() : "")) {
                if (connected) {
                    const mx::InputPtr target = authored
                        ? authored : materialNode->addInput(inputName, declared->getType());
                    target->setConnectedNode(connected);
                } else if (authored) {
                    authored->setConnectedNode(nullptr);
                }
                changed = true;
            }
            newConnectionState += materialNode->getName() + ":" + inputName + "="
                + (connected ? connected->getName() : "") + ";";
        }
    }
    if (!connectionState.empty() && newConnectionState != connectionState)
        changed = true;
    connectionState = std::move(newConnectionState);

    if (changed) {
        layoutDirty = false;
        persistDocument(target, true);
    } else if (layoutDirty && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
        // Layout is persisted so a reopened graph keeps its arrangement, but it
        // is not a shading change, so the material is not invalidated.
        layoutDirty = false;
        persistDocument(target, false);
    }
}

// ── Add menu ──────────────────────────────────────────────────────────────────

void MaterialXNodeEditorPanel::drawAddNodeMenu(const MaterialTarget& target)
{
    const MaterialXNodeCatalog& catalog = MaterialXNodeCatalog::instance();
    if (!catalog.loadError().empty()) {
        ImGui::TextDisabled("Node library unavailable: %s", catalog.loadError().c_str());
        return;
    }

    if (!addMenuOpen) {
        addMenuOpen = true;
        focusNodeSearch = true;
        nodeSearch.clear();
    }

    // The menu is a search box, not a panel, and sat too large against the
    // graph. One factor drives text and both dimensions so it stays in
    // proportion; it is applied to the popup window only and reset below.
    ImGui::SetWindowFontScale(AddMenuScale);

    ImGui::TextDisabled("Add node");
    std::array<char, 128> buffer{};
    const size_t count = std::min(nodeSearch.size(), buffer.size() - 1);
    std::copy_n(nodeSearch.data(), count, buffer.data());
    ImGui::SetNextItemWidth(240.0f * AddMenuScale);
    if (focusNodeSearch) {
        // Typing must go straight to the search box, so focus is set on the
        // frame the popup first appears.
        ImGui::SetKeyboardFocusHere();
        focusNodeSearch = false;
    }
    if (ImGui::InputTextWithHint("##MaterialXNodeSearch", "Search nodes...",
            buffer.data(), buffer.size()))
        nodeSearch = buffer.data();
    ImGui::Separator();

    const auto addNode = [&](const MaterialXNodeType& type) {
        const mx::ConstNodeDefPtr definition = catalog.findNodeDef(type);
        if (!definition)
            return;
        try {
            const mx::NodePtr created = document->addNode(type.category,
                document->createValidChildName(type.category), type.outputType);
            // Copy the declared inputs and their defaults instead of importing
            // the whole standard library into the document, which would be
            // written out with every save.
            for (const mx::InputPtr& declared : definition->getActiveInputs()) {
                if (!declared)
                    continue;
                const mx::InputPtr input =
                    created->addInput(declared->getName(), declared->getType());
                if (declared->hasValueString())
                    input->setValueString(declared->getValueString());
            }
            writePosition(*created, addNodePosition);
            // Added straight to the live graph rather than by forcing a
            // rebuild: recreating the flow would throw away the pan, zoom and
            // selection the user is working with.
            uiNodes.emplace(created->getName(),
                addMaterialXGraphNode(*flow, addNodePosition, created));
            // The new node contributes inputs the recorded state does not know
            // about. Reseed it next frame so the diff does not read as an edit.
            connectionState.clear();
            // A newly-created node has no effect until it is connected. Keep
            // it in the editor's working document so it can be wired in, but
            // do not write it back to the scene/file or invalidate the
            // material yet. If the graph is reloaded before it is connected,
            // the unreferenced node is intentionally discarded.
        }
        catch (const mx::Exception& error) {
            loadError = error.what();
        }
        ImGui::CloseCurrentPopup();
    };

    // The list is what gives the menu its size, so it is what has to give when
    // the canvas is too small to hold the menu at its natural size. Bounding it
    // here keeps the whole menu visible instead of letting the window size
    // constraint clip the bottom of the list off.
    const ImVec2 canvasSize = canvasMax - canvasMin;
    const float listWidth = std::clamp(canvasSize.x - 48.0f, 140.0f, 280.0f * AddMenuScale);
    const float listHeight = std::clamp(canvasSize.y - 96.0f, 100.0f, 320.0f * AddMenuScale);
    ImGui::BeginChild("MaterialXNodeList", ImVec2(listWidth, listHeight));
    if (!nodeSearch.empty()) {
        const std::vector<const MaterialXNodeType*> matches = catalog.search(nodeSearch);
        if (matches.empty())
            ImGui::TextDisabled("No matching nodes");
        for (const MaterialXNodeType* type : matches) {
            ImGui::PushID(type);
            if (ImGui::Selectable(type->label.c_str()))
                addNode(*type);
            if (ImGui::IsItemHovered() && !type->documentation.empty())
                ImGui::SetTooltip("%s", type->documentation.c_str());
            ImGui::PopID();
        }
    } else {
        for (const std::string& group : catalog.groups()) {
            if (!ImGui::TreeNode(group.c_str()))
                continue;
            for (const MaterialXNodeType* type : catalog.inGroup(group)) {
                ImGui::PushID(type);
                if (ImGui::Selectable(type->label.c_str()))
                    addNode(*type);
                if (ImGui::IsItemHovered() && !type->documentation.empty())
                    ImGui::SetTooltip("%s", type->documentation.c_str());
                ImGui::PopID();
            }
            ImGui::TreePop();
        }
    }
    ImGui::EndChild();
    // Window font scale is sticky per window, so it has to be put back or the
    // next popup drawn under this name inherits it.
    ImGui::SetWindowFontScale(1.0f);
}

// ── Parameters ────────────────────────────────────────────────────────────────

void MaterialXNodeEditorPanel::drawParameterPane(const MaterialTarget& target)
{
    if (selectedNodeName.empty()) {
        ImGui::TextDisabled("Select a node to edit its inputs.");
        return;
    }

    const auto entry = uiNodes.find(selectedNodeName);
    if (entry == uiNodes.end())
        return;
    const mx::NodePtr& editable = entry->second->materialNode();

    // Keep the heading readable when the properties pane is narrow. The
    // details panel below uses the same table-based layout, which lets labels
    // and controls negotiate space instead of controls growing beyond the
    // pane or being clipped at its right edge.
    ImGui::TextWrapped("Node: %s (%s)", editable->getName().c_str(),
        editable->getCategory().c_str());
    ImGui::Separator();

    const ImGuiTableFlags tableFlags =
        ImGuiTableFlags_SizingStretchProp
        | ImGuiTableFlags_Resizable
        | ImGuiTableFlags_PadOuterX;
    if (!ImGui::BeginTable("NodeProperties", 2, tableFlags))
        return;

    ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthStretch, 0.42f);
    ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch, 0.58f);

    // The declared input list, not the authored one, and an input is written
    // onto the node lazily the first time it is edited (see exposedInputs).
    for (const mx::InputPtr& declaredInput : exposedInputs(editable)) {
        if (!declaredInput)
            continue;
        const std::string name = declaredInput->getName();
        // The authored input is the value actually in effect; the declared one
        // only supplies the name, type and default.
        const mx::InputPtr authored = editable->getInput(name);
        if (authored && authored->getConnectedNode()) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextWrapped("%s", name.c_str());
            ImGui::TableSetColumnIndex(1);
            ImGui::PushStyleColor(ImGuiCol_Text,
                ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
            ImGui::TextWrapped("Connected to %s",
                authored->getConnectedNode()->getName().c_str());
            ImGui::PopStyleColor();
            continue;
        }
        const std::string type = authored ? authored->getType() : declaredInput->getType();
        const std::string value = authored && authored->hasValueString()
            ? authored->getValueString()
            : declaredInput->getValueString();
        const std::vector<std::string> enumLabels =
            splitCommaSeparated(declaredInput->getAttribute("enum"));
        const std::vector<std::string> enumValues =
            splitCommaSeparated(declaredInput->getAttribute("enumvalues"));
        const auto serializedEnumValue = [&](const size_t index) {
            if (enumValues.size() == enumLabels.size())
                return enumValues[index];

            size_t componentCount = 1;
            if (type == "color2" || type == "vector2")
                componentCount = 2;
            else if (type == "color3" || type == "vector3")
                componentCount = 3;
            else if (type == "color4" || type == "vector4")
                componentCount = 4;

            if (enumValues.size() == enumLabels.size() * componentCount) {
                std::string result;
                for (size_t component = 0; component < componentCount; ++component) {
                    if (component != 0)
                        result += ", ";
                    result += enumValues[index * componentCount + component];
                }
                return result;
            }
            return enumLabels[index];
        };
        // Writes go to the node, never to the declaration, which is shared
        // library state owned by the catalog.
        const auto write = [&](const std::string& serialized) {
            const mx::InputPtr target = authored
                ? authored : editable->addInput(name, type);
            target->setValueString(serialized);
        };
        bool changed = false;
        ImGui::PushID(name.c_str());

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextWrapped("%s", name.c_str());
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-FLT_MIN);
        if (!enumLabels.empty()) {
            int currentIndex = -1;
            for (size_t index = 0; index < enumLabels.size(); ++index) {
                const std::string enumValue = serializedEnumValue(index);
                if (value == enumLabels[index] || value == enumValue) {
                    currentIndex = static_cast<int>(index);
                    break;
                }
            }

            const std::string preview = currentIndex >= 0
                ? enumLabels[static_cast<size_t>(currentIndex)]
                : value;
            if (ImGui::BeginCombo("##Enum", preview.c_str())) {
                for (size_t index = 0; index < enumLabels.size(); ++index) {
                    const bool selected = static_cast<int>(index) == currentIndex;
                    if (ImGui::Selectable(enumLabels[index].c_str(), selected)) {
                        write(serializedEnumValue(index));
                        changed = true;
                    }
                    if (selected)
                        ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
        } else if (type == "boolean") {
            bool enabled = value == "true" || value == "1";
            changed = ImGui::Checkbox("##Boolean", &enabled);
            if (changed)
                write(enabled ? "true" : "false");
        } else if (type == "color3" || type == "color4") {
            auto color = parseFloatComponents<4>(value, 0.8f);
            color[3] = value.empty() ? 1.0f : color[3];
            changed = type == "color3" ? ImGui::ColorEdit3("##Color", color.data())
                                       : ImGui::ColorEdit4("##Color", color.data());
            if (changed) {
                if (type == "color3")
                    write(serializeFloatComponents<3>({color[0], color[1], color[2]}));
                else
                    write(serializeFloatComponents<4>(color));
            }
        } else if (type == "vector2" || type == "color2") {
            auto vector = parseFloatComponents<2>(value, 0.0f);
            changed = ImGui::DragFloat2("##Vector2", vector.data(), 0.01f);
            if (changed)
                write(serializeFloatComponents(vector));
        } else if (type == "vector3") {
            auto vector = parseFloatComponents<3>(value, 0.0f);
            changed = MathInput::DragFloat3("##Vector3", vector.data(), 0.01f);
            if (changed)
                write(serializeFloatComponents(vector));
        } else if (type == "vector4") {
            auto vector = parseFloatComponents<4>(value, 0.0f);
            changed = ImGui::DragFloat4("##Vector4", vector.data(), 0.01f);
            if (changed)
                write(serializeFloatComponents(vector));
        } else if (type == "float" || type == "half" || type == "double") {
            float number = std::strtof(value.c_str(), nullptr);
            // Disney transmission is a weight, not an unrestricted scalar.
            // Keep invalid values out of the authored graph: changing a
            // MaterialX input invalidates the GPU program, which is an
            // especially expensive mistake for large Gaussian scenes.
            const bool unitInterval = name == "specTrans"
                || name == "transmission"
                || name == "transmission_weight";
            const float maximum = unitInterval ? 1.0f : 0.0f;
            changed = MathInput::DragFloat("##Number", &number, 0.01f,
                0.0f, maximum, "%.3f",
                unitInterval ? ImGuiSliderFlags_ClampOnInput : 0);
            if (changed)
                write(std::to_string(number));
        } else if (type == "integer") {
            int number = static_cast<int>(std::strtol(value.c_str(), nullptr, 10));
            changed = MathInput::DragInt("##Integer", &number, 1.0f);
            if (changed)
                write(std::to_string(number));
        } else {
            std::array<char, 512> buffer{};
            const size_t count = std::min(value.size(), buffer.size() - 1);
            std::copy_n(value.data(), count, buffer.data());
            changed = ImGui::InputText("##Text", buffer.data(), buffer.size(),
                ImGuiInputTextFlags_EnterReturnsTrue);
            if (changed) {
                try {
                    write(buffer.data());
                } catch (const mx::Exception& error) {
                    loadError = error.what();
                    changed = false;
                }
            }
        }
        ImGui::PopID();
        if (changed)
            persistDocument(target, true);
    }

    ImGui::EndTable();
}
