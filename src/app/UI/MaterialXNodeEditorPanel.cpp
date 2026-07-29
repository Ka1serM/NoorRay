#include "MaterialXNodeEditorPanel.h"

#include "MaterialX/MaterialXCompiler.h"
#include "Mesh/Assets/MeshAsset.h"
#include "Scene/MeshInstance.h"
#include "UI/MaterialXNodeCatalog.h"
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
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <cmath>
#include <unordered_set>

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
    ImGui::BeginChild("MaterialXParameterPane", ImVec2(0.0f, 0.0f), true);
    drawParameterPane(target);
    ImGui::EndChild();
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
        // Right-click opens the add menu, so panning uses the middle button.
        flow->getGrid().config().scroll_button = ImGuiMouseButton_Middle;
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
            layout.height = size.y > 1.0f ? size.y : 60.0f;
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

void MaterialXNodeEditorPanel::drawGraph(const MaterialTarget& target)
{
    if (graphRevision != documentRevision) {
        graphRevision = documentRevision;
        syncGraph();
    }

    // Bound to a copy: ImNodeFlow keeps the callback past the end of this
    // frame, and only calls it from inside update() below.
    addMenuDrawnThisFrame = false;
    flow->rightClickPopUpContent([this, target](ImFlow::BaseNode*) {
        drawAddNodeMenu(target);
    });
    flow->update();
    // The popup lives in ImNodeFlow's own ImGui context, so "it closed" is
    // simply "it did not draw this frame".
    if (!addMenuDrawnThisFrame)
        addMenuOpen = false;

    // Nodes created this frame without a stored position were parked off-canvas
    // by syncGraph; now that the flow has measured them, give them the layout
    // the MaterialX graph editor uses.
    if (!pendingLayoutNodes.empty())
        applyAutoLayout();

    selectedNodeName.clear();
    for (const auto& [nodeName, uiNode] : uiNodes)
        if (uiNode->isSelected())
            selectedNodeName = nodeName;

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
        for (const mx::InputPtr& input : materialNode->getInputs()) {
            if (!input)
                continue;
            // Inputs drawn inline by the node body have no pin, so their
            // connection state cannot be read back here and must be left alone.
            ImFlow::Pin* pin = uiNode->inPin(input->getName());
            if (!pin)
                continue;
            mx::NodePtr connected;
            if (const auto link = pin->getLink().lock()) {
                if (auto* source = dynamic_cast<MaterialXGraphNode*>(link->left()->getParent()))
                    connected = source->materialNode();
            }
            const mx::NodePtr old = input->getConnectedNode();
            if ((old ? old->getName() : "") != (connected ? connected->getName() : "")) {
                input->setConnectedNode(connected);
                changed = true;
            }
            newConnectionState += materialNode->getName() + ":" + input->getName() + "="
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

    addMenuDrawnThisFrame = true;
    if (!addMenuOpen) {
        addMenuOpen = true;
        focusNodeSearch = true;
        nodeSearch.clear();
        // The popup opens where the right-click happened, which is where the
        // new node should land.
        addNodePosition = flow->screen2grid(ImGui::GetMousePosOnOpeningCurrentPopup());
    }

    ImGui::TextDisabled("Add node");
    std::array<char, 128> buffer{};
    const size_t count = std::min(nodeSearch.size(), buffer.size() - 1);
    std::copy_n(nodeSearch.data(), count, buffer.data());
    ImGui::SetNextItemWidth(240.0f);
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
            persistDocument(target, true);
        }
        catch (const mx::Exception& error) {
            loadError = error.what();
        }
        ImGui::CloseCurrentPopup();
    };

    ImGui::BeginChild("MaterialXNodeList", ImVec2(280.0f, 320.0f));
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

    ImGui::Text("Node: %s (%s)", editable->getName().c_str(), editable->getCategory().c_str());
    for (const mx::InputPtr& input : editable->getInputs()) {
        if (!input)
            continue;
        if (input->getConnectedNode()) {
            ImGui::TextDisabled("%s  <-  %s", input->getName().c_str(),
                input->getConnectedNode()->getName().c_str());
            continue;
        }
        const std::string type = input->getType();
        const std::string value = input->getValueString();
        bool changed = false;
        if (type == "color3" || type == "color4") {
            float color[4] = {0.8f, 0.8f, 0.8f, 1.0f};
            std::sscanf(value.c_str(), "%f, %f, %f, %f", &color[0], &color[1], &color[2], &color[3]);
            changed = type == "color3" ? ImGui::ColorEdit3(input->getName().c_str(), color)
                                       : ImGui::ColorEdit4(input->getName().c_str(), color);
            if (changed) {
                std::string serialized = std::to_string(color[0]) + ", "
                    + std::to_string(color[1]) + ", " + std::to_string(color[2]);
                if (type == "color4") serialized += ", " + std::to_string(color[3]);
                input->setValueString(serialized);
            }
        } else if (type == "float" || type == "half" || type == "double") {
            float number = std::strtof(value.c_str(), nullptr);
            changed = ImGui::DragFloat(input->getName().c_str(), &number, 0.01f);
            if (changed)
                input->setValueString(std::to_string(number));
        } else {
            std::array<char, 512> buffer{};
            const size_t count = std::min(value.size(), buffer.size() - 1);
            std::copy_n(value.data(), count, buffer.data());
            changed = ImGui::InputText(input->getName().c_str(), buffer.data(), buffer.size(),
                ImGuiInputTextFlags_EnterReturnsTrue);
            if (changed) {
                try {
                    input->setValueString(buffer.data());
                } catch (const mx::Exception& error) {
                    loadError = error.what();
                }
            }
        }
        if (changed)
            persistDocument(target, true);
    }
}
