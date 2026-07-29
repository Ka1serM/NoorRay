#pragma once

#include "UI/ImGuiComponent.h"
#include "UI/MaterialXNodes/MaterialXGraphNode.h"
#include "Scene/Scene.h"

#include <MaterialXCore/Document.h>
#include <imgui.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

class MaterialXNodeEditorPanel final : public ImGuiComponent
{
public:
    MaterialXNodeEditorPanel(std::string name, Scene& scene);
    void renderUi() override;

private:
    // Identity of the material currently being edited. Deliberately does not
    // depend on the document's contents: editing a value must not look like a
    // different document, or the graph would be rebuilt on every keystroke and
    // the view would jump back to its initial pan, zoom and selection.
    struct MaterialTarget
    {
        std::string key;         // Stable identity, e.g. "material:3"
        std::string sourcePath;  // Non-empty when backed by a .mtlx file
        MaterialX::DocumentPtr document; // The Scene's stored document, when not file-backed
        MaterialHandle handle;
        bool valid{};
    };

    MaterialTarget resolveTarget() const;
    void loadDocument(const MaterialTarget& target);
    // Writes the document back to its file or to the Scene. Only pass
    // invalidateMaterial=true for edits that change shading: layout-only
    // changes must not trigger a recompile.
    void persistDocument(const MaterialTarget& target, bool invalidateMaterial);

    void drawGraph(const MaterialTarget& target);
    // Reconciles the live node objects against the document instead of
    // rebuilding them: survivors are rebound and keep their position and
    // selection, only genuinely added or removed nodes change.
    void syncGraph();
    // Places the nodes that were created without a stored position using the
    // depth-based layout the MaterialX graph editor uses. Called after the flow
    // has drawn its first frame so the measured node sizes are available.
    void applyAutoLayout();
    void drawAddNodeMenu(const MaterialTarget& target);
    void drawParameterPane(const MaterialTarget& target);
    // Nodes reachable from the material root, followed by any other top-level
    // node in the document so freshly added, not-yet-connected nodes survive.
    std::vector<MaterialX::NodePtr> collectNodes() const;
    // Copies node positions the user dragged into the document's xpos/ypos
    // attributes, the same convention the MaterialX graph editor uses, so a
    // saved graph reopens with the layout it was left in.
    bool captureMovedPositions();

    Scene& scene;
    MaterialX::DocumentPtr document;
    std::string loadedKey;
    std::string loadError;
    // Document the editor last read from or wrote to the Scene. An incoming
    // document that differs from this (by identity) is an edit from outside
    // the editor (a Blender sync, say) and is the only in-place change that
    // forces a reload.
    MaterialX::DocumentPtr syncedDocument;

    // Created once and kept for the lifetime of the panel: it owns the pan and
    // zoom, and ImNodeFlow offers no way to restore them.
    std::unique_ptr<ImFlow::ImNodeFlow> flow;
    uint64_t documentRevision{};
    uint64_t graphRevision{~0ull};
    // Keyed by MaterialX node name, not by node pointer: the pointers are
    // replaced whenever the document is reparsed, the names are what identify
    // the same node across loads.
    std::unordered_map<std::string, std::shared_ptr<MaterialXGraphNode>> uiNodes;
    std::string connectionState;
    bool layoutDirty{};
    // Nodes that syncGraph created without a recorded xpos/ypos. They are parked
    // off-canvas until the first draw pass has measured them, then applyAutoLayout
    // gives each one a position and clears the list.
    std::vector<std::string> pendingLayoutNodes;

    std::string selectedNodeName;
    std::string nodeSearch;
    ImVec2 addNodePosition{};
    // The add menu is drawn from inside ImNodeFlow's update, so "is it open" is
    // tracked by whether it drew this frame rather than by an ImGui query.
    bool addMenuOpen{};
    bool addMenuDrawnThisFrame{};
    bool focusNodeSearch{};
    float graphRatio{0.66f};
};
