#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>
#include <utility>
#include <vector>

#include <RmlUi/Core/DataModelHandle.h>
#include <RmlUi/Core/Event.h>
#include <RmlUi/Core/Types.h>

#include "NoorRay/Core/Error.h"
#include "NoorRay/Materials/NodeTree.h"
#include "NoorRay/Session/Session.h"

namespace RmlKit::UI {
class Model;
}

namespace NoorRay::App {

class AppModel;
class NodeTreeEditor;

//! The parameter pane of the node editor: the name, the settings and the unlinked inputs of the active node.
class NodePanelController {
public:
    NodePanelController(RmlKit::UI::Model& model, AppModel& app, Session::Session& session, NodeTreeEditor& editor);
    NodePanelController(const NodePanelController&) = delete;
    NodePanelController& operator=(const NodePanelController&) = delete;

    //! Shows a node of the shown tree, or nothing for noNode; the caller marks the model dirty.
    void show(std::uint32_t shown);
    //! Opens the colour picker of an input of the shown node.
    void pick(std::uint32_t socket);

private:
    void onName(Rml::DataModelHandle, Rml::Event& event, const Rml::VariantList& arguments);
    void onLabel(Rml::DataModelHandle, Rml::Event& event, const Rml::VariantList& arguments);
    void onValue(Rml::DataModelHandle, Rml::Event& event, const Rml::VariantList& arguments);
    void onPick(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onColor(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onSetting(Rml::DataModelHandle, Rml::Event& event, const Rml::VariantList& arguments);
    void onStop(Rml::DataModelHandle, Rml::Event& event, const Rml::VariantList& arguments);
    void onStopAdd(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onStopRemove(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onStopPick(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onPoint(Rml::DataModelHandle, Rml::Event& event, const Rml::VariantList& arguments);
    void onPointAdd(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onPointRemove(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);

    void showSettings(const Materials::Node& shown);
    void showInputs(const Materials::NodeTree& tree, const Materials::Node& shown);
    const Materials::Node* findNode() const;
    Materials::NodeProperty findProperty(std::size_t setting) const;
    std::size_t settingArgument(const Rml::VariantList& arguments, const char* event) const;
    std::pair<std::size_t, std::size_t> pairArguments(const Rml::VariantList& arguments, const char* event, std::size_t count) const;
    void setNumbers(std::size_t setting, std::vector<float> numbers, const std::string& merge);
    void setProperty(Materials::NodeProperty property, const std::string& merge);
    void setInput(std::uint32_t socket, Core::float4 value, const std::string& merge);
    void report(const std::expected<void, Core::Error>& result);

    RmlKit::UI::Model& model;
    AppModel& app;
    Session::Session& session;
    NodeTreeEditor& editor;
    std::uint32_t node = Materials::noNode;
    //! The property name of each setting row; the image row has the name image.
    std::vector<std::string> settingNames;
    std::vector<Scene::TextureHandle> textures;
};

}
