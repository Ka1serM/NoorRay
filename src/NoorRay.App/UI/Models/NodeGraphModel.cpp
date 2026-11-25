#include "NoorRay.App/UI/Models/NodeGraphModel.h"

namespace NoorRay::App {

namespace {

void bindPanel(Rml::DataModelConstructor& model) {
    if (auto handle = model.RegisterStruct<NodeOptionRow>()) {
        handle.RegisterMember("value", &NodeOptionRow::value);
        handle.RegisterMember("name", &NodeOptionRow::name);
    }
    model.RegisterArray<std::vector<NodeOptionRow>>();
    if (auto handle = model.RegisterStruct<NodeRampStopRow>()) {
        handle.RegisterMember("position", &NodeRampStopRow::position);
        handle.RegisterMember("alpha", &NodeRampStopRow::alpha);
        handle.RegisterMember("swatch", &NodeRampStopRow::swatch);
    }
    model.RegisterArray<std::vector<NodeRampStopRow>>();
    if (auto handle = model.RegisterStruct<NodeCurvePointRow>()) {
        handle.RegisterMember("x", &NodeCurvePointRow::x);
        handle.RegisterMember("y", &NodeCurvePointRow::y);
    }
    model.RegisterArray<std::vector<NodeCurvePointRow>>();
    if (auto handle = model.RegisterStruct<NodeSettingRow>()) {
        handle.RegisterMember("label", &NodeSettingRow::label);
        handle.RegisterMember("editor", &NodeSettingRow::editor);
        handle.RegisterMember("options", &NodeSettingRow::options);
        handle.RegisterMember("text", &NodeSettingRow::text);
        handle.RegisterMember("number", &NodeSettingRow::number);
        handle.RegisterMember("checked", &NodeSettingRow::checked);
        handle.RegisterMember("stops", &NodeSettingRow::stops);
        handle.RegisterMember("gradient", &NodeSettingRow::gradient);
        handle.RegisterMember("points", &NodeSettingRow::points);
    }
    model.RegisterArray<std::vector<NodeSettingRow>>();
    if (auto handle = model.RegisterStruct<NodePanelModel>()) {
        handle.RegisterMember("shown", &NodePanelModel::shown);
        handle.RegisterMember("name", &NodePanelModel::name);
        handle.RegisterMember("label", &NodePanelModel::label);
        handle.RegisterMember("type", &NodePanelModel::type);
        handle.RegisterMember("settings", &NodePanelModel::settings);
        handle.RegisterMember("inputs", &NodePanelModel::inputs);
        handle.RegisterMember("pickerInput", &NodePanelModel::pickerInput);
        handle.RegisterMember("pickerSetting", &NodePanelModel::pickerSetting);
        handle.RegisterMember("pickerStop", &NodePanelModel::pickerStop);
        handle.RegisterMember("pickerColor", &NodePanelModel::pickerColor);
    }
}

void bindMenu(Rml::DataModelConstructor& model) {
    if (auto handle = model.RegisterStruct<NodeMenuItem>()) {
        handle.RegisterMember("name", &NodeMenuItem::name);
        handle.RegisterMember("type", &NodeMenuItem::type);
        handle.RegisterMember("category", &NodeMenuItem::category);
    }
    model.RegisterArray<std::vector<NodeMenuItem>>();
    if (auto handle = model.RegisterStruct<NodeMenuModel>()) {
        handle.RegisterMember("open", &NodeMenuModel::open);
        handle.RegisterMember("x", &NodeMenuModel::x);
        handle.RegisterMember("y", &NodeMenuModel::y);
        handle.RegisterMember("search", &NodeMenuModel::search);
        handle.RegisterMember("category", &NodeMenuModel::category);
        handle.RegisterMember("items", &NodeMenuModel::items);
        handle.RegisterMember("active", &NodeMenuModel::active);
    }
}

}

void bind(Rml::DataModelConstructor& model, NodeGraphModel& graph) {
    if (auto handle = model.RegisterStruct<NodeGraphSocketRow>()) {
        handle.RegisterMember("socket", &NodeGraphSocketRow::socket);
        handle.RegisterMember("name", &NodeGraphSocketRow::name);
        handle.RegisterMember("color", &NodeGraphSocketRow::color);
        handle.RegisterMember("editor", &NodeGraphSocketRow::editor);
        handle.RegisterMember("components", &NodeGraphSocketRow::components);
        handle.RegisterMember("x", &NodeGraphSocketRow::x);
        handle.RegisterMember("y", &NodeGraphSocketRow::y);
        handle.RegisterMember("z", &NodeGraphSocketRow::z);
        handle.RegisterMember("checked", &NodeGraphSocketRow::checked);
        handle.RegisterMember("swatch", &NodeGraphSocketRow::swatch);
        handle.RegisterMember("text", &NodeGraphSocketRow::text);
        handle.RegisterMember("dim", &NodeGraphSocketRow::dim);
    }
    model.RegisterArray<std::vector<NodeGraphSocketRow>>();
    if (auto handle = model.RegisterStruct<NodeGraphNodeRow>()) {
        handle.RegisterMember("title", &NodeGraphNodeRow::title);
        handle.RegisterMember("category", &NodeGraphNodeRow::category);
        handle.RegisterMember("x", &NodeGraphNodeRow::x);
        handle.RegisterMember("y", &NodeGraphNodeRow::y);
        handle.RegisterMember("width", &NodeGraphNodeRow::width);
        handle.RegisterMember("height", &NodeGraphNodeRow::height);
        handle.RegisterMember("muted", &NodeGraphNodeRow::muted);
        handle.RegisterMember("hidden", &NodeGraphNodeRow::hidden);
        handle.RegisterMember("reroute", &NodeGraphNodeRow::reroute);
        handle.RegisterMember("group", &NodeGraphNodeRow::group);
        handle.RegisterMember("selected", &NodeGraphNodeRow::selected);
        handle.RegisterMember("active", &NodeGraphNodeRow::active);
        handle.RegisterMember("outputs", &NodeGraphNodeRow::outputs);
        handle.RegisterMember("inputs", &NodeGraphNodeRow::inputs);
    }
    model.RegisterArray<std::vector<NodeGraphNodeRow>>();
    if (auto handle = model.RegisterStruct<NodeGraphFrameRow>()) {
        handle.RegisterMember("label", &NodeGraphFrameRow::label);
        handle.RegisterMember("x", &NodeGraphFrameRow::x);
        handle.RegisterMember("y", &NodeGraphFrameRow::y);
        handle.RegisterMember("width", &NodeGraphFrameRow::width);
    }
    model.RegisterArray<std::vector<NodeGraphFrameRow>>();
    bindMenu(model);
    bindPanel(model);
    auto fields = model.RegisterStruct<NodeGraphModel>();
    fields.RegisterMember("owner", &NodeGraphModel::owner);
    fields.RegisterMember("hasOwner", &NodeGraphModel::hasOwner);
    fields.RegisterMember("hasTree", &NodeGraphModel::hasTree);
    fields.RegisterMember("path", &NodeGraphModel::path);
    fields.RegisterMember("nodes", &NodeGraphModel::nodes);
    fields.RegisterMember("frames", &NodeGraphModel::frames);
    fields.RegisterMember("warnings", &NodeGraphModel::warnings);
    fields.RegisterMember("error", &NodeGraphModel::error);
    fields.RegisterMember("notice", &NodeGraphModel::notice);
    fields.RegisterMember("canUndo", &NodeGraphModel::canUndo);
    fields.RegisterMember("canRedo", &NodeGraphModel::canRedo);
    fields.RegisterMember("menu", &NodeGraphModel::menu);
    fields.RegisterMember("panel", &NodeGraphModel::panel);
    model.Bind("nodeGraph", &graph);
}

}
