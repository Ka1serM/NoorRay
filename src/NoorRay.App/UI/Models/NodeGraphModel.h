#pragma once

#include <cstdint>
#include <vector>

#include <RmlUi/Core/DataModelHandle.h>
#include <RmlUi/Core/Types.h>

#include "NoorRay.App/UI/Models/NumberField.h"

namespace NoorRay::App {

//! A shown socket; editor names how an unlinked input shows its value: number, integer, toggle, color, vector or text, and empty for a name only.
struct NodeGraphSocketRow {
    std::uint32_t socket = 0;
    Rml::String name;
    Rml::String color;
    Rml::String editor;
    int components = 1;
    NumberField x = 0.0f;
    NumberField y = 0.0f;
    NumberField z = 0.0f;
    bool checked = false;
    Rml::String swatch;
    Rml::String text;
    //! The socket cannot take the link that the user drags.
    bool dim = false;
};

//! Positions and sizes are in dp of the graph canvas, whose y axis points down.
struct NodeGraphNodeRow {
    Rml::String title;
    Rml::String category;
    float x = 0.0f;
    float y = 0.0f;
    float width = 0.0f;
    float height = 0.0f;
    bool muted = false;
    bool hidden = false;
    bool reroute = false;
    bool group = false;
    bool selected = false;
    bool active = false;
    std::vector<NodeGraphSocketRow> outputs;
    std::vector<NodeGraphSocketRow> inputs;
};

struct NodeGraphFrameRow {
    Rml::String label;
    float x = 0.0f;
    float y = 0.0f;
    float width = 0.0f;
};

//! A category of the add menu when type is empty, and else a node type.
struct NodeMenuItem {
    Rml::String name;
    Rml::String type;
    Rml::String category;
};

//! The add menu; x and y are in dp of the graph area.
struct NodeMenuModel {
    bool open = false;
    float x = 0.0f;
    float y = 0.0f;
    Rml::String search;
    Rml::String category;
    std::vector<NodeMenuItem> items;
    int active = 0;
};

struct NodeOptionRow {
    Rml::String value;
    Rml::String name;
};

struct NodeRampStopRow {
    NumberField position = 0.0f;
    NumberField alpha = 1.0f;
    Rml::String swatch;
};

struct NodeCurvePointRow {
    NumberField x = 0.0f;
    NumberField y = 0.0f;
};

//! A node setting in the parameter pane; editor is enum, number, integer, toggle, text, ramp, curve or image.
struct NodeSettingRow {
    Rml::String label;
    Rml::String editor;
    std::vector<NodeOptionRow> options;
    Rml::String text;
    NumberField number = 0.0f;
    bool checked = false;
    std::vector<NodeRampStopRow> stops;
    Rml::String gradient;
    std::vector<NodeCurvePointRow> points;
};

//! The parameter pane of the active node.
struct NodePanelModel {
    bool shown = false;
    Rml::String name;
    Rml::String label;
    Rml::String type;
    std::vector<NodeSettingRow> settings;
    std::vector<NodeGraphSocketRow> inputs;
    //! The input or the ramp stop whose colour picker is open, or -1.
    int pickerInput = -1;
    int pickerSetting = -1;
    int pickerStop = -1;
    Rml::String pickerColor;
};

struct NodeGraphModel {
    Rml::String owner;
    bool hasOwner = false;
    bool hasTree = false;
    //! The names of the groups entered from the owner's tree.
    std::vector<Rml::String> path;
    std::vector<NodeGraphNodeRow> nodes;
    std::vector<NodeGraphFrameRow> frames;
    std::vector<Rml::String> warnings;
    Rml::String error;
    //! A message of the last editor action, such as a link that the tree refuses.
    Rml::String notice;
    bool canUndo = false;
    bool canRedo = false;
    NodeMenuModel menu;
    NodePanelModel panel;
};

void bind(Rml::DataModelConstructor& model, NodeGraphModel& graph);

}
