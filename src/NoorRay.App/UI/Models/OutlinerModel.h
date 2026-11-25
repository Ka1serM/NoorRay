#pragma once

#include <vector>

#include <RmlUi/Core/DataModelHandle.h>
#include <RmlUi/Core/Types.h>

#include "RmlKit/UI/Models/GridRow.h"

namespace NoorRay::App {

struct OutlinerRow {
    Rml::String name;
    Rml::String kind;
    Rml::String icon;
    int depth = 0;
    bool parent = false;
    bool expanded = false;
    bool selected = false;
    bool active = false;
    //! The dense scene material index of each subset of a mesh, or -1 for an empty slot.
    std::vector<int> slots;
};

//! A material tile of the drawer; index is the dense scene index.
struct MaterialCard {
    int index = -1;
    Rml::String name;
    bool selected = false;
    bool nodes = false;
};

struct OutlinerModel {
    std::vector<OutlinerRow> rows;
    Rml::String filter;
    Rml::String fileName;
    RmlKit::UI::GridModel<MaterialCard> materialGrid;
    //! How many materials are selected in the drawer.
    int selectedMaterials = 0;
    int dropRow = -1;
    int dropSlot = -1;
    Rml::String dropPlace;
};

void bind(Rml::DataModelConstructor& model, OutlinerModel& state);

}
