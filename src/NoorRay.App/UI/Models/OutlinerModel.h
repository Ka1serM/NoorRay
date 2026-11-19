#pragma once

#include <vector>

#include <RmlUi/Core/DataModelHandle.h>
#include <RmlUi/Core/Types.h>

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
    //! The dense scene index of the instance's own material, or -1.
    int material = -1;
};

struct OutlinerModel {
    std::vector<OutlinerRow> rows;
    Rml::String filter;
    Rml::String fileName;
    std::vector<Rml::String> materials;
    int dropRow = -1;
    Rml::String dropPlace;
};

void bind(Rml::DataModelConstructor& model, OutlinerModel& state);

}
