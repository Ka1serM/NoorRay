#pragma once

#include <vector>

#include <RmlUi/Core/DataModelHandle.h>
#include <RmlUi/Core/Types.h>

#include "NoorRay.App/UI/Models/NumberField.h"

namespace NoorRay::App {

struct MaterialParameterRow {
    Rml::String name;
    int components = 1;
    NumberField x = 0.0f;
    NumberField y = 0.0f;
    NumberField z = 0.0f;
    NumberField hue = 0.0f;
    NumberField saturation = 0.0f;
    NumberField value = 0.0f;
    Rml::String color;
    //! The colour at full saturation and value, for the end of the saturation bar.
    Rml::String hueColor;
    bool textureSlot = false;
    Rml::String texture;
    Rml::String channel = "r";
};

struct MaterialEditorModel {
    //! The dense scene index of the edited material, or -1.
    int sceneIndex = -1;
    Rml::String name;
    Rml::String alphaMode = "opaque";
    std::vector<MaterialParameterRow> parameters;
    int parameterIndex = 0;
    //! The colour parameter whose picker is open, or -1.
    int pickerParameter = -1;
};

void bind(Rml::DataModelConstructor& model, MaterialEditorModel& editor);

}
