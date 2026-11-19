#pragma once

#include <RmlUi/Core/DataModelHandle.h>
#include <RmlUi/Core/Types.h>

#include "NoorRay.App/UI/Models/NumberField.h"

namespace NoorRay::App {

struct RenderSettingsModel {
    //! raster shades the first hit with direct light only; pathTraced runs the full path tracer.
    Rml::String mode = "pathTraced";
    //! The path tracer traces the camera rays too unless this is cleared; then the rasterizer finds the first hit.
    bool tracedCameraRays = true;
    bool pathReuse = true;
    bool accumulate = true;
    int maxSamples = 0;
    int maxDepth = 16;
    int rouletteStart = 3;
    NumberField indirectClamp = 0.0f;
    Rml::String gaussianShading = "emissive";
    Rml::String toneMapping = "agx";
    NumberField exposure = 0.0f;
    //! While the camera moves the viewport renders at 1/motionDivisor of its resolution; 1 is off.
    int motionDivisor = 2;
};

void bind(Rml::DataModelConstructor& model, RenderSettingsModel& state);

}
