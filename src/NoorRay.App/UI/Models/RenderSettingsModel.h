#pragma once

#include <RmlUi/Core/DataModelHandle.h>
#include <RmlUi/Core/Types.h>

#include "NoorRay.App/UI/Models/NumberField.h"

namespace NoorRay::App {

//! The render settings of the Blender add-on with its names and defaults; Render::RenderOptions maps them to the renderer.
struct RenderSettingsModel {
    //! raster shades the first hit with direct light only; pathTraced runs the full path tracer; the Render menu sets it.
    Rml::String mode = "pathTraced";
    //! While the camera moves the viewport renders at 1/motionDivisor of its resolution; 1 is off.
    int motionDivisor = 2;

    //! off, nrd or openDlssRr, as the denoiser of the add-on; a build without OpenDLSS-RR starts with nrd.
    Rml::String denoiser = "openDlssRr";
    //! False in a build without the OpenDLSS-RR plugin, such as the web build, which then hides DLSS-RR.
    bool openDlssRrBuilt = false;
    int maxFrames = 256;
    bool lightTree = true;
    bool visibilityCache = true;
    //! off, emissive or scattering; hidden in a build without Gaussian ray tracing.
    Rml::String gaussianShading = "emissive";
    bool gaussianTracingBuilt = false;
    int seed = 0;

    bool temporal = true;
    //! off, di (ReSTIR DI) or pt (ReSTIR PT).
    Rml::String reuse = "off";
    bool radianceCache = true;
    bool upscale = true;
    //! native, quality, balanced, performance or ultraPerformance.
    Rml::String upscaleQuality = "quality";
    NumberField sharpness = 0.2f;

    int maxBounces = 12;
    int diffuseBounces = 4;
    int glossyBounces = 4;
    int transmissionBounces = 12;
    int volumeBounces = 0;
    int transparentBounces = 8;
    NumberField clampDirect = 0.0f;
    NumberField clampIndirect = 10.0f;
    NumberField filterGlossy = 1.0f;

    NumberField filmExposure = 1.0f;
    NumberField pixelFilter = 1.5f;
};

void bind(Rml::DataModelConstructor& model, RenderSettingsModel& state);

}
