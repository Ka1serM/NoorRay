#pragma once

#include <RmlUi/Core/DataModelHandle.h>
#include <RmlUi/Core/Types.h>

namespace NoorRay::App {

struct StatusModel {
    int samples = 0;
    int maxSamples = 0;
    bool complete = false;
    Rml::String frameTime;
    Rml::String framesPerSecond;
    //! What still compiles in the background; the frames render without it until then.
    Rml::String compiling;
    Rml::String resolution;
    bool loading = false;
    Rml::String loadingText;
    Rml::String error;

    int meshes = 0;
    int instances = 0;
    int materials = 0;
    int lights = 0;
    Rml::String gaussians;
    Rml::String meshMemory;
    Rml::String textureMemory;
};

void bind(Rml::DataModelConstructor& model, StatusModel& state);

}
