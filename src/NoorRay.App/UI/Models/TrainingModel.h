#pragma once

#include <vector>

#include <RmlUi/Core/DataModelHandle.h>
#include <RmlUi/Core/Types.h>

namespace NoorRay::App {

struct LossBar {
    float height = 0.0f;
};

struct TrainingModel {
    Rml::String datasetPath;
    Rml::String datasetInfo;
    int iterations = 30000;
    int iteration = 0;
    bool running = false;
    Rml::String loss;
    Rml::String psnr;
    std::vector<LossBar> lossBars;
};

void bind(Rml::DataModelConstructor& model, TrainingModel& state);

}
