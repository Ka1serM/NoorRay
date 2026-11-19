#include "NoorRay.App/UI/Models/RenderSettingsModel.h"

namespace NoorRay::App {

void bind(Rml::DataModelConstructor& model, RenderSettingsModel& state) {
    auto fields = model.RegisterStruct<RenderSettingsModel>();
    fields.RegisterMember("mode", &RenderSettingsModel::mode);
    fields.RegisterMember("tracedCameraRays", &RenderSettingsModel::tracedCameraRays);
    fields.RegisterMember("pathReuse", &RenderSettingsModel::pathReuse);
    fields.RegisterMember("accumulate", &RenderSettingsModel::accumulate);
    fields.RegisterMember("maxSamples", &RenderSettingsModel::maxSamples);
    fields.RegisterMember("maxDepth", &RenderSettingsModel::maxDepth);
    fields.RegisterMember("rouletteStart", &RenderSettingsModel::rouletteStart);
    fields.RegisterMember("indirectClamp", &RenderSettingsModel::indirectClamp);
    fields.RegisterMember("gaussianShading", &RenderSettingsModel::gaussianShading);
    fields.RegisterMember("toneMapping", &RenderSettingsModel::toneMapping);
    fields.RegisterMember("exposure", &RenderSettingsModel::exposure);
    fields.RegisterMember("motionDivisor", &RenderSettingsModel::motionDivisor);
    model.Bind("renderSettings", &state);
}

}
