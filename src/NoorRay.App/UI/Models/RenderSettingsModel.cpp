#include "NoorRay.App/UI/Models/RenderSettingsModel.h"

namespace NoorRay::App {

void bind(Rml::DataModelConstructor& model, RenderSettingsModel& state) {
    auto fields = model.RegisterStruct<RenderSettingsModel>();
    fields.RegisterMember("mode", &RenderSettingsModel::mode);
    fields.RegisterMember("motionDivisor", &RenderSettingsModel::motionDivisor);
    fields.RegisterMember("denoiser", &RenderSettingsModel::denoiser);
    fields.RegisterMember("openDlssRrBuilt", &RenderSettingsModel::openDlssRrBuilt);
    fields.RegisterMember("maxFrames", &RenderSettingsModel::maxFrames);
    fields.RegisterMember("lightTree", &RenderSettingsModel::lightTree);
    fields.RegisterMember("visibilityCache", &RenderSettingsModel::visibilityCache);
    fields.RegisterMember("gaussianShading", &RenderSettingsModel::gaussianShading);
    fields.RegisterMember("gaussianTracingBuilt", &RenderSettingsModel::gaussianTracingBuilt);
    fields.RegisterMember("seed", &RenderSettingsModel::seed);
    fields.RegisterMember("temporal", &RenderSettingsModel::temporal);
    fields.RegisterMember("reuse", &RenderSettingsModel::reuse);
    fields.RegisterMember("radianceCache", &RenderSettingsModel::radianceCache);
    fields.RegisterMember("upscale", &RenderSettingsModel::upscale);
    fields.RegisterMember("upscaleQuality", &RenderSettingsModel::upscaleQuality);
    fields.RegisterMember("sharpness", &RenderSettingsModel::sharpness);
    fields.RegisterMember("maxBounces", &RenderSettingsModel::maxBounces);
    fields.RegisterMember("diffuseBounces", &RenderSettingsModel::diffuseBounces);
    fields.RegisterMember("glossyBounces", &RenderSettingsModel::glossyBounces);
    fields.RegisterMember("transmissionBounces", &RenderSettingsModel::transmissionBounces);
    fields.RegisterMember("volumeBounces", &RenderSettingsModel::volumeBounces);
    fields.RegisterMember("transparentBounces", &RenderSettingsModel::transparentBounces);
    fields.RegisterMember("clampDirect", &RenderSettingsModel::clampDirect);
    fields.RegisterMember("clampIndirect", &RenderSettingsModel::clampIndirect);
    fields.RegisterMember("filterGlossy", &RenderSettingsModel::filterGlossy);
    fields.RegisterMember("filmExposure", &RenderSettingsModel::filmExposure);
    fields.RegisterMember("pixelFilter", &RenderSettingsModel::pixelFilter);
    model.Bind("renderSettings", &state);
}

}
