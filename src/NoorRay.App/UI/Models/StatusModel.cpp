#include "NoorRay.App/UI/Models/StatusModel.h"

namespace NoorRay::App {

void bind(Rml::DataModelConstructor& model, StatusModel& state) {
    auto fields = model.RegisterStruct<StatusModel>();
    fields.RegisterMember("samples", &StatusModel::samples);
    fields.RegisterMember("maxSamples", &StatusModel::maxSamples);
    fields.RegisterMember("complete", &StatusModel::complete);
    fields.RegisterMember("frameTime", &StatusModel::frameTime);
    fields.RegisterMember("framesPerSecond", &StatusModel::framesPerSecond);
    fields.RegisterMember("compiling", &StatusModel::compiling);
    fields.RegisterMember("resolution", &StatusModel::resolution);
    fields.RegisterMember("loading", &StatusModel::loading);
    fields.RegisterMember("loadingText", &StatusModel::loadingText);
    fields.RegisterMember("error", &StatusModel::error);
    fields.RegisterMember("meshes", &StatusModel::meshes);
    fields.RegisterMember("instances", &StatusModel::instances);
    fields.RegisterMember("materials", &StatusModel::materials);
    fields.RegisterMember("lights", &StatusModel::lights);
    fields.RegisterMember("gaussians", &StatusModel::gaussians);
    fields.RegisterMember("meshMemory", &StatusModel::meshMemory);
    fields.RegisterMember("textureMemory", &StatusModel::textureMemory);
    model.Bind("status", &state);
}

}
