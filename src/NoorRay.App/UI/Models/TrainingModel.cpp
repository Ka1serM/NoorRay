#include "NoorRay.App/UI/Models/TrainingModel.h"

namespace NoorRay::App {

void bind(Rml::DataModelConstructor& model, TrainingModel& state) {
    if (auto handle = model.RegisterStruct<LossBar>())
        handle.RegisterMember("height", &LossBar::height);
    model.RegisterArray<std::vector<LossBar>>();
    auto fields = model.RegisterStruct<TrainingModel>();
    fields.RegisterMember("datasetPath", &TrainingModel::datasetPath);
    fields.RegisterMember("datasetInfo", &TrainingModel::datasetInfo);
    fields.RegisterMember("iterations", &TrainingModel::iterations);
    fields.RegisterMember("iteration", &TrainingModel::iteration);
    fields.RegisterMember("running", &TrainingModel::running);
    fields.RegisterMember("loss", &TrainingModel::loss);
    fields.RegisterMember("psnr", &TrainingModel::psnr);
    fields.RegisterMember("lossBars", &TrainingModel::lossBars);
    model.Bind("training", &state);
}

}
