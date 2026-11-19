#include "NoorRay.App/UI/Controllers/TrainingController.h"

#include <filesystem>
#include <format>

#include "NoorRay.App/UI/Models/AppModel.h"
#include "RmlKit/UI/Backend/FileDialog.h"
#include "RmlKit/UI/Models/Model.h"

namespace NoorRay::App {

namespace {

Rml::String describeDataset(const std::filesystem::path& folder) {
    if (std::filesystem::exists(folder / "transforms_train.json") || std::filesystem::exists(folder / "transforms.json"))
        return "NeRF transforms";
    if (std::filesystem::exists(folder / "sparse"))
        return "COLMAP reconstruction";
    return "No COLMAP or transforms.json data found";
}

}

TrainingController::TrainingController(RmlKit::UI::Model& model, AppModel& app, SDL_Window* window) : model(model), app(app), window(window) {
    model.bindEvent("trainingChooseDataset", &TrainingController::onChooseDataset, this);
    model.bindEvent("trainingStart", &TrainingController::onStart, this);
    model.bindEvent("trainingStop", &TrainingController::onStop, this);
}

void TrainingController::onChooseDataset(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    RmlKit::UI::Model::requireNoArguments(arguments, "trainingChooseDataset");
    const std::filesystem::path location = app.training.datasetPath.empty() ? std::filesystem::path() : std::filesystem::path(app.training.datasetPath);
    RmlKit::UI::FileDialog::openFolder(window, RmlKit::UI::FolderKind::Export, location, [this](const std::filesystem::path& folder) {
        app.training.datasetPath = folder.string();
        app.training.datasetInfo = describeDataset(folder);
        model.dirty("training");
    });
}

void TrainingController::onStart(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    RmlKit::UI::Model::requireNoArguments(arguments, "trainingStart");
    model.report(RmlKit::UI::ReportLevel::Error, "training", "The Gaussian trainer is not part of this build yet.");
}

void TrainingController::onStop(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    RmlKit::UI::Model::requireNoArguments(arguments, "trainingStop");
    app.training.running = false;
    model.dirty("training");
}

}
