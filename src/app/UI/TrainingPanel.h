#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "Training/GaussianTrainer.h"
#include "Training/ColmapReconstruction.h"
#include "UI/ImGuiComponent.h"

class GaussianTrainRenderer;
class Raytracer;
class Scene;

class TrainingPanel : public ImGuiComponent
{
public:
    TrainingPanel(std::string name, Scene& scene, Raytracer& raytracer);
    ~TrainingPanel() override;

    void renderUi() override;
    bool tick();
    bool isRunning() const { return running; }

private:
    void selectImageDirectory();
    void reconstructAndBegin();
    void exportResult();

    Scene& scene;
    Raytracer& raytracer;
    std::unique_ptr<GaussianTrainRenderer> trainRenderer;
    std::unique_ptr<noorray::GaussianTrainer> trainer;
    noorray::GaussianTrainingConfig config;
    noorray::ReconstructionMapper reconstructionMapper{noorray::ReconstructionMapper::Global};
    noorray::GaussianTrainingStep lastStep;
    int targetIterations{30000};
    bool running{};
    bool stopped{};
    std::string imageDirectory;
    std::string status{"Choose an image folder to reconstruct and train."};
};
