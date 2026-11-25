#pragma once

#include <cstdint>
#include <vector>

#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/ElementInstancer.h>
#include <RmlUi/Core/Geometry.h>

#include "NoorRay.Reconstruction/Trainer.h"

namespace NoorRay::Reconstruction {

struct ChartRange {
    float low = 0.0f;
    float high = 1.0f;
};

//! The loss of every trained iteration and the PSNR of every test evaluation since the training started.
class TrainingHistory {
public:
    //! Adds the values of a new iteration; an iteration below the last one starts a new history, as after a restart.
    void record(const TrainerStatus& status);
    void clear();

    bool isEmpty() const { return lossIterations.empty(); }
    std::uint64_t getRevision() const { return revision; }
    //! The iterations that the x axis shows: the planned count, or more when the training went on.
    float getIterationSpan() const;
    //! In log10 of the loss.
    ChartRange getLossRange() const;
    //! In dB; empty history gives the default range.
    ChartRange getPsnrRange() const;

    const std::vector<float>& getLossIterations() const { return lossIterations; }
    const std::vector<float>& getLogLoss() const { return logLoss; }
    const std::vector<float>& getPsnrIterations() const { return psnrIterations; }
    const std::vector<float>& getPsnr() const { return psnr; }

private:
    std::vector<float> lossIterations;
    std::vector<float> logLoss;
    std::vector<float> psnrIterations;
    std::vector<float> psnr;
    int planned = 0;
    std::uint64_t revision = 0;
};

//! <trainingchart> draws the loss of the training history on a log scale, with its spread per column, and the test PSNR on its own scale.
class TrainingChartElement final : public Rml::Element {
public:
    TrainingChartElement(const Rml::String& tag, const TrainingHistory& history);

protected:
    void OnUpdate() override;
    void OnRender() override;
    void OnResize() override { built = false; }

private:
    void build();

    const TrainingHistory& history;
    Rml::Geometry geometry;
    std::uint64_t builtRevision = 0;
    bool built = false;
};

class TrainingChartInstancer final : public Rml::ElementInstancer {
public:
    explicit TrainingChartInstancer(const TrainingHistory& history) : history(history) {}

    Rml::ElementPtr InstanceElement(Rml::Element* parent, const Rml::String& tag, const Rml::XMLAttributes& attributes) override;
    void ReleaseElement(Rml::Element* element) override;

private:
    const TrainingHistory& history;
};

}
