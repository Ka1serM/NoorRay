#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "DiffRenderer/Methods/Method.h"
#include "DiffRenderer/Scenes/Dataset.h"
#include "DiffRenderer/Training/Loss.h"
#include "DiffRenderer/Training/Targets.h"
#include "NoorRHI/NoorRHI.h"

namespace DiffRenderer::Training {

struct Budget {
    int iterations = 30000;
    //! The limit of the training time in seconds, without the evaluations; 0 sets no limit.
    double seconds = 0.0;
    //! Evaluates every n iterations on every testStride-th test view; 0 evaluates only at the end, which always uses every test view.
    int evaluateEvery = 0;
    std::uint32_t testStride = 1;
};

struct Evaluation {
    int iteration = 0;
    double trainSeconds = 0.0;
    double psnr = 0.0;
    double ssim = 0.0;
};

struct ViewResult {
    std::string name;
    double psnr = 0.0;
    double ssim = 0.0;
    double renderMilliseconds = 0.0;
};

//! What a comparison records for one method.
struct MethodResult {
    Methods::Method* method = nullptr;
    std::vector<float> losses;
    std::vector<float> stepMilliseconds;
    std::vector<Evaluation> evaluations;
    //! The final evaluation on every test view; its renders are in the output folder.
    std::vector<ViewResult> views;
    Methods::Statistics statistics;
    double openMilliseconds = 0.0;
    double trainSeconds = 0.0;
    bool finished = false;

    int getIterations() const { return static_cast<int>(losses.size()); }
    double getMedianStepMilliseconds() const;
    double getMeanRenderMilliseconds() const;
};

//! Trains several methods one after another on one dataset with the same budget, loss and view order, and evaluates them on the test views.
class Comparison {
public:
    //! Writes the references of the test views and the final renders of each method as PNG into the output folder, if it is not empty.
    [[nodiscard]] std::expected<void, NoorRHI::Error> open(NoorRHI::Device& device, const Scenes::Dataset& dataset, std::span<Methods::Method* const> methods,
                                                           const Budget& budget, const Methods::Settings& settings, const std::filesystem::path& output);
    //! Opens the next method, or runs one iteration of the current one with its evaluations, the final one included.
    [[nodiscard]] std::expected<void, NoorRHI::Error> step();
    [[nodiscard]] std::expected<void, NoorRHI::Error> runAll();

    bool isFinished() const { return current >= results.size(); }
    std::span<const MethodResult> getResults() const { return results; }
    std::size_t getCurrentIndex() const { return current; }
    const Scenes::Dataset& getDataset() const { return *dataset; }
    const Budget& getBudget() const { return budget; }
    const Methods::Settings& getSettings() const { return settings; }
    LossKind getLossKind() const { return loss.getKind(); }
    const std::filesystem::path& getOutput() const { return output; }
    //! The folder of the reference PNGs and of the render PNGs of a method, relative to the output folder.
    static std::filesystem::path getReferenceFolder() { return "references"; }
    static std::filesystem::path getRenderFolder(const Methods::Method& method);
    static std::filesystem::path getImageName(std::size_t testIndex);

private:
    [[nodiscard]] std::expected<void, NoorRHI::Error> openMethod();
    [[nodiscard]] std::expected<void, NoorRHI::Error> train();
    [[nodiscard]] std::expected<void, NoorRHI::Error> evaluate(bool final);
    [[nodiscard]] std::expected<void, NoorRHI::Error> writeReferences() const;
    std::uint32_t getTrainView(int iteration);

    NoorRHI::Device* device = nullptr;
    const Scenes::Dataset* dataset = nullptr;
    Budget budget;
    Methods::Settings settings;
    std::filesystem::path output;
    Loss loss;
    Targets targets;
    std::vector<MethodResult> results;
    std::size_t current = 0;
    bool methodOpen = false;
    std::vector<std::uint32_t> order;
    int orderEpoch = -1;
    Core::Image render;
};

}
