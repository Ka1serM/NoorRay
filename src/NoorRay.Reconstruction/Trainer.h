#pragma once

#include <condition_variable>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <stop_token>
#include <string>
#include <thread>
#include <variant>
#include <vector>

#include "DiffRenderer/Core/Image.h"
#include "DiffRenderer/Methods/Catalog.h"
#include "DiffRenderer/Scenes/Dataset.h"
#include "DiffRenderer/Training/Loss.h"
#include "DiffRenderer/Training/Targets.h"
#include "NoorRHI/NoorRHI.h"
#include "NoorRay/Scene/GaussianCloud.h"
#include "NoorRay/Scene/Mesh.h"
#include "NoorRay/Scene/Texture.h"

namespace NoorRay::Reconstruction {

enum class Stage : std::uint8_t { Empty, Colmap, Loading, Ready, Failed };

struct TrainerStatus {
    Stage stage = Stage::Empty;
    std::string message;
    float progress = 0.0f;
    std::string method;
    bool training = false;
    int iteration = 0;
    int iterations = 0;
    double loss = 0.0;
    //! Of the last test evaluation; zero before the first.
    double psnr = 0.0;
    double stepMilliseconds = 0.0;
    std::size_t primitives = 0;
    //! The method has Gaussians or a mesh to add to the scene.
    bool hasResult = false;
    bool resultIsMesh = false;
    //! A requested result is not ready to take yet.
    bool readingResult = false;
};

//! A render of the trainer in the RGBA8 of the display, so the UI thread only uploads it.
struct TrainerFrame {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<std::uint32_t> pixels;
};

//! A trained mesh in the space of the dataset, with the textures of its material: the base colour, then occlusion, roughness and metallic in rgb, then the normal map.
struct TrainerMesh {
    Scene::Mesh mesh;
    std::vector<Scene::Texture> textures;
    //! Without textures the vertex colours are the base colour, and these are the means of the vertices.
    float roughness = 0.5f;
    float metallic = 0.0f;
};

//! The Gaussians or the mesh in the space of the dataset, without a name.
struct TrainerResult {
    std::variant<Scene::GaussianCloud, TrainerMesh> object;
    int iteration = 0;
};

//! Turns a folder of photographs into a trained representation on its own thread, between the renders that the viewport asks for.
class Trainer {
public:
    Trainer() = default;
    ~Trainer() { close(); }
    Trainer(const Trainer&) = delete;
    Trainer& operator=(const Trainer&) = delete;

    void open(NoorRHI::Device& device, const std::filesystem::path& photos, std::string method, int iterations);
    void close();

    //! Opens the method anew, so its training starts again from the first iteration.
    void setMethod(std::string method, int iterations);
    void setTraining(bool training);
    //! The newest request replaces an older one that the thread has not rendered yet; while training, the thread renders the last camera again now and then.
    void requestFrame(const DiffRenderer::Scenes::Camera& camera);
    std::optional<TrainerFrame> takeFrame();
    //! The thread drops the request when the method has no result or has not trained yet.
    void requestResult();
    std::optional<TrainerResult> takeResult();
    TrainerStatus getStatus();
    //! Not null once the stage is Ready; the dataset does not change after that.
    const DiffRenderer::Scenes::Dataset* getDataset() const;

private:
    void run(std::stop_token token, std::filesystem::path photos);
    //! A NeRF synthetic folder loads directly; any other folder goes through runColmap first.
    std::expected<DiffRenderer::Scenes::Dataset, std::string> loadFolder(const std::stop_token& token, const std::filesystem::path& folder);
    bool load(std::stop_token token, const std::filesystem::path& photos);
    bool openMethod(const std::string& id, int iterations);
    bool step(int iterations);
    void evaluate();
    void readResult();
    bool render(const DiffRenderer::Scenes::Camera& camera);
    void fail(std::string message);
    void setStage(Stage stage, std::string message, float progress);

    NoorRHI::Device* device = nullptr;
    mutable std::mutex mutex;
    std::condition_variable_any wake;
    TrainerStatus status;
    std::string wantedMethod;
    int wantedIterations = 0;
    std::uint64_t methodRequest = 0;
    bool wantTraining = false;
    std::optional<DiffRenderer::Scenes::Camera> frameRequest;
    std::optional<TrainerFrame> frame;
    std::optional<TrainerResult> result;
    std::unique_ptr<DiffRenderer::Scenes::Dataset> dataset;

    //! The thread creates it and the close frees it, so the methods free their GPU memory while the device is open.
    std::unique_ptr<DiffRenderer::Methods::Catalog> catalog;
    DiffRenderer::Methods::Method* method = nullptr;
    std::optional<DiffRenderer::Training::Targets> targets;
    std::optional<DiffRenderer::Training::Loss> loss;
    std::vector<std::uint32_t> order;
    int iteration = 0;
    DiffRenderer::Core::Image rendered;

    std::jthread worker;
};

}
