#include "NoorRay.App/Plugins/Plugin.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>
#include <optional>
#include <variant>
#include <vector>

#include <RmlUi/Core/Context.h>
#include <RmlUi/Core/DataModelHandle.h>
#include <RmlUi/Core/ElementDocument.h>
#include <RmlUi/Core/Factory.h>
#include <SDL3/SDL_video.h>

#include "NoorRay.Reconstruction/Alignment.h"
#include "NoorRay.Reconstruction/Datasets.h"
#include "NoorRay.Reconstruction/Source.h"
#include "NoorRay.Reconstruction/Trainer.h"
#include "NoorRay.Reconstruction/TrainingChart.h"
#include "NoorRay/Gpu/GaussianAccelerationStructure.h"
#include "RmlKit/UI/Backend/FileDialog.h"
#include "RmlKit/UI/Models/Model.h"

namespace NoorRay::Reconstruction {

namespace {

constexpr const char* documentPath = "UI/Reconstruction/reconstruction.rml";
//! The path tracer starts again with each refresh and builds the acceleration structure of the cloud.
constexpr std::chrono::seconds refreshInterval{1};
//! A mesh refresh reads the textures back and uploads them again.
constexpr std::chrono::seconds meshRefreshInterval{10};
constexpr const char* source = "Differentiable Rendering";

struct MethodRow {
    Rml::String id;
    Rml::String name;
    Rml::String renderer;
};

struct DatasetRow {
    Rml::String group;
    Rml::String name;
    Rml::String kind;
    Rml::String folder;
};

struct Model {
    Rml::String datasetsRoot = NOORRAY_DATASETS;
    std::vector<DatasetRow> datasets;
    Rml::String folder;
    Rml::String stage;
    Rml::String message;
    float progress = 0.0f;
    std::vector<MethodRow> methods;
    Rml::String method = "3dgs";
    int iterations = 30000;
    int iteration = 0;
    Rml::String loss;
    Rml::String psnr;
    //! The axis labels of the training chart.
    Rml::String lossHigh;
    Rml::String lossLow;
    Rml::String psnrHigh;
    Rml::String psnrLow;
    Rml::String iterationSpan;
    bool hasHistory = false;
    bool hasPsnr = false;
    Rml::String speed;
    Rml::String primitives;
    bool busy = false;
    bool ready = false;
    bool trained = false;
    bool training = false;
    bool failed = false;
    bool hasResult = false;
    bool resultIsMesh = false;
    bool added = false;
    bool adding = false;
    bool live = false;
    bool previewing = false;
    int view = 0;
    int viewCount = 0;
};

Rml::String getStageName(const Stage stage) {
    switch (stage) {
    case Stage::Empty:
        return "No dataset loaded";
    case Stage::Colmap:
        return "COLMAP";
    case Stage::Loading:
        return "Loading";
    case Stage::Ready:
        return "Ready";
    case Stage::Failed:
        return "Failed";
    }
    return {};
}

Rml::String formatCount(const std::size_t count) {
    if (count >= 1'000'000)
        return std::format("{:.2f} M", static_cast<double>(count) / 1e6);
    if (count >= 1'000)
        return std::format("{:.1f} k", static_cast<double>(count) / 1e3);
    return std::format("{}", count);
}

class ReconstructionPlugin final : public App::Plugin {
public:
    std::string_view getName() const override { return "Differentiable Rendering"; }
    App::PluginWindowSize getWindowSize() const override { return {.width = 900, .height = 680, .minimumWidth = 560, .minimumHeight = 420}; }

    void open(App::PluginHost& opened) override {
        host = &opened;
        Rml::Factory::RegisterElementInstancer("trainingchart", &chartInstancer);
        display.emplace(opened.getSession().getDevice(), trainer,
                        Source::Events{[this] { return getBanner(); }, [this] { requestAdd(); }, [this] { showScene(); }});
        DiffRenderer::Methods::Catalog catalog;
        for (const DiffRenderer::Methods::Method* method : catalog.getAll())
            if (method->getDescription().scene)
                state.methods.push_back(
                    {Rml::String(method->getDescription().id), Rml::String(method->getDescription().name), Rml::String(method->getDescription().renderer)});
        scan();
    }

    void show() override {
        if (window) {
            SDL_RaiseWindow(window->getNative());
            return;
        }
        RmlKit::UI::SecondaryWindow& opened = host->openWindow(*this);
        bind(opened.getModel());
        Rml::ElementDocument* document = opened.getContext().LoadDocument(documentPath);
        if (!document) {
            host->reportError(source, std::format("RmlUi cannot load {}", documentPath));
            opened.close();
            return;
        }
        document->Show();
        window = &opened;
    }

    bool update() override {
        const TrainerStatus status = trainer.getStatus();
        if (status.stage == Stage::Ready && !shown)
            showDataset();
        state.stage = getStageName(status.stage);
        state.message = status.message;
        state.progress = std::round(status.progress * 100.0f);
        state.busy = status.stage == Stage::Colmap || status.stage == Stage::Loading;
        state.ready = status.stage == Stage::Ready;
        state.failed = status.stage == Stage::Failed;
        state.training = status.training;
        state.iteration = status.iteration;
        state.trained = status.iteration > 0;
        state.loss = status.iteration > 0 ? std::format("{:.4f}", status.loss) : "–";
        state.psnr = status.psnr > 0.0 ? std::format("{:.2f} dB", status.psnr) : "–";
        state.speed = status.stepMilliseconds > 0.0 ? std::format("{:.0f} it/s", 1000.0 / status.stepMilliseconds) : "–";
        state.primitives = status.primitives > 0 ? formatCount(status.primitives) : "–";
        state.hasResult = status.hasResult && (status.resultIsMesh || Gpu::GaussianAccelerationStructure::isAvailable());
        state.resultIsMesh = status.resultIsMesh;
        updateChart(status);
        updateCloud(status);
        state.previewing = shown && viewing;
        if (status.stage == Stage::Failed && !reported) {
            host->reportError(source, status.message);
            reported = true;
        }
        if (window)
            window->getModel().dirty("recon");
        return state.busy || status.training || trainer.getStatus().readingResult || isOutdated(status);
    }

    void close() override {
        trainer.close();
        if (display)
            display->close();
        display.reset();
    }

    void windowClosed(RmlKit::UI::SecondaryWindow& closed) override {
        if (&closed == window)
            window = nullptr;
    }

private:
    void bind(RmlKit::UI::Model& model) {
        model.bindData([this](Rml::DataModelConstructor& constructor) {
            if (auto row = constructor.RegisterStruct<MethodRow>()) {
                row.RegisterMember("id", &MethodRow::id);
                row.RegisterMember("name", &MethodRow::name);
                row.RegisterMember("renderer", &MethodRow::renderer);
            }
            constructor.RegisterArray<std::vector<MethodRow>>();
            if (auto row = constructor.RegisterStruct<DatasetRow>()) {
                row.RegisterMember("group", &DatasetRow::group);
                row.RegisterMember("name", &DatasetRow::name);
                row.RegisterMember("kind", &DatasetRow::kind);
                row.RegisterMember("folder", &DatasetRow::folder);
            }
            constructor.RegisterArray<std::vector<DatasetRow>>();
            if (auto fields = constructor.RegisterStruct<Model>()) {
                fields.RegisterMember("datasetsRoot", &Model::datasetsRoot);
                fields.RegisterMember("datasets", &Model::datasets);
                fields.RegisterMember("folder", &Model::folder);
                fields.RegisterMember("stage", &Model::stage);
                fields.RegisterMember("message", &Model::message);
                fields.RegisterMember("progress", &Model::progress);
                fields.RegisterMember("methods", &Model::methods);
                fields.RegisterMember("method", &Model::method);
                fields.RegisterMember("iterations", &Model::iterations);
                fields.RegisterMember("iteration", &Model::iteration);
                fields.RegisterMember("loss", &Model::loss);
                fields.RegisterMember("psnr", &Model::psnr);
                fields.RegisterMember("lossHigh", &Model::lossHigh);
                fields.RegisterMember("lossLow", &Model::lossLow);
                fields.RegisterMember("psnrHigh", &Model::psnrHigh);
                fields.RegisterMember("psnrLow", &Model::psnrLow);
                fields.RegisterMember("iterationSpan", &Model::iterationSpan);
                fields.RegisterMember("hasHistory", &Model::hasHistory);
                fields.RegisterMember("hasPsnr", &Model::hasPsnr);
                fields.RegisterMember("speed", &Model::speed);
                fields.RegisterMember("primitives", &Model::primitives);
                fields.RegisterMember("busy", &Model::busy);
                fields.RegisterMember("ready", &Model::ready);
                fields.RegisterMember("trained", &Model::trained);
                fields.RegisterMember("training", &Model::training);
                fields.RegisterMember("failed", &Model::failed);
                fields.RegisterMember("hasResult", &Model::hasResult);
                fields.RegisterMember("resultIsMesh", &Model::resultIsMesh);
                fields.RegisterMember("added", &Model::added);
                fields.RegisterMember("adding", &Model::adding);
                fields.RegisterMember("live", &Model::live);
                fields.RegisterMember("previewing", &Model::previewing);
                fields.RegisterMember("view", &Model::view);
                fields.RegisterMember("viewCount", &Model::viewCount);
            }
            constructor.Bind("recon", &state);
        });
        model.bindEvent("reconDatasetsRoot", &ReconstructionPlugin::onDatasetsRoot, this);
        model.bindEvent("reconRescan", &ReconstructionPlugin::onRescan, this);
        model.bindEvent("reconDataset", &ReconstructionPlugin::onDataset, this);
        model.bindEvent("reconOpen", &ReconstructionPlugin::onOpen, this);
        model.bindEvent("reconLoad", &ReconstructionPlugin::onLoad, this);
        model.bindEvent("reconMethod", &ReconstructionPlugin::onMethod, this);
        model.bindEvent("reconTrain", &ReconstructionPlugin::onTrain, this);
        model.bindEvent("reconRestart", &ReconstructionPlugin::onRestart, this);
        model.bindEvent("reconView", &ReconstructionPlugin::onView, this);
        model.bindEvent("reconAdd", &ReconstructionPlugin::onAdd, this);
        model.bindEvent("reconLive", &ReconstructionPlugin::onLive, this);
        model.bindEvent("reconPathTracer", &ReconstructionPlugin::onPathTracer, this);
    }

    void scan() {
        state.datasets.clear();
        for (const DatasetEntry& entry : scanDatasets(std::filesystem::path(state.datasetsRoot)))
            state.datasets.push_back({Rml::String(entry.group), Rml::String(entry.name), getDatasetKindName(entry.kind), entry.folder.string()});
    }

    void openFolder(const std::filesystem::path& folder) {
        showScene();
        display->close();
        previewCamera.reset();
        shown = false;
        reported = false;
        cloud = {};
        placedMesh = {};
        state.added = false;
        state.adding = false;
        state.live = false;
        state.folder = folder.string();
        state.view = 0;
        state.viewCount = 0;
        trainer.open(host->getSession().getDevice(), folder, state.method, std::max(state.iterations, 1));
        host->redraw();
    }

    void showDataset() {
        const DiffRenderer::Scenes::Dataset* dataset = trainer.getDataset();
        if (!dataset)
            return;
        shown = true;
        alignment = Alignment(*dataset);
        display->open(alignment, alignment.getCameraBounds(*dataset));
        state.viewCount = static_cast<int>(dataset->train.size());
        showPreview();
        moveToView(0);
    }

    bool isOutdated(const TrainerStatus& status) const {
        return state.added && state.live && status.hasResult && status.iteration > 0 && status.iteration != cloudIteration;
    }

    //! The status is older than the cloud that it takes, so a request that the status no longer lists has its result here or has none.
    void updateCloud(const TrainerStatus& status) {
        const Scene::Scene& scene = host->getSession().getScene();
        if (state.added && !scene.find(cloud) && !scene.find(placedMesh.node)) {
            cloud = {};
            placedMesh = {};
            state.added = false;
            state.live = false;
        }
        if (std::optional<TrainerResult> taken = trainer.takeResult())
            place(std::move(*taken));
        else if (!status.readingResult)
            state.adding = false;
        const std::chrono::seconds interval = status.resultIsMesh ? meshRefreshInterval : refreshInterval;
        if (status.readingResult || !isOutdated(status) || std::chrono::steady_clock::now() - placedAt < interval)
            return;
        trainer.requestResult();
    }

    void place(TrainerResult taken) {
        if (!state.adding && !state.added)
            return;
        cloudIteration = taken.iteration;
        placedAt = std::chrono::steady_clock::now();
        if (Scene::GaussianCloud* cloudResult = std::get_if<Scene::GaussianCloud>(&taken.object))
            placeCloud(std::move(*cloudResult));
        else
            placeMesh(std::move(std::get<TrainerMesh>(taken.object)));
        if (!state.adding)
            return;
        state.adding = false;
        //! The viewport stays at the camera of the preview, so it shows the new object in the same place.
        sceneCamera.reset();
        showScene();
        host->reportDone(source, std::format("Added the result to the scene after {} iterations", cloudIteration));
    }

    //! The mesh, material and textures of a placed mesh, which a refresh replaces together.
    struct PlacedMesh {
        Scene::NodeHandle node;
        Scene::InstanceHandle instance;
        Scene::MeshHandle mesh;
        Scene::MaterialHandle material;
        std::vector<Scene::TextureHandle> textures;
    };

    void placeMesh(TrainerMesh taken) {
        Scene::Scene& scene = host->getSession().getScene();
        Core::Transform transform = alignment.getDatasetToScene();
        std::string name = trainer.getDataset()->name;
        if (const Scene::Node* existing = scene.find(placedMesh.node)) {
            transform = existing->localTransform;
            name = existing->name;
            removeMesh();
        }
        Scene::Material material{.name = name};
        material.baseColor = {1.0f, 1.0f, 1.0f};
        material.roughness = taken.roughness;
        material.metallic = taken.metallic;
        if (taken.textures.empty()) {
            material.useVertexColor = true;
        } else {
            for (Scene::Texture& texture : taken.textures) {
                texture.name = std::format("{} {}", name, texture.name);
                placedMesh.textures.push_back(scene.addTexture(std::move(texture)));
            }
            material.roughness = 1.0f;
            material.metallic = 1.0f;
            material.texture(Scene::TextureSlot::BaseColor) = Scene::MaterialTexture{.texture = placedMesh.textures[0]};
            material.texture(Scene::TextureSlot::Roughness) = Scene::MaterialTexture{.texture = placedMesh.textures[1], .channel = 1};
            material.texture(Scene::TextureSlot::Metallic) = Scene::MaterialTexture{.texture = placedMesh.textures[1], .channel = 2};
            material.texture(Scene::TextureSlot::Normal) = Scene::MaterialTexture{.texture = placedMesh.textures[2]};
        }
        placedMesh.material = scene.addMaterial(std::move(material));
        taken.mesh.name = name;
        placedMesh.mesh = scene.addMesh(std::move(taken.mesh));
        placedMesh.instance = scene.addInstance(
            {.transform = transform, .mesh = placedMesh.mesh, .flags = Scene::InstanceFlags::DoubleSided, .material = placedMesh.material});
        placedMesh.node = scene.addNode({std::move(name), transform, {}, placedMesh.instance});
        state.added = true;
        host->sceneChanged();
    }

    void removeMesh() {
        Scene::Scene& scene = host->getSession().getScene();
        scene.remove(placedMesh.node);
        scene.remove(placedMesh.mesh);
        scene.remove(placedMesh.material);
        for (const Scene::TextureHandle texture : placedMesh.textures)
            scene.remove(texture);
        placedMesh = {};
    }

    void placeCloud(Scene::GaussianCloud taken) {
        Scene::Scene& scene = host->getSession().getScene();
        if (Scene::GaussianCloud* existing = scene.findForTraining(cloud)) {
            taken.name = std::move(existing->name);
            taken.transform = existing->transform;
            *existing = std::move(taken);
            scene.markGaussiansChanged();
            host->redraw();
            return;
        }
        const Core::Transform transform = alignment.getDatasetToScene();
        std::string name = trainer.getDataset()->name;
        taken.name = name;
        taken.transform = transform;
        cloud = scene.addGaussianCloud(std::move(taken));
        scene.addNode({std::move(name), transform, {}, cloud});
        state.added = true;
        host->sceneChanged();
    }

    App::ViewportBanner getBanner() const {
        const auto row = std::ranges::find(state.methods, state.method, &MethodRow::id);
        const DiffRenderer::Scenes::Dataset* dataset = trainer.getDataset();
        return {.title = std::format("Training preview of {}", dataset ? dataset->name : ""),
                .detail = std::format("{}, iteration {}", row->name, state.iteration),
                .action = state.adding  ? "Adding…"
                          : state.added ? "Update in scene"
                                        : "Add to scene",
                .actionEnabled = canAdd()};
    }

    bool canAdd() const { return state.hasResult && state.trained && !state.adding; }

    void requestAdd() {
        if (!canAdd())
            return;
        state.adding = true;
        trainer.requestResult();
        host->redraw();
    }

    void showPreview() {
        if (const Scene::Camera* active = host->getSession().getScene().findActiveCamera())
            sceneCamera = *active;
        viewing = true;
        host->setViewportSource(&*display);
        if (previewCamera)
            host->moveCamera(previewCamera->transform, previewCamera->verticalFov);
    }

    void showScene() {
        if (!viewing)
            return;
        if (const Scene::Camera* active = host->getSession().getScene().findActiveCamera())
            previewCamera = *active;
        viewing = false;
        host->setViewportSource(nullptr);
        if (sceneCamera)
            host->moveCamera(sceneCamera->transform, sceneCamera->verticalFov);
    }

    void moveToView(const int index) {
        const DiffRenderer::Scenes::Dataset* dataset = trainer.getDataset();
        if (!dataset || dataset->train.empty())
            return;
        const int count = static_cast<int>(dataset->train.size());
        state.view = (index % count + count) % count;
        const DiffRenderer::Scenes::Camera& camera = dataset->train[static_cast<std::size_t>(state.view)].camera;
        host->moveCamera(alignment.toScene(camera), Alignment::getVerticalFov(camera));
    }

    void onDatasetsRoot(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
        RmlKit::UI::Model::requireNoArguments(arguments, "reconDatasetsRoot");
        RmlKit::UI::FileDialog::openFolder(window ? window->getNative() : host->getWindow(), RmlKit::UI::FolderKind::Export, state.datasetsRoot,
                                           [this](const std::filesystem::path& folder) {
                                               state.datasetsRoot = folder.string();
                                               scan();
                                               host->redraw();
                                           });
    }

    void onRescan(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
        RmlKit::UI::Model::requireNoArguments(arguments, "reconRescan");
        scan();
        host->redraw();
    }

    void onDataset(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
        const std::size_t index = RmlKit::UI::Model::indexArgument(arguments, "reconDataset", state.datasets.size());
        openFolder(std::filesystem::path(state.datasets[index].folder));
    }

    void onOpen(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
        RmlKit::UI::Model::requireNoArguments(arguments, "reconOpen");
        RmlKit::UI::FileDialog::openFolder(window ? window->getNative() : host->getWindow(), RmlKit::UI::FolderKind::Export, state.folder,
                                           [this](const std::filesystem::path& folder) { openFolder(folder); });
    }

    void onLoad(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList&) {
        if (state.folder.empty())
            return;
        openFolder(std::filesystem::path(state.folder));
    }

    void onMethod(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
        const std::size_t index = RmlKit::UI::Model::indexArgument(arguments, "reconMethod", state.methods.size());
        state.method = state.methods[index].id;
        trainer.setMethod(state.method, std::max(state.iterations, 1));
        history.clear();
        host->redraw();
    }

    void updateChart(const TrainerStatus& status) {
        history.record(status);
        state.hasHistory = !history.isEmpty();
        state.hasPsnr = !history.getPsnr().empty();
        const ChartRange loss = history.getLossRange();
        const ChartRange psnr = history.getPsnrRange();
        state.lossHigh = std::format("{:.3g}", std::pow(10.0f, loss.high));
        state.lossLow = std::format("{:.3g}", std::pow(10.0f, loss.low));
        state.psnrHigh = std::format("{:.1f} dB", psnr.high);
        state.psnrLow = std::format("{:.1f} dB", psnr.low);
        state.iterationSpan = formatCount(static_cast<std::size_t>(history.getIterationSpan()));
    }

    void onTrain(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
        RmlKit::UI::Model::requireNoArguments(arguments, "reconTrain");
        trainer.setTraining(!state.training);
        host->redraw();
    }

    void onRestart(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
        RmlKit::UI::Model::requireNoArguments(arguments, "reconRestart");
        trainer.setMethod(state.method, std::max(state.iterations, 1));
        host->redraw();
    }

    void onView(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
        if (arguments.size() != 1)
            return;
        if (shown && !viewing)
            showPreview();
        moveToView(state.view + arguments[0].Get<int>());
    }

    void onAdd(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
        RmlKit::UI::Model::requireNoArguments(arguments, "reconAdd");
        requestAdd();
    }

    void onLive(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
        RmlKit::UI::Model::requireNoArguments(arguments, "reconLive");
        host->redraw();
    }

    void onPathTracer(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
        RmlKit::UI::Model::requireNoArguments(arguments, "reconPathTracer");
        if (!shown)
            return;
        if (viewing)
            showScene();
        else
            showPreview();
    }

    App::PluginHost* host = nullptr;
    RmlKit::UI::SecondaryWindow* window = nullptr;
    Trainer trainer;
    std::optional<Source> display;
    Alignment alignment;
    Model state;
    TrainingHistory history;
    TrainingChartInstancer chartInstancer{history};
    Scene::GaussianCloudHandle cloud;
    PlacedMesh placedMesh;
    int cloudIteration = 0;
    std::chrono::steady_clock::time_point placedAt;
    bool shown = false;
    bool viewing = false;
    //! The camera of the scene before the preview, and of the preview before the scene.
    std::optional<Scene::Camera> sceneCamera;
    std::optional<Scene::Camera> previewCamera;
    bool reported = false;
};

ReconstructionPlugin plugin;
const App::PluginRegistration registration{plugin};

}

}
