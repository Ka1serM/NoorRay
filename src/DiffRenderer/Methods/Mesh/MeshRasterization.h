#pragma once

#include <expected>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "DiffRenderer/Methods/Mesh/LargeStepOptimizer.h"
#include "DiffRenderer/Methods/Mesh/MeshRasterizer.h"
#include "DiffRenderer/Methods/Mesh/SurfaceMesh.h"
#include "DiffRenderer/Methods/Mesh/TextureAtlas.h"
#include "DiffRenderer/Methods/Mesh/TrainableLight.h"
#include "DiffRenderer/Methods/Method.h"
#include "DiffRenderer/Training/GpuAdam.h"

namespace DiffRenderer::Methods {

//! nvdiffrec on a triangle mesh from the visual hull or a sphere: PBR materials and a learned environment light; first remeshed with large steps and vertex materials, then textured through a UV atlas.
class MeshRasterization final : public Method {
public:
    const Description& getDescription() const override;
    std::expected<void, NoorRHI::Error> open(NoorRHI::Device& device, const Scenes::Dataset& dataset, const Settings& settings) override;
    std::expected<double, NoorRHI::Error> step(const Target& target, Training::Loss& loss) override;
    std::expected<void, NoorRHI::Error> render(const Scenes::Camera& camera, Core::Image& image) override;
    Statistics getStatistics() const override;
    std::expected<std::optional<TrainedMesh>, NoorRHI::Error> readMesh() override;
    void close() override;
    //! Writes the trained mesh as a textured OBJ; works after the close too.
    [[nodiscard]] std::expected<void, std::string> writeMesh(const std::filesystem::path& path) const;

private:
    [[nodiscard]] std::expected<void, NoorRHI::Error> setMesh();
    [[nodiscard]] std::expected<void, NoorRHI::Error> remesh();
    [[nodiscard]] std::expected<void, NoorRHI::Error> startTexture();
    [[nodiscard]] std::expected<void, NoorRHI::Error> restartMaterialMoments(std::size_t count);
    //! The learning rate factor of nvdiffrec in the current pass: a linear warmup, then a fall to a tenth at the end of the pass.
    float getSchedule() const;
    [[nodiscard]] std::expected<void, NoorRHI::Error> download(SurfaceMesh& trained, std::vector<float>& texels) const;
    [[nodiscard]] std::expected<void, std::string> writeTrainedMesh(const SurfaceMesh& trained, std::span<const float> texels,
                                                                    const std::filesystem::path& path) const;
    float getTargetLength() const;

    NoorRHI::Device* device = nullptr;
    Settings settings;
    SurfaceMesh mesh;
    TextureAtlas atlas;
    std::vector<float> texture;
    MeshRasterizer rasterizer;
    LargeStepOptimizer optimizer;
    TrainableLight light;
    Training::GpuAdam materialAdam;
    NoorRHI::Buffer<float> materialFirst;
    NoorRHI::Buffer<float> materialSecond;
    std::uint32_t materialStep = 0;
    bool lightBuilt = false;
    int iteration = 0;
    //! The first iteration of the textured pass, or -1 in the first pass.
    int textureStart = -1;
    float startLength = 0.0f;
    float area = 0.0f;
};

}
