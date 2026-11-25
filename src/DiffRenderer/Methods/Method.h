#pragma once

#include <cstdint>
#include <expected>
#include <optional>
#include <string_view>
#include <vector>

#include "DiffRenderer/Core/Image.h"
#include "DiffRenderer/Scenes/Dataset.h"
#include "DiffRenderer/Training/Loss.h"
#include "NoorRHI/NoorRHI.h"

namespace DiffRenderer::Methods {

struct Settings {
    std::uint64_t seed = 1;
    float learningRateScale = 1.0f;
    //! The planned number of iterations; the schedules of densification and learning rates scale to it.
    int iterations = 30000;
    //! Turns the treatment of the discontinuities on or off, for the methods that have one.
    bool boundaryTerm = true;
};

struct Description {
    std::string_view id;
    std::string_view name;
    std::string_view representation;
    std::string_view renderer;
    std::string_view discontinuities;
    std::string_view reference;
    bool scene = false;
    bool object = false;
    bool volume = false;
    bool hasBoundaryTerm = false;
};

struct Statistics {
    //! Gaussians, voxels, triangles or texels, whichever the representation counts.
    std::size_t primitives = 0;
    std::size_t bytes = 0;
};

//! One training view with its image already on the GPU.
struct Target {
    const Scenes::View* view = nullptr;
    std::uint32_t index = 0;
    //! The image as RGBA8 in the encoding of the dataset, rows from the top.
    NoorRHI::Ptr<NoorRHI::uint> image;
};

struct TrainedGaussians {
    std::vector<float> means;
    //! Unit quaternions as (x, y, z, w).
    std::vector<float> quaternions;
    std::vector<float> scales;
    std::vector<float> opacities;
    //! Sixteen RGB coefficients per Gaussian, in the order Gaussian, coefficient, channel.
    std::vector<float> sh;
    std::uint32_t shDegree = 0;
};

//! A triangle mesh in the space of the dataset with the PBR materials of nvdiffrec: kd linear, ks as occlusion, roughness and metallic.
struct TrainedMesh {
    std::vector<float> positions;
    std::vector<std::uint32_t> triangles;
    //! kd and ks of each vertex, six values each; only without a texture.
    std::vector<float> vertexMaterials;
    //! Corner c reads uvs[uvTriangles[c]], with v down the rows of the texture.
    std::vector<float> uvs;
    std::vector<std::uint32_t> uvTriangles;
    std::uint32_t textureWidth = 0;
    std::uint32_t textureHeight = 0;
    //! kd, ks and the tangent space normal of each texel, nine values each, rows from the top.
    std::vector<float> texels;
};

//! A representation and its differentiable renderer, trained on the views of a dataset.
class Method {
public:
    virtual ~Method() = default;

    virtual const Description& getDescription() const = 0;
    bool supports(Scenes::Kind kind) const;
    [[nodiscard]] virtual std::expected<void, NoorRHI::Error> open(NoorRHI::Device& device, const Scenes::Dataset& dataset, const Settings& settings) = 0;
    //! One iteration: renders the view, records the loss, back-propagates, updates and runs the schedules of this iteration. Returns the loss.
    [[nodiscard]] virtual std::expected<double, NoorRHI::Error> step(const Target& target, Training::Loss& loss) = 0;
    [[nodiscard]] virtual std::expected<void, NoorRHI::Error> render(const Scenes::Camera& camera, Core::Image& image) = 0;
    virtual Statistics getStatistics() const = 0;
    [[nodiscard]] virtual std::expected<std::optional<TrainedGaussians>, NoorRHI::Error> readGaussians() { return std::nullopt; }
    [[nodiscard]] virtual std::expected<std::optional<TrainedMesh>, NoorRHI::Error> readMesh() { return std::nullopt; }
    //! Frees the GPU memory; the comparison closes each method after its final evaluation, so the next one has the whole GPU.
    virtual void close() {}
};

}
