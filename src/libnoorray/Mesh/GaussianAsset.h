#pragma once

#include <array>
#include <cstdint>
#include <string>

#include <glm/ext/matrix_float4x3.hpp>
#include <glm/vec3.hpp>

#include "CUDA/rstd/Vector.h"
#include "Mesh/GaussianCutoff.h"
#include "Scene/Inspectable.h"

class Scene;

struct Gaussian
{
    glm::mat4x3 transform; // 4 columns of vec3: [R*S_col0, R*S_col1, R*S_col2, pos]
    float opacity; // sigmoid of the PLY logit, used by the any-hit's Russian roulette
    // Raw 3DGS SH coefficients: f_dc followed by f_rest in coefficient-major
    // RGB order. Color is evaluated and converted to a spectrum at runtime.
    std::array<glm::vec3, 16> shCoeffs{};
    uint32_t shCoeffCount = 0;
};

class GaussianAsset : public Inspectable
{
public:
    static GaussianAsset CreateFromFile(
        Scene& scene, const std::string& name, const std::string& path);

    GaussianAsset(Scene& scene, std::string name, nr::rstd::vector<Gaussian> gaussians);
    GaussianAsset(GaussianAsset&& other) noexcept = default;
    GaussianAsset(const GaussianAsset&) = delete;
    GaussianAsset& operator=(const GaussianAsset&) = delete;
    GaussianAsset& operator=(GaussianAsset&&) = delete;
    ~GaussianAsset() = default;

    const std::string& getName() const override { return name; }
    std::string getType() const override { return "Gaussian Asset"; }

    const nr::rstd::vector<Gaussian>& getGaussians() const { return gaussians; }
    nr::rstd::vector<Gaussian>& getGaussians() { return gaussians; }
    uint32_t getGaussianCount() const { return static_cast<uint32_t>(gaussians.size()); }

    // Grows/shrinks the backing storage (e.g. after a training densify/prune
    // step changes the Gaussian count). New entries are default-constructed;
    // callers (GaussianTrainRenderer::bakeTransformsAndUpdateTlas) overwrite
    // every field for every index right after resizing, every call.
    void resizeGaussians(const uint32_t count) { gaussians.resize(count); dirty = true; }

    const std::string& getPath() const { return path; }

    bool isDirty() const { return dirty; }
    void clearDirtyFlag() { dirty = false; }

private:
    Scene& scene;
    std::string name;
    std::string path;
    bool dirty = false;
    nr::rstd::vector<Gaussian> gaussians;
};
