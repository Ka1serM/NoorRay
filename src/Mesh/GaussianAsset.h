#pragma once

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
    // Jakob-Hanika RGB->spectrum sigmoid-polynomial coefficients (c0,c1,c2) for
    // colorDc, precomputed once here instead of doing the 64^3 table lookup on
    // every GPU hit — only the cheap sigmoid itself needs the per-sample
    // wavelength, so that's all Shade.cu still has to evaluate. Since these
    // fully replace colorDc for shading purposes, the raw RGB is never stored.
    glm::vec3 spectrumCoeffs;
};

class GaussianAsset : public Inspectable
{
public:
    static GaussianAsset CreateFromFile(Scene& scene, const std::string& name, const std::string& path);

    GaussianAsset(Scene& scene, std::string name, nr::rstd::vector<Gaussian> gaussians);
    GaussianAsset(GaussianAsset&& other) noexcept = default;
    GaussianAsset(const GaussianAsset&) = delete;
    GaussianAsset& operator=(const GaussianAsset&) = delete;
    GaussianAsset& operator=(GaussianAsset&&) = delete;
    ~GaussianAsset() = default;

    const std::string& getName() const override { return name; }
    std::string getType() const override { return "Gaussian Asset"; }
    bool renderUi() override;

    const nr::rstd::vector<Gaussian>& getGaussians() const { return gaussians; }
    uint32_t getGaussianCount() const { return static_cast<uint32_t>(gaussians.size()); }

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
