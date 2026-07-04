#pragma once

#include <cstdint>
#include <string>

#include <glm/ext/matrix_float4x3.hpp>
#include <glm/vec3.hpp>

#include "CUDA/rstd/Vector.h"
#include "Scene/Inspectable.h"

class Scene;

struct Gaussian
{
    glm::mat4x3 transform; // 4 columns of vec3: [R*S_col0, R*S_col1, R*S_col2, pos]
    uint32_t packedOpacityColor; // RGBA8: A = opacity, RGB = colorDc
};

class GaussianAsset : public Inspectable
{
public:
    static GaussianAsset CreateFromPly(Scene& scene, const std::string& name, const std::string& path);

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
