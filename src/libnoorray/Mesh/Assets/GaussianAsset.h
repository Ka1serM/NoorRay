#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <glm/ext/matrix_float4x3.hpp>

#include "CUDA/rstd/Vector.h"
#include "Shading/SphericalHarmonics.h"
#include "Scene/Inspectable.h"

class Scene;

struct Gaussian
{
    glm::mat4x3 transform;
    float opacity;
    SphericalHarmonicsCoefficients sphericalHarmonics;

    glm::vec3 getShCoefficient(const uint32_t index) const
    {
        return sphericalHarmonics.get(index);
    }

    void setShCoefficient(const uint32_t index, const glm::vec3 value)
    {
        sphericalHarmonics.set(index, value);
    }
};

static_assert(sizeof(Gaussian) == 152);

class GaussianAsset : public Inspectable
{
public:
    static GaussianAsset CreateFromFile(
        Scene& scene, const std::string& name, const std::string& path);

    GaussianAsset(Scene& scene, std::string name, std::vector<Gaussian> gaussians);
    GaussianAsset(GaussianAsset&& other) noexcept = default;
    GaussianAsset(const GaussianAsset&) = delete;
    GaussianAsset& operator=(const GaussianAsset&) = delete;
    GaussianAsset& operator=(GaussianAsset&& other) noexcept = default;
    ~GaussianAsset() = default;

    // Frees the splat data, which is by far the largest resource in a scene.
    // Called by the registry once no instance refers to this asset any more.
    void releaseResources();

    const std::string& getName() const override { return name; }
    std::string getType() const override { return "Gaussian Asset"; }

    uint32_t getGaussianCount() const { return static_cast<uint32_t>(gaussians.size()); }
    Gaussian getGaussian(uint32_t index) const { return gaussians[index]; }
    const nr::rstd::vector<Gaussian>& getGaussians() const { return gaussians; }
    nr::rstd::vector<Gaussian>& getGaussians() { return gaussians; }

    // Direct mutations own their synchronization and invalidation. A bulk
    // replacement performs each exactly once, which is the trainer path.
    void setGaussian(uint32_t index, const Gaussian& gaussian);
    void setGaussians(const std::vector<Gaussian>& gaussians);
    void notifyGaussiansChanged();

    const std::string& getPath() const { return path; }

    bool isDirty() const { return dirty; }
    void clearDirtyFlag() { dirty = false; }

private:
    Scene* scene;
    std::string name;
    std::string path;
    bool dirty = false;
    nr::rstd::vector<Gaussian> gaussians;

};
