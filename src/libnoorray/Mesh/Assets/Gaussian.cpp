#include "Mesh/Assets/Gaussian.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include <glm/glm.hpp>

#include "Scene/Scene.h"

GaussianAsset::GaussianAsset(Scene& scene, std::string name, std::vector<Gaussian> gaussians)
    : scene(scene), name(std::move(name)), gaussians(std::move(gaussians))
{
}

void GaussianAsset::releaseResources()
{
    name.clear();
    path.clear();
    dirty = false;
    // Move-assign from an empty vector: clear() would keep the managed
    // allocation alive, and freeing it is the entire point here.
    gaussians = std::vector<Gaussian>{};
}

void GaussianAsset::setGaussian(const uint32_t index, const Gaussian& gaussian)
{
    scene.synchronizeBeforeMutation();
    gaussians[index] = gaussian;
    notifyGaussiansChanged();
}

void GaussianAsset::notifyGaussiansChanged()
{
    dirty = true;
    scene.setDirtyFlag(GaussianData);
    scene.setDirtyFlag(TLAS);
    scene.setDirtyFlag(Accumulation);
}

void GaussianAsset::setGaussians(const std::vector<Gaussian>& gaussians)
{
    scene.synchronizeBeforeMutation();
    this->gaussians = gaussians;
    notifyGaussiansChanged();
}
