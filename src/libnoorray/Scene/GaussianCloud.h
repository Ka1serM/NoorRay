#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "Mesh/Assets/Gaussian.h"
#include "Scene/SceneObject.h"

struct Material;
class Mesh;

// The mesh every splat of a cloud instances, scaled so its inradius reaches
// the cloud's cutoff.
enum class GaussianProxyType : int
{
    Icosphere,
    Octahedron,
    Icosahedron,
    IcosphereLevel2,
};

// A Gaussian splat cloud. Each splat is an ordinary mesh instance, a child of
// the cloud, of the proxy mesh the proxy type selects; the splat's rotation
// and scale are the instance's transform. The splats' opacity and spherical
// harmonics are the parameters of one material of the GaussianSplat kind,
// which the splat hit stages read at each instance's material entry.
class GaussianCloud : public SceneObject
{
public:
    GaussianCloud(Scene& scene, const std::string& name, std::vector<Gaussian> gaussians,
        GaussianProxyType proxyType = GaussianProxyType::Icosahedron, float cutoffSigma = 3.0f);
    // A copy shares its source's proxy mesh and material.
    GaussianCloud(const GaussianCloud& other);

    std::unique_ptr<SceneObject> clone() const override;
    void onAdded() override;

    uint32_t getSplatCount() const { return splatCount; }
    GaussianProxyType getProxyType() const { return proxyType; }
    float getCutoffSigma() const { return cutoffSigma; }
    // Rebuilds the proxy mesh all splats instance: one BLAS build.
    void setProxy(GaussianProxyType type, float sigma);

private:
    // The splats until onAdded() hands them to the scene.
    std::vector<Gaussian> pendingGaussians;
    uint32_t splatCount{};
    GaussianProxyType proxyType;
    float cutoffSigma;
    Mesh* proxy{};
    Material* material{};
};
