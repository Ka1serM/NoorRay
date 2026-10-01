#include "Scene/GaussianCloud.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

#include <glm/glm.hpp>

#include "Materials/Material.h"
#include "Mesh/Assets/Mesh.h"
#include "Scene/MeshInstance.h"
#include "Scene/Scene.h"

namespace
{
struct ProxyMesh { std::vector<glm::vec3> vertices; std::vector<uint32_t> indices; };

float proxyInradius(const ProxyMesh& mesh)
{
    float result = std::numeric_limits<float>::max();
    for (size_t face = 0; face < mesh.indices.size(); face += 3u)
    {
        const glm::vec3 a = mesh.vertices[mesh.indices[face]];
        const glm::vec3 b = mesh.vertices[mesh.indices[face + 1u]];
        const glm::vec3 c = mesh.vertices[mesh.indices[face + 2u]];
        const glm::vec3 normal = glm::normalize(glm::cross(b - a, c - a));
        result = std::min(result, std::abs(glm::dot(normal, a)));
    }
    if (!(result > 0.0f) || !std::isfinite(result))
        throw std::runtime_error("Gaussian proxy has an invalid inradius");
    return result;
}

ProxyMesh unitProxy(const GaussianProxyType type)
{
    if (type == GaussianProxyType::Octahedron) {
        return {{{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}},
            {0,2,4, 0,5,2, 0,3,5, 0,4,3, 1,4,2, 1,2,5, 1,5,3, 1,3,4}};
    }
    const float phi = 0.5f * (1.0f + std::sqrt(5.0f));
    ProxyMesh mesh{{{-1,phi,0},{1,phi,0},{-1,-phi,0},{1,-phi,0},
        {0,-1,phi},{0,1,phi},{0,-1,-phi},{0,1,-phi},
        {phi,0,-1},{phi,0,1},{-phi,0,-1},{-phi,0,1}},
        {0,11,5, 0,5,1, 0,1,7, 0,7,10, 0,10,11,
         1,5,9, 5,11,4, 11,10,2, 10,7,6, 7,1,8,
         3,9,4, 3,4,2, 3,2,6, 3,6,8, 3,8,9,
         4,9,5, 2,4,11, 6,2,10, 8,6,7, 9,8,1}};
    for (glm::vec3& vertex : mesh.vertices)
        vertex = glm::normalize(vertex);
    const uint32_t subdivisions = type == GaussianProxyType::IcosphereLevel2
        ? 2u : type == GaussianProxyType::Icosphere ? 1u : 0u;
    for (uint32_t level = 0; level < subdivisions; ++level) {
        std::map<std::pair<uint32_t, uint32_t>, uint32_t> midpoints;
        auto midpoint = [&](uint32_t a, uint32_t b) {
            if (a > b) std::swap(a, b);
            const auto key = std::pair{a, b};
            if (const auto found = midpoints.find(key); found != midpoints.end())
                return found->second;
            const uint32_t index = static_cast<uint32_t>(mesh.vertices.size());
            mesh.vertices.push_back(glm::normalize(mesh.vertices[a] + mesh.vertices[b]));
            midpoints.emplace(key, index);
            return index;
        };
        std::vector<uint32_t> refined;
        refined.reserve(mesh.indices.size() * 4u);
        for (size_t face = 0; face < mesh.indices.size(); face += 3u) {
            const uint32_t a = mesh.indices[face], b = mesh.indices[face + 1u],
                c = mesh.indices[face + 2u];
            const uint32_t ab = midpoint(a, b), bc = midpoint(b, c), ca = midpoint(c, a);
            refined.insert(refined.end(), {a,ab,ca, b,bc,ab, c,ca,bc, ab,bc,ca});
        }
        mesh.indices = std::move(refined);
    }
    return mesh;
}

MeshGeometry proxyGeometry(const GaussianProxyType type, const float cutoffSigma)
{
    ProxyMesh unit = unitProxy(type);
    const float scale = cutoffSigma / proxyInradius(unit);
    MeshStreams streams;
    for (const glm::vec3 direction : unit.vertices) {
        streams.positions.push_back(direction * scale);
        streams.tangents.emplace_back(glm::vec3(0.0f), direction, 1.0f);
        streams.uvs.emplace_back(0.0f);
    }
    streams.indices = std::move(unit.indices);
    streams.sections = singleSection(streams.indices);
    return MeshGeometry(std::move(streams));
}

// The layout Gaussian.slang reads: a four-word header, then one entry per
// splat of its opacity, SH coefficient count and 48 packed half SH values.
constexpr size_t HeaderWords = 4;
constexpr size_t EntryWords = 2 + MaxSphericalHarmonicsCoefficientCount
    * SphericalHarmonicsChannelCount / 2;

std::vector<std::uint32_t> splatParameters(const std::vector<Gaussian>& gaussians,
    const float cutoffSigma)
{
    std::vector<std::uint32_t> words(HeaderWords + gaussians.size() * EntryWords);
    words[0] = std::bit_cast<std::uint32_t>(cutoffSigma * cutoffSigma);
    words[1] = static_cast<std::uint32_t>(gaussians.size());
    for (size_t splat = 0; splat < gaussians.size(); ++splat) {
        const Gaussian& gaussian = gaussians[splat];
        std::uint32_t* entry = words.data() + HeaderWords + splat * EntryWords;
        entry[0] = std::bit_cast<std::uint32_t>(gaussian.opacity);
        entry[1] = gaussian.sphericalHarmonics.count;
        const auto& values = gaussian.sphericalHarmonics.values;
        for (size_t half = 0; half < values.size(); half += 2)
            entry[2 + half / 2] = values[half] | (static_cast<std::uint32_t>(values[half + 1]) << 16);
    }
    return words;
}

// A splat's columns are its R*S axes and its center.
Transform splatTransform(const Gaussian& gaussian)
{
    glm::mat4 matrix(1.0f);
    for (int column = 0; column < 4; ++column)
        matrix[column] = glm::vec4(gaussian.transform[column], column == 3 ? 1.0f : 0.0f);
    return Transform(matrix);
}
}

GaussianCloud::GaussianCloud(Scene& scene, const std::string& name, std::vector<Gaussian> gaussians,
    const GaussianProxyType proxyType, const float cutoffSigma)
    : SceneObject(scene, name, Transform())
    , pendingGaussians(std::move(gaussians))
    , splatCount(static_cast<uint32_t>(pendingGaussians.size()))
    , proxyType(proxyType)
    , cutoffSigma(cutoffSigma)
{
}

GaussianCloud::GaussianCloud(const GaussianCloud& other)
    : SceneObject(other)
    , splatCount(other.splatCount)
    , proxyType(other.proxyType)
    , cutoffSigma(other.cutoffSigma)
    , proxy(other.proxy)
    , material(other.material)
{
}

std::unique_ptr<SceneObject> GaussianCloud::clone() const
{
    return std::make_unique<GaussianCloud>(*this);
}

void GaussianCloud::onAdded()
{
    if (proxy)
        return;
    Material splatMaterial;
    splatMaterial.kind = MaterialKind::GaussianSplat;
    splatMaterial.compiled = true;
    // Acceptance happens in any-hit, so the proxies are never opaque.
    splatMaterial.shaderProgram.transparent = true;
    splatMaterial.shaderProgram.parameters = splatParameters(pendingGaussians, cutoffSigma);
    material = scene->add(std::move(splatMaterial));
    proxy = scene->add(Mesh(*scene, getName() + " Proxy", proxyGeometry(proxyType, cutoffSigma)), false);
    const SceneObjectHandle cloud = getHandle();
    for (uint32_t splat = 0; splat < pendingGaussians.size(); ++splat)
        scene->add(std::make_unique<MeshInstance>(*scene, "Splat", proxy,
            std::vector<Material*>{material}, splatTransform(pendingGaussians[splat]), splat), cloud);
    pendingGaussians = {};
}

void GaussianCloud::setProxy(const GaussianProxyType type, const float sigma)
{
    proxyType = type;
    cutoffSigma = sigma;
    if (!proxy)
        return;
    proxy->replaceGeometry(proxyGeometry(type, sigma));
    MaterialShaderProgram program = material->shaderProgram;
    program.parameters[0] = std::bit_cast<std::uint32_t>(sigma * sigma);
    scene->setMaterialProgram(scene->getMaterialIndex(material), std::move(program));
}
