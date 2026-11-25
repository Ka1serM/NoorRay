#include "DiffRenderer/Algorithms/GaussianTracing/GaussianScene.h"

namespace DiffRenderer::Algorithms {

using namespace NoorRHI;

namespace {

float3 add(const float3 a, const float3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

}

void GaussianScene::addTriangle(const float3 a, const float3 b, const float3 c, const float3 albedo) {
    for (const float3 corner : {a, b, c}) {
        indices.push_back(static_cast<std::uint32_t>(positions.size()));
        positions.push_back(corner);
    }
    albedos.push_back(albedo);
}

void GaussianScene::addQuad(const float3 corner, const float3 edgeU, const float3 edgeV, const float3 albedo) {
    const float3 opposite = add(add(corner, edgeU), edgeV);
    addTriangle(corner, add(corner, edgeU), opposite, albedo);
    addTriangle(corner, opposite, add(corner, edgeV), albedo);
}

void GaussianScene::addBox(const float3 low, const float3 high, const float3 albedo) {
    const float3 x{high.x - low.x, 0.0f, 0.0f};
    const float3 y{0.0f, high.y - low.y, 0.0f};
    const float3 z{0.0f, 0.0f, high.z - low.z};
    for (const float3 corner : {low, add(low, x)})
        addQuad(corner, y, z, albedo);
    for (const float3 corner : {low, add(low, y)})
        addQuad(corner, z, x, albedo);
    for (const float3 corner : {low, add(low, z)})
        addQuad(corner, x, y, albedo);
}

GaussianScene getFloorAndBox() {
    GaussianScene scene;
    scene.addQuad({-4.0f, -1.3f, -4.0f}, {8.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 8.0f}, {0.55f, 0.5f, 0.45f});
    scene.addBox({0.1f, -1.3f, 0.3f}, {0.7f, -0.5f, 0.9f}, {0.2f, 0.45f, 0.65f});
    return scene;
}

}
