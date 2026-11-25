#include "NoorRay.Reconstruction/Alignment.h"

#include <cmath>

namespace NoorRay::Reconstruction {

namespace {

constexpr float sceneRadius = 4.0f;

Core::float3 xyz(const NoorRHI::float4 value) {
    return {value.x, value.y, value.z};
}

Core::float3 getCameraCenter(const DiffRenderer::Scenes::Camera& camera) {
    const Core::float3 translation{camera.worldToCamera[0].w, camera.worldToCamera[1].w, camera.worldToCamera[2].w};
    return -(xyz(camera.worldToCamera[0]) * translation.x + xyz(camera.worldToCamera[1]) * translation.y + xyz(camera.worldToCamera[2]) * translation.z);
}

//! The rotation that turns from onto +y, by the Rodrigues formula.
void rotateOntoY(const Core::float3 from, Core::float3 (&rows)[3]) {
    const Core::float3 to{0.0f, 1.0f, 0.0f};
    const Core::float3 axis = Core::cross(from, to);
    const float sine = Core::length(axis);
    const float cosine = Core::dot(from, to);
    if (sine < 1e-6f) {
        const float sign = cosine < 0.0f ? -1.0f : 1.0f;
        rows[0] = {1.0f, 0.0f, 0.0f};
        rows[1] = {0.0f, sign, 0.0f};
        rows[2] = {0.0f, 0.0f, sign};
        return;
    }
    const Core::float3 k = axis * (1.0f / sine);
    const float c = cosine;
    const float s = sine;
    const float t = 1.0f - c;
    rows[0] = {t * k.x * k.x + c, t * k.x * k.y - s * k.z, t * k.x * k.z + s * k.y};
    rows[1] = {t * k.x * k.y + s * k.z, t * k.y * k.y + c, t * k.y * k.z - s * k.x};
    rows[2] = {t * k.x * k.z - s * k.y, t * k.y * k.z + s * k.x, t * k.z * k.z + c};
}

}

Alignment::Alignment(const DiffRenderer::Scenes::Dataset& dataset) {
    Core::float3 up{};
    for (const DiffRenderer::Scenes::View& view : dataset.train)
        up = up - xyz(view.camera.worldToCamera[1]);
    if (Core::length(up) > 0.0f)
        rotateOntoY(Core::normalize(up), rows);
    center = {dataset.center.x, dataset.center.y, dataset.center.z};
    scale = dataset.extent > 0.0f ? sceneRadius / dataset.extent : 1.0f;
}

Core::float3 Alignment::toSceneVector(const Core::float3 vector) const {
    return {Core::dot(rows[0], vector), Core::dot(rows[1], vector), Core::dot(rows[2], vector)};
}

Core::float3 Alignment::toMethodVector(const Core::float3 vector) const {
    return rows[0] * vector.x + rows[1] * vector.y + rows[2] * vector.z;
}

DiffRenderer::Scenes::Camera Alignment::toMethod(const Core::Transform& cameraToWorld, const float verticalFov, const std::uint32_t width,
                                                 const std::uint32_t height) const {
    const Core::float3 right = Core::normalize(toMethodVector(cameraToWorld.getColumn(0)));
    const Core::float3 up = Core::normalize(toMethodVector(cameraToWorld.getColumn(1)));
    const Core::float3 back = Core::normalize(toMethodVector(cameraToWorld.getColumn(2)));
    const Core::float3 eye = toMethodVector(cameraToWorld.getTranslation()) * (1.0f / scale) + center;
    const Core::float3 axes[3] = {right, -up, -back};
    DiffRenderer::Scenes::Camera camera;
    for (int row = 0; row < 3; ++row)
        camera.worldToCamera[row] = {axes[row].x, axes[row].y, axes[row].z, -Core::dot(axes[row], eye)};
    const float focal = 0.5f * static_cast<float>(height) / std::tan(0.5f * verticalFov);
    camera.intrinsics = {focal, focal, 0.5f * static_cast<float>(width), 0.5f * static_cast<float>(height)};
    camera.width = width;
    camera.height = height;
    return camera;
}

Core::Transform Alignment::toScene(const DiffRenderer::Scenes::Camera& camera) const {
    const Core::float3 columns[4] = {toSceneVector(xyz(camera.worldToCamera[0])), toSceneVector(-xyz(camera.worldToCamera[1])),
                                     toSceneVector(-xyz(camera.worldToCamera[2])), toSceneVector(getCameraCenter(camera) - center) * scale};
    Core::Transform transform;
    for (int column = 0; column < 4; ++column) {
        transform.rows[0][column] = columns[column].x;
        transform.rows[1][column] = columns[column].y;
        transform.rows[2][column] = columns[column].z;
    }
    return transform;
}

Core::Transform Alignment::getDatasetToScene() const {
    const Core::float3 origin = toSceneVector(-center) * scale;
    const float translation[3] = {origin.x, origin.y, origin.z};
    Core::Transform transform;
    for (int row = 0; row < 3; ++row) {
        transform.rows[row][0] = rows[row].x * scale;
        transform.rows[row][1] = rows[row].y * scale;
        transform.rows[row][2] = rows[row].z * scale;
        transform.rows[row][3] = translation[row];
    }
    return transform;
}

float Alignment::getVerticalFov(const DiffRenderer::Scenes::Camera& camera) {
    return 2.0f * std::atan(0.5f * static_cast<float>(camera.height) / camera.intrinsics.y);
}

Core::Bounds Alignment::getCameraBounds(const DiffRenderer::Scenes::Dataset& dataset) const {
    Core::Bounds bounds;
    for (const DiffRenderer::Scenes::View& view : dataset.train)
        bounds.grow(toSceneVector(getCameraCenter(view.camera) - center) * scale);
    return bounds;
}

}
