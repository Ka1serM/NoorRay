#include "NoorRay/Gpu/CameraBuffers.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "NoorRay/Gpu/BufferUpload.h"

namespace NoorRay::Gpu {

namespace {

void writeRows(Core::float4 (&rows)[3], const Core::Transform& transform) {
    for (int row = 0; row < 3; ++row)
        rows[row] = {transform.rows[row][0], transform.rows[row][1], transform.rows[row][2], transform.rows[row][3]};
}

struct Ray {
    Core::float3 origin;
    Core::float3 direction;
};

bool intersectElement(const float radius, const float vertexZ, const float aperture, Ray& ray) {
    float t = 0.0f;
    if (radius == 0.0f) {
        if (ray.direction.z == 0.0f)
            return false;
        t = (vertexZ - ray.origin.z) / ray.direction.z;
    } else {
        const Core::float3 center{0.0f, 0.0f, vertexZ + radius};
        const Core::float3 offset = ray.origin - center;
        const float b = Core::dot(offset, ray.direction);
        const float c = Core::dot(offset, offset) - radius * radius;
        const float discriminant = b * b - c;
        if (discriminant < 0.0f)
            return false;
        const float root = std::sqrt(discriminant);
        const bool closer = (ray.direction.z > 0.0f) != (radius < 0.0f);
        t = closer ? -b - root : -b + root;
    }
    if (!(t > 0.0f))
        return false;
    const Core::float3 hit = ray.origin + ray.direction * t;
    if (aperture > 0.0f && hit.x * hit.x + hit.y * hit.y > aperture * aperture)
        return false;
    ray.origin = hit;
    return true;
}

bool refract(const Core::float3 incident, Core::float3 normal, const float eta, Core::float3& transmitted) {
    if (Core::dot(incident, normal) > 0.0f)
        normal = normal * -1.0f;
    const float cosIncident = -Core::dot(normal, incident);
    const float sinSquared = eta * eta * std::max(0.0f, 1.0f - cosIncident * cosIncident);
    if (sinSquared >= 1.0f)
        return false;
    transmitted = Core::normalize(incident * eta + normal * (eta * cosIncident - std::sqrt(1.0f - sinSquared)));
    return true;
}

float mediumIor(const Scene::Lens& lens, const std::size_t index) {
    const float ior = lens.elements[index].ior;
    return ior > 0.0f ? ior : 1.0f;
}

}

float focusLens(const Scene::Lens& lens, const float focusDistance) {
    const std::size_t count = lens.elements.size();
    const float nominal = count ? lens.elements.back().thickness : 0.0f;
    if (count == 0)
        return nominal;
    std::vector<float> vertexZ(count);
    vertexZ[count - 1] = -nominal;
    for (std::size_t index = count - 1; index > 0; --index)
        vertexZ[index - 1] = vertexZ[index] - lens.elements[index - 1].thickness;
    float stop = lens.elements.front().apertureRadius;
    for (const Scene::LensElement& element : lens.elements)
        if (element.apertureRadius > 0.0f)
            stop = std::min(stop > 0.0f ? stop : element.apertureRadius, element.apertureRadius);
    const float height = 0.05f * std::max(stop, 1e-4f);
    Ray ray;
    ray.origin = {0.0f, 0.0f, vertexZ.front() - std::max(focusDistance, 1e-3f)};
    ray.direction = Core::normalize(Core::float3{0.0f, height, vertexZ.front()} - ray.origin);
    for (std::size_t index = 0; index < count; ++index) {
        const Scene::LensElement& element = lens.elements[index];
        if (!intersectElement(element.curvatureRadius, vertexZ[index], 0.0f, ray))
            return nominal;
        if (element.curvatureRadius == 0.0f)
            continue;
        const Core::float3 center{0.0f, 0.0f, vertexZ[index] + element.curvatureRadius};
        const float before = index > 0 ? mediumIor(lens, index - 1) : 1.0f;
        if (!refract(ray.direction, Core::normalize(ray.origin - center), before / mediumIor(lens, index), ray.direction))
            return nominal;
    }
    if (ray.direction.y >= 0.0f || ray.direction.z <= 0.0f)
        return nominal;
    const float t = -ray.origin.y / ray.direction.y;
    const float filmZ = ray.origin.z + ray.direction.z * t;
    const float distance = filmZ - vertexZ.back();
    return distance > 0.0f ? distance : nominal;
}

std::expected<void, Core::Error> CameraBuffers::update(NoorRHI::Device& device, const Scene::Camera& camera, const std::uint32_t width,
                                                       const std::uint32_t height) {
    const float aspect = height > 0 ? float(width) / float(height) : 1.0f;
    const float sensorHeight = camera.sensor.height;
    const float sensorWidth = sensorHeight * aspect;
    const float focal = camera.getFocalLength();

    record = {};
    writeRows(record.cameraToWorld, camera.transform);
    writeRows(record.worldToCamera, camera.transform.inverse());
    writeRows(record.previousCameraToWorld, camera.previousTransform);
    writeRows(record.previousWorldToCamera, camera.previousTransform.inverse());
    record.sensor = {sensorWidth, sensorHeight, camera.sensor.offset.x, camera.sensor.offset.y};
    const float spread = std::atan(sensorHeight / (focal * float(std::max(height, 1u))));
    record.projection = {focal, camera.orthographicHeight, camera.focusDistance, spread};
    record.aperture = {camera.type == Scene::CameraType::Orthographic ? 0.0f : camera.getApertureRadius(), float(camera.blades), 0.0f, 1.0f};
    record.clip = {camera.near, camera.far, camera.orthographicHeight / float(std::max(height, 1u)), 0.0f};
    record.meta = {static_cast<std::uint32_t>(camera.type), 0, width, height};

    std::vector<Shaders::LensElementRecord> elements;
    if (camera.type == Scene::CameraType::Realistic && !camera.lens.isEmpty()) {
        const Scene::Lens& lens = camera.lens;
        const std::size_t count = lens.elements.size();
        float z = -focusLens(lens, camera.focusDistance);
        elements.resize(count);
        for (std::size_t index = count; index-- > 0;) {
            const Scene::LensElement& element = lens.elements[index];
            const float inverseAbbe = element.abbe > 0.0f ? 1.0f / element.abbe : 0.0f;
            elements[index].geometry = {element.curvatureRadius, z, element.apertureRadius, 0.0f};
            elements[index].medium = {mediumIor(lens, index), inverseAbbe, 0.0f, 0.0f};
            if (index > 0)
                z -= lens.elements[index - 1].thickness;
        }
        record.lensFront = elements.front().geometry.y;
        record.rearRadius = lens.elements.back().apertureRadius;
        record.meta.y = static_cast<std::uint32_t>(count);
    }
    const bool same = elements.size() == uploaded.size() && std::memcmp(elements.data(), uploaded.data(), elements.size() * sizeof(elements[0])) == 0;
    if (!same || !lensElements) {
        if (auto result = uploadValues<Shaders::LensElementRecord>(device, lensElements, elements); !result)
            return result;
        uploaded = elements;
    }
    record.lensElements = lensElements.getPtr();
    return {};
}

}
