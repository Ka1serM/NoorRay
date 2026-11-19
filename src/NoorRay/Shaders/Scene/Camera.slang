import NoorRHI;
import SceneRecords;
import Common.Math;
import Common.Sampling;
import Common.RayCone;
import Scene.Lens;

struct CameraRay {
    float3 origin;
    float3 direction;
    float weight;
    bool valid;
    bool dispersive;
};

//! film is the position on the image in [0, 1]^2 with y down. lensSample picks the aperture point. lambda is the hero wavelength, used only by a realistic lens.
CameraRay generateCameraRay(CameraRecord camera, float2 film, float2 lensSample, float lambda) {
    CameraRay ray;
    ray.origin = float3(0.0, 0.0, 0.0);
    ray.direction = float3(0.0, 0.0, -1.0);
    ray.weight = 1.0;
    ray.valid = true;
    ray.dispersive = false;
    const float2 sensor = float2((film.x - 0.5) * camera.sensor.x + camera.sensor.z, (0.5 - film.y) * camera.sensor.y + camera.sensor.w);
    const float focal = max(camera.projection.x, 1e-6);
    const uint type = camera.meta.x;
    if (type == cameraTypeOrthographic) {
        ray.origin = float3(sensor * (camera.projection.y / camera.sensor.y), 0.0);
    } else if (type == cameraTypeFisheye) {
        const float radius = length(sensor);
        const float theta = radius / focal;
        if (theta >= pi) {
            ray.valid = false;
            return ray;
        }
        if (radius > 1e-9)
            ray.direction = float3(sensor / radius * sin(theta), -cos(theta));
    } else if (type == cameraTypeRealistic && camera.meta.y != 0) {
        const float3 filmPoint = float3(-sensor, 0.0);
        const float rearZ = camera.lensElements[camera.meta.y - 1].geometry.y;
        const float3 pupil = float3(sampleConcentricDisk(lensSample) * camera.rearRadius, rearZ);
        float3 origin = filmPoint;
        float3 direction = normalize(pupil - filmPoint);
        const float cosTheta = -direction.z;
        bool dispersive;
        if (!traceLens(camera, origin, direction, lambda, dispersive)) {
            ray.valid = false;
            return ray;
        }
        ray.origin = origin - float3(0.0, 0.0, camera.lensFront);
        ray.direction = direction;
        ray.weight = square(square(cosTheta)) * camera.aperture.w;
        ray.dispersive = dispersive;
    } else {
        ray.direction = normalize(float3(sensor / focal, -1.0));
    }

    const float apertureRadius = camera.aperture.x;
    if (apertureRadius > 0.0 && type != cameraTypeOrthographic && type != cameraTypeRealistic) {
        const uint blades = uint(camera.aperture.y);
        const float2 lens = (blades >= 3 ? samplePolygon(lensSample, blades, camera.aperture.z) : sampleConcentricDisk(lensSample)) * apertureRadius;
        const float focus = camera.projection.z;
        const float3 focusPoint = ray.direction * (focus / max(-ray.direction.z, 1e-6));
        ray.origin = float3(lens, 0.0);
        ray.direction = normalize(focusPoint - ray.origin);
    }

    ray.origin = transformPoint(camera.cameraToWorld, ray.origin);
    ray.direction = normalize(transformVector(camera.cameraToWorld, ray.direction));
    return ray;
}

RayCone cameraRayCone(CameraRecord camera) {
    if (camera.meta.x == cameraTypeOrthographic)
        return makeRayCone(camera.clip.z, 0.0);
    return makeRayCone(0.0, camera.projection.w);
}

float3 cameraPosition(CameraRecord camera) { return float3(camera.cameraToWorld[0].w, camera.cameraToWorld[1].w, camera.cameraToWorld[2].w); }

//! The film position in [0, 1]^2 (y down) of a world point seen by the camera transform rows; false behind the camera. Pinhole projection for every type but orthographic, which is what reprojection needs.
bool projectToFilm(CameraRecord camera, float4 worldToCamera[3], float3 world, out float2 film) {
    const float3 local = transformPoint(worldToCamera, world);
    film = float2(0.0, 0.0);
    float2 sensor;
    if (camera.meta.x == cameraTypeOrthographic)
        sensor = local.xy * (camera.sensor.y / camera.projection.y);
    else {
        if (local.z >= -1e-6)
            return false;
        if (camera.meta.x == cameraTypeFisheye) {
            const float3 direction = normalize(local);
            const float theta = acos(clamp(-direction.z, -1.0, 1.0));
            const float sinTheta = length(direction.xy);
            sensor = sinTheta > 1e-9 ? direction.xy / sinTheta * theta * camera.projection.x : float2(0.0, 0.0);
        } else
            sensor = local.xy / -local.z * camera.projection.x;
    }
    sensor -= camera.sensor.zw;
    film = float2(sensor.x / camera.sensor.x + 0.5, 0.5 - sensor.y / camera.sensor.y);
    return true;
}

//! The clip position of a world point for a perspective or orthographic rasterizer: the inverse of generateCameraRay without an aperture. The rasterizer samples pixel centres, so the image shifts by the jitter to sample (pixel + jitter) / size like a traced primary ray. Clip y points up, depth is [0, 1] from near to far.
float4 projectToClip(CameraRecord camera, float3 world, float2 jitter) {
    const float3 local = transformPoint(camera.worldToCamera, world);
    const float depth = -local.z;
    const float near = camera.clip.x;
    const float far = max(camera.clip.y, near * 2.0);
    const bool orthographic = camera.meta.x == cameraTypeOrthographic;
    const float w = orthographic ? 1.0 : depth;
    const float2 scaledSensor = orthographic ? local.xy * (camera.sensor.y / camera.projection.y) : local.xy * camera.projection.x;
    const float2 offset = (0.5 - jitter) / float2(camera.meta.zw);
    const float filmX = (scaledSensor.x - camera.sensor.z * w) / camera.sensor.x + (0.5 + offset.x) * w;
    const float filmY = 0.5 * w - (scaledSensor.y - camera.sensor.w * w) / camera.sensor.y + offset.y * w;
    const float z = orthographic ? (depth - near) / (far - near) : far * (depth - near) / (far - near);
    return float4(2.0 * filmX - w, w - 2.0 * filmY, z, w);
}
