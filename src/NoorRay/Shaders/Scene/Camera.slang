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

//! Cycles' panoramas, in its space with +x ahead and +z up; a position without a ray gives zero.
float3 panoramaDirection(CameraRecord camera, float2 film) {
    const float4 p = camera.panorama[0];
    const float2 uv = float2(film.x * camera.window.x + camera.window.z, 1.0 - (film.y * camera.window.y + camera.window.w));
    const uint type = camera.meta.x;
    if (type == cameraTypeEquirectangular) {
        const float phi = uv.x * (p.x - p.y) - p.x;
        const float theta = uv.y * (p.z - p.w) - p.z + 0.5 * pi;
        return float3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
    }
    if (type == cameraTypeMirrorBall) {
        const float2 xz = 2.0 * uv - 1.0;
        if (dot(xz, xz) > 1.0)
            return float3(0.0, 0.0, 0.0);
        const float3 normal = float3(xz.x, -sqrt(max(1.0 - dot(xz, xz), 0.0)), xz.y);
        const float3 reflected = 2.0 * -normal.y * normal - float3(0.0, -1.0, 0.0);
        return float3(reflected.y, -reflected.x, reflected.z);
    }
    if (type == cameraTypeEquiangularCubemapFace)
        return normalize(float3(1.0, tan((1.0 - uv.x) * 0.5 * pi - 0.25 * pi), tan(uv.y * 0.5 * pi - 0.25 * pi)));
    if (type == cameraTypeCentralCylindrical) {
        const float phi = -lerp(p.x, p.y, uv.x);
        return normalize(float3(cos(phi), sin(phi), lerp(p.z, p.w, uv.y) / camera.panorama[1].x));
    }
    const bool equidistant = type == cameraTypeFisheyeEquidistant;
    const float2 point = equidistant ? 2.0 * uv - 1.0 : (uv - 0.5) * p.zw;
    const float radius = length(point);
    float theta = radius * p.x * 0.5;
    float limit = 0.5 * p.x;
    if (type == cameraTypeFisheyeEquisolid) {
        theta = 2.0 * asin(saturate(radius / (2.0 * p.y)));
        limit = 2.0 * p.y * sin(0.25 * p.x) < radius ? -1.0 : floatMax;
    } else if (type == cameraTypeFisheyePolynomial) {
        const float squared = radius * radius;
        const float4 powers = float4(radius, squared, squared * radius, squared * squared);
        theta = -(camera.panorama[1].x + dot(powers, float4(camera.panorama[1].yzw, p.y)));
    }
    if (abs(theta) > limit)
        return float3(0.0, 0.0, 0.0);
    const float2 direction = radius > 0.0 ? point / radius : float2(1.0, 0.0);
    return float3(cos(theta), -direction.x * sin(theta), direction.y * sin(theta));
}

//! film is the image position in [0, 1]^2, y down; lensSample is the aperture point; lambda, the hero wavelength, is for a realistic lens only.
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
    } else if (type >= cameraTypeEquirectangular) {
        const float3 direction = panoramaDirection(camera, film);
        if (all(direction == 0.0)) {
            ray.valid = false;
            return ray;
        }
        ray.direction = float3(-direction.y, direction.z, -direction.x);
    } else {
        ray.direction = normalize(float3(sensor / focal, -1.0));
    }

    const float apertureRadius = camera.aperture.x;
    if (apertureRadius > 0.0 && type != cameraTypeOrthographic && type != cameraTypeRealistic) {
        const uint blades = uint(camera.aperture.y);
        const float2 disk = blades >= 3 ? samplePolygon(lensSample, blades, camera.aperture.z) : sampleConcentricDisk(lensSample);
        const float2 lens = disk * float2(camera.aperture.w, 1.0) * apertureRadius;
        const float focus = camera.projection.z;
        const bool panoramic = type >= cameraTypeEquirectangular;
        const float3 focusPoint = ray.direction * (panoramic ? focus : focus / max(-ray.direction.z, 1e-6));
        ray.origin = float3(lens, 0.0);
        if (panoramic) {
            const float3 right = normalize(abs(ray.direction.y) < 0.99 ? cross(ray.direction, float3(0.0, 1.0, 0.0)) : float3(1.0, 0.0, 0.0));
            ray.origin = right * lens.x + cross(right, ray.direction) * lens.y;
        }
        ray.direction = normalize(focusPoint - ray.origin);
    }
    if (type != cameraTypeRealistic)
        ray.origin += ray.direction * camera.clip.x / (type >= cameraTypeEquirectangular || type == cameraTypeFisheye ? 1.0 : max(-ray.direction.z, 1e-6));

    ray.origin = transformPoint(camera.cameraToWorld, ray.origin);
    ray.direction = normalize(transformVector(camera.cameraToWorld, ray.direction));
    return ray;
}

RayCone cameraRayCone(CameraRecord camera) {
    if (camera.meta.x == cameraTypeOrthographic)
        return makeRayCone(camera.clip.z, 0.0);
    return makeRayCone(0.0, camera.projection.w);
}

//! The distance to the far clip plane along a camera ray, or to the far clip sphere of a panorama.
float cameraFarDistance(CameraRecord camera, float3 origin, float3 direction) {
    const float3 eye = float3(camera.cameraToWorld[0].w, camera.cameraToWorld[1].w, camera.cameraToWorld[2].w);
    if (camera.meta.x >= cameraTypeEquirectangular || camera.meta.x == cameraTypeFisheye)
        return max(camera.clip.y - length(origin - eye), 0.0);
    const float3 forward = -normalize(float3(camera.cameraToWorld[0].z, camera.cameraToWorld[1].z, camera.cameraToWorld[2].z));
    return max(camera.clip.y - dot(origin - eye, forward), 0.0) / max(dot(direction, forward), 1e-6);
}

float3 cameraPosition(CameraRecord camera) { return float3(camera.cameraToWorld[0].w, camera.cameraToWorld[1].w, camera.cameraToWorld[2].w); }

//! The film position (y down) of a world point for reprojection, false behind the camera: pinhole for every type but orthographic.
bool projectToFilm(CameraRecord camera, float4 worldToCamera[3], float3 world, out float2 film) {
    const float3 local = transformPoint(worldToCamera, world);
    film = float2(0.0, 0.0);
    float2 sensor;
    if (camera.meta.x == cameraTypeOrthographic)
        sensor = local.xy * (camera.sensor.y / camera.projection.y);
    else {
        if (local.z >= -1e-6 || camera.meta.x >= cameraTypeEquirectangular)
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

//! The clip position of a world point for the rasterizer, jittered as a traced primary ray: y up, depth 0 to 1 from near to far.
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
