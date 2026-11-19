import NoorRHI;
import SceneRecords;
import Common.Math;
import Common.Sampling;
import Common.Frame;
import Lights.Environment;

//! radiance is linear sRGB and already divided by nothing: the caller weights it with the BSDF and divides by pdf. pdf is per solid angle and includes the light selection; delta lights have pdf equal to their selection probability.
struct LightSample {
    float3 direction;
    float distance;
    float3 radiance;
    float pdf;
    uint light;
    bool delta;
};

LightSample emptyLightSample() {
    LightSample sample;
    sample.direction = float3(0.0, 0.0, 1.0);
    sample.distance = 0.0;
    sample.radiance = float3(0.0, 0.0, 0.0);
    sample.pdf = 0.0;
    sample.light = invalidIndex;
    sample.delta = false;
    return sample;
}

static const uint environmentLight = 0xfffffffeu;

float environmentProbability(SceneRecord scene) { return scene.environmentParams.y; }

float spotMask(LightRecord light, float3 toReceiver) {
    const float cosAngle = dot(light.direction.xyz, toReceiver);
    return smoothstep(light.direction.w, light.direction.w + light.tangent.w, cosAngle);
}

//! The solid angle pdf of an area light point seen from origin, from its area pdf.
float areaToSolidAngle(float areaPdf, float distanceSquared, float cosine) { return cosine > 0.0 ? areaPdf * distanceSquared / cosine : 0.0; }

LightSample sampleLightRecord(LightRecord light, float3 origin, float2 u) {
    LightSample sample = emptyLightSample();
    const uint type = light.meta.x;
    switch (type) {
    case lightTypePoint:
    case lightTypeSpot: {
        const float3 toLight = light.position.xyz - origin;
        const float distanceSquared = dot(toLight, toLight);
        if (distanceSquared <= 0.0)
            return sample;
        sample.distance = sqrt(distanceSquared);
        sample.direction = toLight / sample.distance;
        const float mask = type == lightTypeSpot ? spotMask(light, -sample.direction) : 1.0;
        sample.radiance = light.radiance.xyz * (mask / distanceSquared);
        sample.pdf = 1.0;
        sample.delta = true;
        return sample;
    }
    case lightTypeSphere: {
        const float3 toCenter = light.position.xyz - origin;
        const float distanceSquared = dot(toCenter, toCenter);
        const float radius = light.position.w;
        if (distanceSquared <= radius * radius)
            return sample;
        const float distance = sqrt(distanceSquared);
        const float sinThetaMaxSquared = radius * radius / distanceSquared;
        const float cosThetaMax = safeSqrt(1.0 - sinThetaMaxSquared);
        const Frame frame = makeFrame(toCenter / distance);
        sample.direction = frame.toWorld(sampleUniformCone(u, cosThetaMax));
        const float b = dot(toCenter, sample.direction);
        const float c = distanceSquared - radius * radius;
        sample.distance = b - safeSqrt(b * b - c);
        sample.radiance = light.radiance.xyz;
        sample.pdf = uniformConePdf(cosThetaMax);
        return sample;
    }
    case lightTypeRect:
    case lightTypeDisk: {
        const float2 offset = type == lightTypeRect ? 2.0 * u - 1.0 : sampleConcentricDisk(u);
        const float3 point = light.position.xyz + light.tangent.xyz * offset.x + light.bitangent.xyz * offset.y;
        const float3 toLight = point - origin;
        const float distanceSquared = dot(toLight, toLight);
        sample.distance = sqrt(distanceSquared);
        sample.direction = toLight / sample.distance;
        const float cosine = dot(light.direction.xyz, -sample.direction);
        if (cosine <= 0.0 || light.radiance.w <= 0.0)
            return emptyLightSample();
        sample.radiance = light.radiance.xyz;
        sample.pdf = areaToSolidAngle(1.0 / light.radiance.w, distanceSquared, cosine);
        return sample;
    }
    case lightTypeDirectional: {
        const float cosThetaMax = light.direction.w;
        sample.distance = floatMax;
        if (cosThetaMax >= 1.0 - 1e-7) {
            sample.direction = -light.direction.xyz;
            sample.radiance = light.radiance.xyz;
            sample.pdf = 1.0;
            sample.delta = true;
            return sample;
        }
        sample.direction = makeFrame(-light.direction.xyz).toWorld(sampleUniformCone(u, cosThetaMax));
        sample.radiance = light.radiance.xyz / (pi * (1.0 - cosThetaMax * cosThetaMax));
        sample.pdf = uniformConePdf(cosThetaMax);
        return sample;
    }
    default: {
        const float3 b = sampleTriangleBarycentrics(u);
        const float3 point = light.position.xyz + light.tangent.xyz * b.y + light.bitangent.xyz * b.z;
        const float3 toLight = point - origin;
        const float distanceSquared = dot(toLight, toLight);
        sample.distance = sqrt(distanceSquared);
        sample.direction = toLight / sample.distance;
        const float3 normal = normalize(cross(light.tangent.xyz, light.bitangent.xyz));
        const float cosine = abs(dot(normal, sample.direction));
        if (light.radiance.w <= 0.0)
            return emptyLightSample();
        sample.radiance = light.radiance.xyz;
        sample.pdf = areaToSolidAngle(1.0 / light.radiance.w, distanceSquared, cosine);
        return sample;
    }
    }
}

uint selectLight(SceneRecord scene, float u, out float selection) {
    const float environment = environmentProbability(scene);
    const uint lightCount = scene.counts.y;
    if (u < environment || lightCount == 0) {
        selection = environment;
        return environment > 0.0 ? environmentLight : invalidIndex;
    }
    const uint index = sampleAlias(scene.lightAlias, lightCount, (u - environment) / (1.0 - environment), selection);
    selection *= 1.0 - environment;
    return index;
}


float lightSelection(SceneRecord scene, uint light) {
    const float environment = environmentProbability(scene);
    return light == environmentLight ? environment : scene.lightAlias[light].pdf * (1.0 - environment);
}

LightSample sampleLight(SceneRecord scene, uint light, float3 origin, float3 u) {
    if (light != environmentLight) {
        LightSample sample = sampleLightRecord(scene.lights[light], origin, u.xy);
        sample.light = light;
        return sample;
    }
    LightSample sample = emptyLightSample();
    sample.direction = sampleEnvironment(scene, u, sample.pdf);
    sample.distance = floatMax;
    sample.radiance = environmentRadiance(scene, sample.direction);
    sample.light = environmentLight;
    return sample;
}

//! One light sample: the environment with its selection probability, else a light from the alias table.
LightSample sampleLights(SceneRecord scene, float3 origin, float4 u) {
    float selection;
    const uint light = selectLight(scene, u.w, selection);
    if (light == invalidIndex)
        return emptyLightSample();
    LightSample sample = sampleLight(scene, light, origin, u.xyz);
    sample.pdf *= selection;
    return sample;
}

//! The pdf with which sampleLights picks a direction that reaches a point of an area light (or a triangle emitter) at distance with the given emitter cosine.
float lightPdf(SceneRecord scene, uint index, float3 origin, float3 direction, float distance, float emitterCosine) {
    const float selection = lightSelection(scene, index);
    if (index == environmentLight)
        return selection * environmentPdf(scene, direction);
    const LightRecord light = scene.lights[index];
    switch (light.meta.x) {
    case lightTypeSphere: {
        const float3 toCenter = light.position.xyz - origin;
        const float distanceSquared = dot(toCenter, toCenter);
        const float radius = light.position.w;
        if (distanceSquared <= radius * radius)
            return 0.0;
        return selection * uniformConePdf(safeSqrt(1.0 - radius * radius / distanceSquared));
    }
    case lightTypeDirectional:
        return light.direction.w >= 1.0 - 1e-7 ? 0.0 : selection * uniformConePdf(light.direction.w);
    case lightTypePoint:
    case lightTypeSpot:
        return 0.0;
    default:
        return light.radiance.w > 0.0 ? selection * areaToSolidAngle(1.0 / light.radiance.w, distance * distance, abs(emitterCosine)) : 0.0;
    }
}

struct LightHit {
    float distance;
    uint light;
    float3 radiance;
    float emitterCosine;
};

//! The nearest analytic area light (rect, disk, sphere) along a ray before tMax. Point, spot and triangle lights are not intersected here.
LightHit intersectLights(SceneRecord scene, float3 origin, float3 direction, float tMax) {
    LightHit hit;
    hit.distance = tMax;
    hit.light = invalidIndex;
    hit.radiance = float3(0.0, 0.0, 0.0);
    hit.emitterCosine = 0.0;
    for (uint index = 0; index < scene.counts.y; ++index) {
        const LightRecord light = scene.lights[index];
        const uint type = light.meta.x;
        if (type == lightTypeTriangle)
            break;
        if (type == lightTypeSphere) {
            const float3 toOrigin = origin - light.position.xyz;
            const float b = dot(toOrigin, direction);
            const float c = dot(toOrigin, toOrigin) - light.position.w * light.position.w;
            const float discriminant = b * b - c;
            if (discriminant < 0.0 || c <= 0.0)
                continue;
            const float t = -b - sqrt(discriminant);
            if (t > 0.0 && t < hit.distance) {
                hit.distance = t;
                hit.light = index;
                hit.radiance = light.radiance.xyz;
                hit.emitterCosine = 1.0;
            }
            continue;
        }
        if (type != lightTypeRect && type != lightTypeDisk)
            continue;
        const float3 normal = light.direction.xyz;
        const float denominator = dot(normal, direction);
        if (denominator >= 0.0)
            continue;
        const float t = dot(light.position.xyz - origin, normal) / denominator;
        if (t <= 0.0 || t >= hit.distance)
            continue;
        const float3 local = origin + direction * t - light.position.xyz;
        const float x = dot(local, light.tangent.xyz) / dot(light.tangent.xyz, light.tangent.xyz);
        const float y = dot(local, light.bitangent.xyz) / dot(light.bitangent.xyz, light.bitangent.xyz);
        const bool inside = type == lightTypeRect ? (abs(x) <= 1.0 && abs(y) <= 1.0) : (x * x + y * y <= 1.0);
        if (!inside)
            continue;
        hit.distance = t;
        hit.light = index;
        hit.radiance = light.radiance.xyz;
        hit.emitterCosine = -denominator;
    }
    return hit;
}

//! Radiance from directional lights with a finite angle that a ray leaving the scene sees.
float3 directionalRadiance(SceneRecord scene, float3 direction, out uint light) {
    light = invalidIndex;
    float3 total = float3(0.0, 0.0, 0.0);
    for (uint index = 0; index < scene.counts.y; ++index) {
        const LightRecord record = scene.lights[index];
        if (record.meta.x == lightTypeTriangle)
            break;
        if (record.meta.x != lightTypeDirectional || record.direction.w >= 1.0 - 1e-7)
            continue;
        if (dot(-record.direction.xyz, direction) >= record.direction.w) {
            total += record.radiance.xyz / (pi * (1.0 - record.direction.w * record.direction.w));
            light = index;
        }
    }
    return total;
}
