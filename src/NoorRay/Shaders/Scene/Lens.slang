import NoorRHI;
import SceneRecords;
import Common.Math;
import Spectral.Dispersion;

bool intersectLensElement(LensElementRecord element, inout float3 origin, float3 direction) {
    const float radius = element.geometry.x;
    const float vertexZ = element.geometry.y;
    float t;
    if (radius == 0.0) {
        if (direction.z == 0.0)
            return false;
        t = (vertexZ - origin.z) / direction.z;
    } else {
        const float3 offset = origin - float3(0.0, 0.0, vertexZ + radius);
        const float b = dot(offset, direction);
        const float c = dot(offset, offset) - radius * radius;
        const float discriminant = b * b - c;
        if (discriminant < 0.0)
            return false;
        const float root = sqrt(discriminant);
        const bool closer = (direction.z > 0.0) != (radius < 0.0);
        t = closer ? -b - root : -b + root;
    }
    if (!(t > 0.0))
        return false;
    const float3 hit = origin + direction * t;
    const float aperture = element.geometry.z;
    if (aperture > 0.0 && dot(hit.xy, hit.xy) > aperture * aperture)
        return false;
    origin = hit;
    return true;
}

bool refractLens(float3 incident, float3 normal, float eta, out float3 transmitted) {
    transmitted = incident;
    if (dot(incident, normal) > 0.0)
        normal = -normal;
    const float cosIncident = -dot(normal, incident);
    const float sinSquared = eta * eta * max(0.0, 1.0 - cosIncident * cosIncident);
    if (sinSquared >= 1.0)
        return false;
    transmitted = normalize(eta * incident + (eta * cosIncident - sqrt(1.0 - sinSquared)) * normal);
    return true;
}

float lensIor(LensElementRecord element, float lambda) {
    return element.medium.y > 0.0 ? cauchyIor(element.medium.x, element.medium.y, lambda) : element.medium.x;
}

//! Traces a ray from the film toward the scene through all elements, back to front, at one wavelength. dispersive reports a medium with an Abbe number.
bool traceLens(CameraRecord camera, inout float3 origin, inout float3 direction, float lambda, out bool dispersive) {
    dispersive = false;
    const uint count = camera.meta.y;
    for (uint step = 0; step < count; ++step) {
        const uint index = count - 1 - step;
        const LensElementRecord element = camera.lensElements[index];
        if (!intersectLensElement(element, origin, direction))
            return false;
        const float radius = element.geometry.x;
        if (radius == 0.0)
            continue;
        const float inside = lensIor(element, lambda);
        const float outside = index > 0 ? lensIor(camera.lensElements[index - 1], lambda) : 1.0;
        dispersive = dispersive || element.medium.y > 0.0 || (index > 0 && camera.lensElements[index - 1].medium.y > 0.0);
        const float3 normal = normalize(origin - float3(0.0, 0.0, element.geometry.y + radius));
        float3 transmitted;
        if (!refractLens(direction, normal, inside / outside, transmitted))
            return false;
        direction = transmitted;
    }
    return count != 0;
}
