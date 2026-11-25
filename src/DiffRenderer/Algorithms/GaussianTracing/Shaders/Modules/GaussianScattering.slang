import NoorRHI;
import Common.Frame;
import Common.Math;
import Common.Sampling;
import GaussianList;
import GaussianTransport;
import Scene.GaussianKernel;
import Spectral.Wavelengths;

enum VertexKind : uint { Gaussian, Surface, Environment }

struct PathVertex {
    VertexKind kind;
    uint gaussian;
    float3 origin;
    float3 direction;
    float tMax;
    float3 position;
    float3 normal;
    //! The reflected part of the incoming light; one for the environment.
    float4 albedo;
    //! The light of the light source before the albedo, or the radiance of the environment.
    float4 direct;
    float4 throughput;

    float3 getLeaveOrigin() { return kind == VertexKind.Surface ? position + normal * rayOffset : position; }
};

struct GaussianPath {
    uint count;
    PathVertex vertices[maxVertices];
};

struct Segment {
    uint gaussian;
    float t;
};

//! Chooses an entry of the list by its weight, or none with the transmittance of the whole list.
Segment sampleSegment(Cloud cloud, float3 origin, float3 direction, float tMax, float u) {
    Segment segment;
    segment.gaussian = noGaussian;
    segment.t = tMax;
    float transmittance = 1.0;
    GaussianListEntry after = getListStart();
    for (uint step = 0; step < maxBatches; ++step) {
        const GaussianBatch batch = cloud.next(origin, direction, tMax, after);
        for (uint index = 0; index < batch.count; ++index) {
            const GaussianListEntry entry = batch.entries[index];
            transmittance *= 1.0 - cloud.findPeak(entry.gaussian, origin, direction).alpha;
            if (u < 1.0 - transmittance) {
                segment.gaussian = entry.gaussian;
                segment.t = entry.t;
                return segment;
            }
            if (transmittance < minTransmittance)
                return segment;
        }
        if (!batch.isFull)
            return segment;
        after = batch.entries[batch.count - 1];
    }
    return segment;
}

float getShadowTransmittance(Cloud cloud, float3 origin, float3 direction) {
    float transmittance = 1.0;
    GaussianListEntry after = getListStart();
    for (uint step = 0; step < maxBatches; ++step) {
        const GaussianBatch batch = cloud.next(origin, direction, 1.0, after);
        for (uint index = 0; index < batch.count; ++index) {
            transmittance *= 1.0 - cloud.findPeak(batch.entries[index].gaussian, origin, direction).alpha;
            if (transmittance < minTransmittance)
                return transmittance;
        }
        if (!batch.isFull)
            return transmittance;
        after = batch.entries[batch.count - 1];
    }
    return transmittance;
}

float4 getDirect(Scene scene, PathVertex vertex) {
    const float3 origin = vertex.getLeaveOrigin();
    const float geometry = getLightGeometry(vertex.position, scene.arguments.light.xyz, vertex.normal, vertex.kind == VertexKind.Surface);
    if (geometry <= 0.0 || !scene.isLitThroughMesh(origin))
        return float4(0.0);
    return scene.lightSpectrum * (geometry * getShadowTransmittance(scene.cloud, origin, scene.arguments.light.xyz - origin));
}

//! The density of both directions cancels: the isotropic phase function, and the cosine over pi.
float3 sampleDirection(PathVertex vertex, inout Random random) {
    const float2 u = float2(random.next(), random.next());
    return vertex.kind == VertexKind.Surface ? makeFrame(vertex.normal).toWorld(sampleCosineHemisphere(u)) : sampleUniformSphere(u);
}

GaussianPath tracePath(Scene scene, float3 cameraDirection, inout Random random) {
    GaussianPath path;
    path.count = 0;
    float3 origin = scene.arguments.cameraOrigin.xyz;
    float3 direction = cameraDirection;
    float4 throughput = float4(1.0);
    while (path.count < maxVertices) {
        const Surface surface = scene.traceMesh(origin, direction);
        const Segment segment = sampleSegment(scene.cloud, origin, direction, surface.t, random.next());
        PathVertex vertex = {};
        vertex.gaussian = segment.gaussian;
        vertex.origin = origin;
        vertex.direction = direction;
        vertex.tMax = surface.t;
        vertex.throughput = throughput;
        if (segment.gaussian == noGaussian && !surface.valid) {
            vertex.kind = VertexKind.Environment;
            vertex.albedo = float4(1.0);
            vertex.direct = scene.getEnvironment();
            path.vertices[path.count] = vertex;
            ++path.count;
            return path;
        }
        const bool scattered = segment.gaussian != noGaussian;
        vertex.kind = scattered ? VertexKind.Gaussian : VertexKind.Surface;
        vertex.position = origin + direction * segment.t;
        vertex.normal = surface.normal;
        vertex.albedo = scene.wavelengths.upsample(scattered ? decodeColor(scene.cloud.loadEncodedColor(segment.gaussian)) : surface.albedo);
        vertex.direct = getDirect(scene, vertex);
        path.vertices[path.count] = vertex;
        ++path.count;
        throughput *= vertex.albedo;
        origin = vertex.getLeaveOrigin();
        direction = sampleDirection(vertex, random);
    }
    return path;
}

float4 getScattering(Scene scene, float3 cameraDirection, inout Random random) {
    const GaussianPath path = tracePath(scene, cameraDirection, random);
    float4 radiance = float4(0.0);
    for (uint index = 0; index < path.count; ++index)
        radiance += path.vertices[index].throughput * path.vertices[index].albedo * path.vertices[index].direct;
    return radiance;
}

//! Adds the gradient of contribution * log(light geometry * shadow transmittance) and returns the part of the vertex position.
float3 addLightGradient(Scene scene, PathVertex vertex, float contribution) {
    const bool surface = vertex.kind == VertexKind.Surface;
    const float3 light = scene.arguments.light.xyz;
    const float geometry = getLightGeometry(vertex.position, light, vertex.normal, surface);
    var position = diffPair(vertex.position);
    bwd_diff(getLightGeometry)(position, light, vertex.normal, surface, contribution / geometry);
    float3 positionGradient = position.d;
    const float3 origin = vertex.getLeaveOrigin();
    const float3 direction = light - origin;
    float transmittance = 1.0;
    GaussianListEntry after = getListStart();
    for (uint step = 0; step < maxBatches; ++step) {
        const GaussianBatch batch = scene.cloud.next(origin, direction, 1.0, after);
        for (uint index = 0; index < batch.count; ++index) {
            const uint gaussian = batch.entries[index].gaussian;
            const float visible = 1.0 - scene.cloud.findPeak(gaussian, origin, direction).alpha;
            const RayGradient gradient = scene.cloud.addPeak(gaussian, origin, direction, 0.0, -contribution / visible);
            positionGradient += gradient.origin - gradient.direction;
            transmittance *= visible;
            if (transmittance < minTransmittance)
                return positionGradient;
        }
        if (!batch.isFull)
            return positionGradient;
        after = batch.entries[batch.count - 1];
    }
    return positionGradient;
}

//! Adds the gradient of contribution * log(weight of the chosen entry) and of the vertex position, and returns the part of the segment origin.
float3 addSegmentGradient(Cloud cloud, PathVertex vertex, float contribution, float3 positionGradient) {
    float3 originGradient = float3(0.0);
    if (vertex.kind == VertexKind.Gaussian)
        originGradient = positionGradient;
    else if (vertex.kind == VertexKind.Surface)
        originGradient = positionGradient - vertex.normal * (dot(vertex.direction, positionGradient) / dot(vertex.normal, vertex.direction));
    float transmittance = 1.0;
    GaussianListEntry after = getListStart();
    for (uint step = 0; step < maxBatches; ++step) {
        const GaussianBatch batch = cloud.next(vertex.origin, vertex.direction, vertex.tMax, after);
        for (uint index = 0; index < batch.count; ++index) {
            const uint gaussian = batch.entries[index].gaussian;
            const float alpha = cloud.findPeak(gaussian, vertex.origin, vertex.direction).alpha;
            if (gaussian == vertex.gaussian)
                return originGradient
                       + cloud.addPeak(gaussian, vertex.origin, vertex.direction, dot(positionGradient, vertex.direction), contribution / alpha).origin;
            originGradient += cloud.addPeak(gaussian, vertex.origin, vertex.direction, 0.0, -contribution / (1.0 - alpha)).origin;
            transmittance *= 1.0 - alpha;
            if (transmittance < minTransmittance)
                return originGradient;
        }
        if (!batch.isFull)
            return originGradient;
        after = batch.entries[batch.count - 1];
    }
    return originGradient;
}

//! The choices of the path are detached: each vertex has the weight w / detach(w), so its gradient is the radiance times the gradient of log(w).
void addScatteringGradient(Scene scene, float3 cameraDirection, float4 weight, inout Random random) {
    const GaussianPath path = tracePath(scene, cameraDirection, random);
    float4 incoming = float4(0.0);
    float3 positionGradient = float3(0.0);
    for (uint remaining = path.count; remaining > 0; --remaining) {
        const PathVertex vertex = path.vertices[remaining - 1];
        const float4 arriving = vertex.direct + incoming;
        const float4 outgoing = vertex.albedo * arriving;
        const float4 pathWeight = weight * vertex.throughput;
        if (vertex.kind == VertexKind.Gaussian)
            scene.cloud.addColor(vertex.gaussian, getColorWeight(scene.wavelengths, pathWeight * arriving));
        if (vertex.kind != VertexKind.Environment && any(vertex.direct > 0.0))
            positionGradient += addLightGradient(scene, vertex, dot(pathWeight, vertex.albedo * vertex.direct));
        positionGradient = addSegmentGradient(scene.cloud, vertex, dot(pathWeight, outgoing), positionGradient);
        incoming = outgoing;
    }
}
