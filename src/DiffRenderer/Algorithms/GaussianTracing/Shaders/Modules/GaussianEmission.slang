import NoorRHI;
import Common.Math;
import GaussianList;
import GaussianTransport;
import Scene.GaussianKernel;
import Spectral.Wavelengths;

//! Emissive Gaussians cast no shadows, as in the path tracer, so nothing behind the list depends on the parameters.
float4 getBackground(Scene scene, Surface surface, float3 origin, float3 direction) {
    if (!surface.valid)
        return scene.getEnvironment();
    const float3 position = origin + direction * surface.t + surface.normal * rayOffset;
    if (!scene.isLitThroughMesh(position))
        return float4(0.0);
    return scene.wavelengths.upsample(surface.albedo) * scene.lightSpectrum * getLightGeometry(position, scene.arguments.light.xyz, surface.normal, true);
}

float4 getEmission(Scene scene, float3 origin, float3 direction) {
    const Surface surface = scene.traceMesh(origin, direction);
    float3 color = float3(0.0);
    float transmittance = 1.0;
    GaussianListEntry after = getListStart();
    for (uint step = 0; step < maxBatches && transmittance >= minTransmittance; ++step) {
        const GaussianBatch batch = scene.cloud.next(origin, direction, surface.t, after);
        for (uint index = 0; index < batch.count && transmittance >= minTransmittance; ++index) {
            const uint gaussian = batch.entries[index].gaussian;
            const float alpha = scene.cloud.findPeak(gaussian, origin, direction).alpha;
            color += decodeColor(scene.cloud.loadEncodedColor(gaussian)) * (alpha * transmittance);
            transmittance *= 1.0 - alpha;
        }
        if (!batch.isFull)
            break;
        after = batch.entries[batch.count - 1];
    }
    return scene.wavelengths.upsample(color) + getBackground(scene, surface, origin, direction) * transmittance;
}

struct BatchStart {
    GaussianListEntry after;
    float transmittance;
};

//! The forward sweep keeps the start of each batch; the backward sweep goes from the last Gaussian to the first with the radiance behind each one.
void addEmissionGradient(Scene scene, float3 origin, float3 direction, float4 weight) {
    const Surface surface = scene.traceMesh(origin, direction);
    BatchStart starts[maxBatches];
    uint batches = 0;
    BatchStart start;
    start.after = getListStart();
    start.transmittance = 1.0;
    while (batches < maxBatches && start.transmittance >= minTransmittance) {
        starts[batches] = start;
        ++batches;
        const GaussianBatch batch = scene.cloud.next(origin, direction, surface.t, start.after);
        for (uint index = 0; index < batch.count && start.transmittance >= minTransmittance; ++index)
            start.transmittance *= 1.0 - scene.cloud.findPeak(batch.entries[index].gaussian, origin, direction).alpha;
        if (!batch.isFull)
            break;
        start.after = batch.entries[batch.count - 1];
    }
    const float3 colorWeight = getColorWeight(scene.wavelengths, weight);
    float behind = dot(weight, getBackground(scene, surface, origin, direction));
    for (uint remaining = batches; remaining > 0; --remaining) {
        const BatchStart first = starts[remaining - 1];
        const GaussianBatch batch = scene.cloud.next(origin, direction, surface.t, first.after);
        float alphas[gaussianListBatch];
        float transmittances[gaussianListBatch];
        float transmittance = first.transmittance;
        uint count = 0;
        while (count < batch.count && transmittance >= minTransmittance) {
            alphas[count] = scene.cloud.findPeak(batch.entries[count].gaussian, origin, direction).alpha;
            transmittances[count] = transmittance;
            transmittance *= 1.0 - alphas[count];
            ++count;
        }
        for (uint index = count; index > 0; --index) {
            const uint gaussian = batch.entries[index - 1].gaussian;
            const float alpha = alphas[index - 1];
            const float emitted = dot(colorWeight, decodeColor(scene.cloud.loadEncodedColor(gaussian)));
            scene.cloud.addPeak(gaussian, origin, direction, 0.0, transmittances[index - 1] * (emitted - behind));
            scene.cloud.addColor(gaussian, colorWeight * (alpha * transmittances[index - 1]));
            behind = lerp(behind, emitted, alpha);
        }
    }
}
