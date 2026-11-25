import NoorRHI;
import Wavefront.Records;
import Wavefront.PathState;
import Wavefront.Queues;
import Wavefront.Direct;
import Wavefront.SurfaceShading;
import Common.Math;
import Common.LinkConstants;
import Common.RayCone;
import Lights.Lights;
import Lights.Environment;
import Bsdf.Principled;
import Materials.Closure;
import Scene.SceneSurface;
import Spectral.Wavelengths;

//! Every analytic light, unshadowed, from its sample at the centre of its emitter. Triangle lights follow the analytic ones.
float4 analyticLighting(SceneRecord scene, PrincipledBsdf bsdf, float3 position, Wavelengths wavelengths) {
    float4 total = float4(0.0, 0.0, 0.0, 0.0);
    for (uint index = 0; index < scene.counts.y; ++index) {
        const LightRecord light = scene.lights[index];
        if (light.meta.x == lightTypeTriangle)
            break;
        const LightSample sample = sampleLightRecord(light, position, float2(0.5, 0.5));
        if (sample.pdf <= 0.0)
            continue;
        total += bsdf.evaluate(sample.direction).value * wavelengths.upsample(sample.radiance) / sample.pdf;
    }
    return total;
}

void endRasterPath(WavefrontFrame frame, uint slot, PathState path) {
    finishPath(frame.film, path);
    storePath(frame.pool, slot, path);
    pushFree(frame, slot);
}

//! The surface, material and BSDF as the path tracer, shaded in RGB without secondary rays.
void shadeRasterPath(WavefrontFrame frame, uint parity, uint slot) {
    const SceneRecord scene = frame.scene[0];
    PathState path = loadPath(frame.pool, slot);
    const PathHit hit = loadHit(frame.pool, slot);
    const Wavelengths wavelengths = pathWavelengths(scene, path);
    const Surface surface = loadSurface(scene, hit.instance, hit.primitive, hit.barycentrics, path.direction, hit.t, propagate(path.cone, hit.t));
    const ShadingContext context = surface.context;

    const PackedClosure closure = frame.pool.closures[slot];
    if (hasAlphaTest && path.hasFlag(pathFlagAlphaCandidate) && !passesOpacity(frame, path, closureOpacity(closure), 2)) {
        addRadiance(frame, path, wavelengths, path.throughput * wavelengths.upsample(backgroundRadiance(scene, path.direction)));
        endRasterPath(frame, slot, path);
        return;
    }

    const BsdfInputs inputs = unpackClosure(closure);
    const float3 view = -path.direction;
    const PrincipledBsdf bsdf = makePrincipled(inputs, view, context.geometricNormal.xyz, isBackfacing(context), wavelengths, sceneEnergyLut(scene), false);
    const float3 normal = safeNormalize(inputs.normalRotation.xyz);
    writePrimaryAovs(frame.film, path, saturate(wavelengths.toFilm(bsdf.albedo())), normal, hit.t, hit.instance);

    const float3 position = spawnOrigin(surface, view);
    float4 radiance = emittedRadiance(inputs, wavelengths) + analyticLighting(scene, bsdf, position, wavelengths);
    radiance += bsdf.albedo() * wavelengths.upsample(environmentRadiance(scene, normal));
    addRadiance(frame, path, wavelengths, path.throughput * radiance);
    endRasterPath(frame, slot, path);
}
