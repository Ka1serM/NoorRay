import NoorRHI;
import Wavefront.Records;
import Wavefront.Queues;
import Wavefront.PathState;
import Common.RayCone;
import Common.Random;
import Materials.Material;
import Materials.Graph;
import Scene.SceneSurface;

//! Points of a triangle, as the last two barycentrics, whose mean emission stands for the triangle in the light sampling.
static const float2 emitterSamples[4] = { float2(1.0 / 3.0, 1.0 / 3.0), float2(1.0 / 6.0, 1.0 / 6.0), float2(2.0 / 3.0, 1.0 / 6.0), float2(1.0 / 6.0, 2.0 / 3.0) };

//! The mean emission of a triangle light seen along direction: Cycles evaluates it with no ray type set.
float3 emitterSide(WavefrontFrame frame, LightRecord record, uint index, float3 direction) {
    const SceneRecord scene = frame.scene[0];
    const CameraRecord view = frame.camera[0];
    float3 sum = float3(0.0, 0.0, 0.0);
    [unroll]
    for (uint sample = 0; sample < 4; ++sample) {
        const Surface surface = loadSurface(scene, record.meta.y, record.meta.z, emitterSamples[sample], direction, 0.0, makeRayCone(0.0, 0.0));
        ShadingContext context = surface.context;
        setCamera(context, view);
        context.rayType = float4(0.0, 0.0, 0.0, 0.0);
        setGraphTrace(frame.scene, record.meta.y, record.meta.z, hashCombine(index, sample));
        const MaterialRecord material = scene.materials[surface.material];
        const MaterialGraphOutput output = evaluateMaterialGraph(context, graphMaterial(scene.graphParameters + material.graph.y, material.textures));
        sum += max(output.emission.xyz, 0.0) * max(output.emission.w, 0.0) * saturate(1.0 - output.transparency);
    }
    return sum * 0.25;
}

//! The emission that next event estimation sees on each face of one triangle light, as a shader may tell the faces apart by Backfacing.
void bakeEmitter(WavefrontFrame frame, uint program, uint index) {
    const SceneRecord scene = frame.scene[0];
    const uint2 range = scene.emitterRanges[program];
    if (index >= range.y)
        return;
    const uint light = scene.emitterLights[range.x + index];
    LightRecord record = scene.lights[light];
    const float3 facing = normalize(cross(record.tangent.xyz, record.bitangent.xyz));
    const float3 front = emitterSide(frame, record, index, -facing);
    const float3 back = emitterSide(frame, record, index, facing);
    record.radiance.xyz = front;
    record.position.w = back.x;
    record.tangent.w = back.y;
    record.bitangent.w = back.z;
    scene.lights[light] = record;
}

//! The pipeline cache compiles this kernel per material graph with evaluateMaterialGraph as the graph; evaluateFixedMaterials does the rest.
[shader("compute")]
[numthreads(64, 1, 1)]
void evaluateMaterials(uint3 id : SV_DispatchThreadID) {
    const WavefrontArguments arguments = rootArguments<WavefrontArguments>();
    const WavefrontFrame frame = arguments.frame[0];
    if ((arguments.bin & materialSourceEmitters) != 0) {
        bakeEmitter(arguments.frame[0], (arguments.bin & ~materialSourceEmitters) - firstGraphBin, id.y * emitterRowThreads + id.x);
        return;
    }
    const uint bin = arguments.bin & ~materialSourceShadow;
    if (id.x >= binnedCount(frame, bin))
        return;
    const uint slot = binnedSlot(frame, bin, id.x);
    const SceneRecord scene = frame.scene[0];
    const PathHit hit = loadHit(frame.pool, slot);
    const bool shadow = (arguments.bin & materialSourceShadow) != 0;
    const float3 direction = shadow ? frame.pool.shadowDirection[slot].xyz : frame.pool.direction[slot].xyz;
    const RayCone cone = shadow ? makeRayCone(0.0, 0.0) : propagate(unpackRayCone(asuint(frame.pool.origin[slot].w)), hit.t);
    const Surface surface = loadSurface(scene, hit.instance, hit.primitive, hit.barycentrics, direction, hit.t, cone);
    ShadingContext context = surface.context;
    const CameraRecord view = frame.camera[0];
    setCamera(context, view);
    const uint flags = frame.pool.state[slot].z;
    //! A shadow ray sees the light path of its vertex and, as in Cycles, the depth after it, which the path has already taken.
    const uint2 lightPaths = frame.pool.lightPath[slot];
    const uint depth = flags & pathDepthMask;
    const uint lightPath = shadow ? lightPaths.y : lightPaths.x;
    const bool camera = depth == 0 && !shadow;
    //! Cycles starts a perspective camera ray at the near clip plane, so its ray length leaves out the part before it.
    if (camera && context.objectInfo.z != 0.0)
        context.incoming.w -= view.clip.x / max(dot(context.incoming.xyz, normalize(view.worldToCamera[2].xyz)), 1e-6);
    context.rayType = float4(camera ? 1.0 : 0.0, shadow ? 1.0 : 0.0, (lightPath & lightPathDiffuse) != 0 ? 1.0 : 0.0, (lightPath & lightPathGlossy) != 0 ? 1.0 : 0.0);
    const bool scattered = (lightPath & (lightPathDiffuse | lightPathGlossy)) != 0;
    const bool transmitted = (lightPath & lightPathTransmission) != 0;
    context.rayEvent = float4((lightPath & lightPathSingular) != 0 ? 1.0 : 0.0, scattered && !transmitted ? 1.0 : 0.0, transmitted ? 1.0 : 0.0,
                              (lightPath & lightPathVolumeScatter) != 0 ? 1.0 : 0.0);
    context.rayDepth = float4(float(depth), float(lightPathCount(lightPath, lightPathDiffuseShift)), float(lightPathCount(lightPath, lightPathGlossyShift)),
                              float(lightPathCount(lightPath, lightPathTransmissionShift)));
    context.objectInfo.y = float(lightPathCount(lightPath, lightPathTransparentShift));
    setGraphTrace(frame.scene, hit.instance, hit.primitive, hashCombine(hashCombine(slot, frame.settings.z), flags));
    frame.pool.closures[slot] = graphClosure(scene, context, surface.material);
}
