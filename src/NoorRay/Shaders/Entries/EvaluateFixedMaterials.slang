import NoorRHI;
import Wavefront.Records;
import Wavefront.Queues;
import Wavefront.PathState;
import Common.RayCone;
import Materials.Material;
import Scene.SceneSurface;

//! The material kernel of the materials without a graph, which never reaches the graph code.
[shader("compute")]
[numthreads(64, 1, 1)]
void evaluateFixedMaterials(uint3 id : SV_DispatchThreadID) {
    const WavefrontArguments arguments = rootArguments<WavefrontArguments>();
    const WavefrontFrame frame = arguments.frame[0];
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
    frame.pool.closures[slot] = fixedClosure(scene, surface.context, surface.material);
}
