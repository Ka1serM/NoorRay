import NoorRHI;
import Raster.Records;
import Wavefront.Records;
import Common.Math;
import Scene.Camera;
import Scene.SceneSurface;

struct VisibilityVaryings {
    float4 position : SV_Position;
    nointerpolation uint id : VISIBILITY;
};

VisibilityVaryings rasterizeVertex(VisibilityArguments arguments, uint vertex) {
    const SceneRecord scene = arguments.scene[0];
    const InstanceData instance = scene.instances[arguments.instance];
    const MeshRecord mesh = scene.meshes[instance.meta.x];
    const float3 world = transformPoint(instance.objectToWorld, loadPosition(mesh, mesh.indices[vertex]));
    VisibilityVaryings output;
    output.position = projectToClip(arguments.camera[0], world, arguments.jitter);
    output.id = arguments.firstTriangle + vertex / 3 + 1;
    return output;
}

//! The draw whose first triangle id is the largest one not above id.
uint findDraw(RasterRecord raster, uint id) {
    uint low = 0;
    uint high = raster.drawCount;
    while (high - low > 1) {
        const uint middle = (low + high) / 2;
        if (raster.draws[middle].x <= id)
            low = middle;
        else
            high = middle;
    }
    return low;
}

enum RasterCoverage { Background, Mesh, NeedsTrace }

//! At a silhouette edge the rasterizer can keep the back face, or miss the triangle that the jittered ray meets, so such a pixel needs a traced ray.
RasterCoverage loadRasterHit(RasterRecord raster, SceneRecord scene, uint2 pixel, float3 origin, float3 direction, out RayHit hit) {
    hit = (RayHit)0;
    const uint stored = raster.visibility.Load(int2(pixel)).x;
    if (stored == 0 || raster.drawCount == 0)
        return RasterCoverage.Background;
    const uint2 draw = raster.draws[findDraw(raster, stored - 1)];
    const uint primitive = stored - 1 - draw.x;
    const InstanceData instance = scene.instances[draw.y];
    const MeshRecord mesh = scene.meshes[instance.meta.x];
    const uint3 corners = loadTriangle(mesh, primitive);
    const float3 p0 = transformPoint(instance.objectToWorld, loadPosition(mesh, corners.x));
    const float3 e1 = transformPoint(instance.objectToWorld, loadPosition(mesh, corners.y)) - p0;
    const float3 e2 = transformPoint(instance.objectToWorld, loadPosition(mesh, corners.z)) - p0;
    const float3 p = cross(direction, e2);
    const float determinant = dot(e1, p);
    const float facing = determinant3(instance.objectToWorld) < 0.0 ? -determinant : determinant;
    if (facing < 1e-20)
        return RasterCoverage.NeedsTrace;
    const float inverse = 1.0 / determinant;
    const float3 s = origin - p0;
    const float3 q = cross(s, e1);
    hit.t = dot(e2, q) * inverse;
    hit.barycentrics = float2(dot(s, p), dot(direction, q)) * inverse;
    if (hit.t <= 0.0 || any(hit.barycentrics < 0.0) || hit.barycentrics.x + hit.barycentrics.y > 1.0)
        return RasterCoverage.NeedsTrace;
    hit.instanceIndex = draw.y;
    hit.instanceId = draw.y;
    hit.primitiveIndex = primitive;
    hit.frontFace = true;
    return RasterCoverage.Mesh;
}
