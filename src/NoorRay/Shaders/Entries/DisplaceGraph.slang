import NoorRHI;
import SceneRecords;
import Common.Frame;
import Common.Math;
import Materials.Graph;
#include "NoorRay/Shaders/Materials/DisplacementRecords.h"

float2 loadPointUv(Ptr<uint> stream, uint element) { return isNull(stream) ? float2(0.0, 0.0) : unpackHalf2(stream[element]); }

//! A rest vertex as Cycles shades it for its displacement: seen against its normal, with no footprint.
ShadingContext displacementContext(DisplacementGraphRecord record, DisplacementRecord mesh, uint index) {
    const uint vertex = record.points[index].x;
    const uint corner = record.points[index].y;
    const float3 position = loadFloat3(mesh.restPositions, vertex);
    const float3 normal = safeNormalize(transformNormal(record.worldToObject, loadFloat3(record.normals, 2 * index)));
    const float3 geometricNormal = safeNormalize(transformNormal(record.worldToObject, loadFloat3(record.normals, 2 * index + 1)));
    const uint uvElement = (record.cornerStreams & meshCornerUvs) != 0 ? corner : vertex;
    ShadingContext context;
    context.position = float4(transformPoint(record.objectToWorld, position), 1.0);
    context.normal = float4(normal, 0.0);
    context.geometricNormal = float4(geometricNormal, 0.0);
    context.tangent = float4(makeFrame(normal).tangent, 1.0);
    if (!isNull(mesh.restTangents)) {
        const float4 tangent = unpackTangent(mesh.restTangents[(record.cornerStreams & meshCornerTangents) != 0 ? corner : vertex]);
        context.tangent = float4(safeNormalize(transformVector(record.objectToWorld, tangent.xyz)), tangent.w);
    }
    context.incoming = float4(geometricNormal, 1.0);
    context.uv01 = float4(loadPointUv(record.uv0, uvElement), loadPointUv(record.uv1, uvElement));
    context.uv23 = float4(loadPointUv(record.uv2, uvElement), loadPointUv(record.uv3, uvElement));
    context.color = isNull(record.colors) ? float4(1.0, 1.0, 1.0, 1.0)
                                          : unpackUnorm4x8(record.colors[(record.cornerStreams & meshCornerColors) != 0 ? corner : vertex]);
    context.uvGradients = float4(0.0, 0.0, 0.0, 0.0);
    context.dPdx = float4(0.0, 0.0, 0.0, 0.0);
    context.dPdy = float4(0.0, 0.0, 0.0, 0.0);
    context.objectPosition = float4(position, 0.0);
    //! Blender's automatic texture space gives a flat axis a half size of 1 about its middle.
    const float3 extent = mesh.boundsMax.xyz - mesh.boundsMin.xyz;
    const bool3 flat = extent == 0.0;
    const float3 generatedScale = 1.0 / select(flat, float3(2.0, 2.0, 2.0), extent);
    context.generated = float4((position - select(flat, mesh.boundsMin.xyz - 1.0, mesh.boundsMin.xyz)) * generatedScale, 0.0);
    context.generatedScale = float4(generatedScale, 0.0);
    context.objectLocation = float4(record.objectToWorld[0].w, record.objectToWorld[1].w, record.objectToWorld[2].w, 0.0);
    context.objectDx = float4(0.0, 0.0, 0.0, 0.0);
    context.objectDy = float4(0.0, 0.0, 0.0, 0.0);
    context.worldToObject0 = record.worldToObject[0];
    context.worldToObject1 = record.worldToObject[1];
    context.worldToObject2 = record.worldToObject[2];
    context.objectToWorld0 = record.objectToWorld[0];
    context.objectToWorld1 = record.objectToWorld[1];
    context.objectToWorld2 = record.objectToWorld[2];
    setCameraRay(context);
    return context;
}

//! MeshDisplacement compiles this kernel once per graph, with the code of the graph as the body of evaluateMaterialGraph; the emission is the offset.
[shader("compute")]
[numthreads(displacementGroupSize, 1, 1)]
void displaceGraph(uint3 group : SV_GroupID, uint thread : SV_GroupIndex) {
    const DisplacementGraphRecord record = rootArguments<DisplacementGraphArguments>().record[0];
    const uint index = (group.y * displacementRowGroups + group.x) * displacementGroupSize + thread;
    if (index >= record.pointCount)
        return;
    const DisplacementRecord mesh = record.mesh[0];
    const ShadingContext context = displacementContext(record, mesh, index);
    const float3 offset = evaluateMaterialGraph(context, graphMaterial(record.parameters, record.textures)).emission.xyz;
    const float3 position = context.objectPosition.xyz + (isFiniteValue(offset) ? offset : float3(0.0, 0.0, 0.0));
    const uint vertex = record.points[index].x;
    mesh.positions[3 * vertex] = position.x;
    mesh.positions[3 * vertex + 1] = position.y;
    mesh.positions[3 * vertex + 2] = position.z;
}
