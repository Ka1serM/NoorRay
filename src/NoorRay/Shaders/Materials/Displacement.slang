//! Displaces a mesh on the GPU like Cycles: the program moves each vertex along its rest normal, then the face, vertex and corner normals follow.
import NoorRHI;
import Common.Math;
import SceneRecords;
import Materials.Program;
#include "NoorRay/Shaders/Materials/DisplacementRecords.h"

uint getDisplacementIndex(uint3 group, uint thread) { return (group.y * displacementRowGroups + group.x) * displacementGroupSize + thread; }

void storeFloat3(Ptr<float> values, uint index, float3 value) {
    values[3 * index] = value.x;
    values[3 * index + 1] = value.y;
    values[3 * index + 2] = value.z;
}

float3 normalizeOrZero(float3 value) {
    const float size = length(value);
    return size > 0.0 ? value / size : float3(0.0, 0.0, 0.0);
}

void displaceVertex(DisplacementRecord record, uint vertex) {
    ProgramVertex point;
    point.position = loadFloat3(record.restPositions, vertex);
    point.normal = loadFloat3(record.restNormals, vertex);
    const float3 extent = record.boundsMax.xyz - record.boundsMin.xyz;
    //! Blender's automatic texture space gives a flat axis a half size of 1 about its middle.
    const float3 span = select(extent > 0.0, extent, float3(1.0, 1.0, 1.0));
    point.generated = select(extent > 0.0, (point.position - record.boundsMin.xyz) / span, float3(0.5, 0.5, 0.5));
    float4 registers[maxProgramRegisters];
    for (uint index = 0; index < record.registerCount; ++index)
        registers[index] = record.registers[index];
    for (uint index = 0; index < record.instructionCount; ++index) {
        const uint2 instruction = record.instructions[index];
        float4 value;
        if (getProgramOp(instruction) == programOpAttribute)
            value = record.attributes[getProgramMode(instruction) * record.vertexCount + vertex];
        else
            value = evaluateProgramInstruction(instruction, registers[getProgramInput(instruction, 0)], registers[getProgramInput(instruction, 1)],
                                               registers[getProgramInput(instruction, 2)], registers[getProgramInput(instruction, 3)], point);
        registers[getProgramOutput(instruction)] = value;
    }
    const float3 offset = registers[record.displacement].xyz;
    storeFloat3(record.positions, vertex, point.position + (isFiniteValue(offset) ? offset : float3(0.0, 0.0, 0.0)));
}

float3 cornerPosition(DisplacementRecord record, uint corner) { return loadFloat3(record.positions, record.cornerVertices[corner]); }

//! Blender's face normal: the cross product of the diagonals of a quad, Newell's method beyond four corners, and +Z without area.
void computeFaceNormal(DisplacementRecord record, uint face) {
    const uint first = record.faceOffsets[face];
    const uint count = record.faceOffsets[face + 1] - first;
    float3 normal = float3(0.0, 0.0, 0.0);
    if (count == 3)
        normal = cross(cornerPosition(record, first) - cornerPosition(record, first + 1), cornerPosition(record, first + 1) - cornerPosition(record, first + 2));
    else if (count == 4)
        normal = cross(cornerPosition(record, first) - cornerPosition(record, first + 2), cornerPosition(record, first + 1) - cornerPosition(record, first + 3));
    else
        for (uint corner = 0; corner < count; ++corner) {
            const float3 current = cornerPosition(record, first + corner);
            const float3 next = cornerPosition(record, first + (corner + 1) % count);
            normal += float3((current.y - next.y) * (current.z + next.z), (current.z - next.z) * (current.x + next.x), (current.x - next.x) * (current.y + next.y));
        }
    normal = normalizeOrZero(normal);
    storeFloat3(record.faceNormals, face, dot(normal, normal) > 0.0 ? normal : float3(0.0, 0.0, 1.0));
}

//! The rest tangent made orthogonal to the displaced normal keeps the UV direction without a new MikkTSpace pass.
void storeTangent(DisplacementRecord record, uint index, float3 normal) {
    const float4 rest = unpackTangent(record.restTangents[index]);
    const float3 tangent = normalizeOrZero(rest.xyz - normal * dot(normal, rest.xyz));
    record.tangents[index] = packTangent(dot(tangent, tangent) > 0.0 ? tangent : rest.xyz, rest.w);
}

//! Blender's smooth normal: the face normals weighted by the corner angles.
void computeVertexNormal(DisplacementRecord record, uint vertex) {
    const float3 position = loadFloat3(record.positions, vertex);
    float3 normal = float3(0.0, 0.0, 0.0);
    for (uint index = record.cornerOffsets[vertex]; index < record.cornerOffsets[vertex + 1]; ++index) {
        const uint2 entry = record.corners[index];
        const uint first = record.faceOffsets[entry.x];
        const uint count = record.faceOffsets[entry.x + 1] - first;
        const uint corner = entry.y - first;
        const float3 toPrevious = normalizeOrZero(cornerPosition(record, first + (corner + count - 1) % count) - position);
        const float3 toNext = normalizeOrZero(cornerPosition(record, first + (corner + 1) % count) - position);
        normal += loadFloat3(record.faceNormals, entry.x) * acos(clamp(dot(toPrevious, toNext), -1.0, 1.0));
    }
    normal = normalizeOrZero(normal);
    if (dot(normal, normal) == 0.0)
        normal = normalizeOrZero(position);
    normal = dot(normal, normal) > 0.0 ? normal : float3(0.0, 0.0, 1.0);
    storeFloat3(record.vertexNormals, vertex, normal);
    if (record.normalStream == displacementStreamVertex)
        record.normals[vertex] = packOctahedral(normal);
    if (record.tangentStream == displacementStreamVertex)
        storeTangent(record, vertex, normal);
}

//! A corner takes the smooth normal of its vertex, since a stored split normal does not follow the displacement; a flat corner uses the triangle normal.
void scatterCorner(DisplacementRecord record, uint corner) {
    const uint vertex = record.indices[corner];
    float3 normal = loadFloat3(record.vertexNormals, vertex);
    if (record.normalStream == displacementStreamCorner)
        record.normals[corner] = packOctahedral(normal);
    if (record.tangentStream != displacementStreamCorner)
        return;
    if (record.normalStream == displacementStreamNone) {
        const uint first = corner - corner % 3;
        const float3 p0 = loadFloat3(record.positions, record.indices[first]);
        const float3 p1 = loadFloat3(record.positions, record.indices[first + 1]);
        const float3 p2 = loadFloat3(record.positions, record.indices[first + 2]);
        normal = normalizeOrZero(cross(p1 - p0, p2 - p0));
    }
    storeTangent(record, corner, normal);
}
