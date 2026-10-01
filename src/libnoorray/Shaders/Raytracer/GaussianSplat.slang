#pragma once

#include "ByteBuffer.slang"
#include "RaytracingAbi.slang"
#include "../Gaussian/Gaussian.slang"

// The splat a hit on a Gaussian proxy instance belongs to: the parameters of
// its cloud's splat material and its entry in them.
struct GaussianSplat
{
    ByteBuffer records;
    uint id;
};

GaussianSplat loadGaussianSplat(Frame frame, uint instanceId)
{
    const Instance instance = frame.scene.instances[instanceId];
    const Mesh mesh = *frame.scene.meshes[instance.meshIndex];
    const Material material = *frame.materials[instance.materials[mesh.sections[0].slot]];
    GaussianSplat splat;
    splat.records = byteBuffer(material.shaderParameters);
    splat.id = instance.materialEntry;
    return splat;
}
