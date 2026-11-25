//! Compiles one NRD pass as is: the entry defines NRD_PASS_RESOURCES, NRD_PASS_SOURCE, NRD_CS_MAIN, the pass defines and, but Clear, NRD_PASS_CONFIG.
#include "Denoise/NrdBindings.slang"

#define NRD_INTERNAL
#pragma warning(disable : 15400)

//! The first include of the resources only builds the constants struct.
#define NRD_CONSTANTS_START(name) struct NrdConstants {
#define NRD_CONSTANT(type, name) column_major type name;
#define NRD_CONSTANTS_END };
#define NRD_INPUTS_START
#define NRD_INPUT(type, dataType, name, reg, binding, policy)
#define NRD_INPUTS_END
#define NRD_OUTPUTS_START
#define NRD_OUTPUT(type, dataType, name, reg, binding, policy)
#define NRD_OUTPUTS_END
#define NRD_SAMPLERS_START
#define NRD_SAMPLER(type, name, reg, binding)
#define NRD_SAMPLERS_END

#include "NRD.hlsli"
#include "ml.hlsli"
#ifdef NRD_PASS_CONFIG
#include NRD_PASS_CONFIG
#endif
#include NRD_PASS_RESOURCES

//! The pass source includes the resources again: now the constants read the root arguments and the resources are handles.
#undef NRD_CONSTANTS_START
#undef NRD_CONSTANT
#undef NRD_CONSTANTS_END
#undef NRD_INPUT
#undef NRD_OUTPUT
#undef NRD_SAMPLER
#define NRD_CONSTANTS_START(name)
#define NRD_CONSTANT(type, name) property type name { get { return rootArguments<NrdRoot<NrdConstants> >().constants.name; } }
#define NRD_CONSTANTS_END
#define NRD_INPUT(type, dataType, name, reg, binding, policy) static const NrdTexture<dataType> name = { binding }; NRD_DECLARE_RESOURCE(name, policy)
#define NRD_OUTPUT(type, dataType, name, reg, binding, policy) static const NrdStorage<dataType> name = { binding }; NRD_DECLARE_RESOURCE(name, policy)
#define NRD_SAMPLER(type, name, reg, binding) static const NrdSampler name = { binding };
#define Texture2D NrdTexture
#define RWTexture2D NrdStorage

#if defined(NOORRHI_WGSL)
//! Slang has no WGSL groupshared InterlockedAdd, so a barrier reduction replaces it. Every thread calls it and gets the group total. Only
//! RELAX_ClassifyTiles uses InterlockedAdd, in groups of 8x4 threads with an SV_GroupIndex called threadIndex.
#define NRD_GROUP_SUM_SIZE 32
groupshared uint nrdGroupSums[NRD_GROUP_SUM_SIZE];
uint nrdGroupSum(uint value, uint index)
{
    nrdGroupSums[index] = value;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = NRD_GROUP_SUM_SIZE / 2; stride > 0; stride >>= 1)
    {
        if (index < stride)
            nrdGroupSums[index] += nrdGroupSums[index + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    return nrdGroupSums[0];
}
#define InterlockedAdd(destination, value) destination = nrdGroupSum(value, threadIndex)
#endif

#include NRD_PASS_SOURCE
