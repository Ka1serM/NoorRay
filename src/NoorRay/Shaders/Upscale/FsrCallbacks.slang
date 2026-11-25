//! The NoorRHI callbacks of the FSR 3.1 upscaler: every resource is a bindless handle in the root arguments, indexed by its FSR resource identifier.
#include "fsr3upscaler/ffx_fsr3upscaler_resources.h"
#include "ffx_core.h"

#define FFX_PREFER_WAVE64

//! The layout of NoorRay::Render::FsrRoot. constants holds cbFSR3Upscaler; secondary holds cbSPD or cbRCAS.
struct FsrConstants {
    int2 renderSize;
    int2 previousRenderSize;
    int2 upscaleSize;
    int2 previousUpscaleSize;
    int2 maxRenderSize;
    int2 maxUpscaleSize;
    float4 deviceToViewDepth;
    float2 jitter;
    float2 previousJitter;
    float2 motionVectorScale;
    float2 downscaleFactor;
    float2 motionVectorJitterCancellation;
    float tanHalfFov;
    float jitterSequenceLength;
    float deltaTime;
    float deltaPreExposure;
    float viewSpaceToMetersFactor;
    float frameIndex;
    float velocityFactor;
    float reactivenessScale;
    float shadingChangeScale;
    float accumulationAddedPerFrame;
    float minDisocclusionAccumulation;
    float padding0;
    float padding1;
    float padding2;
};

struct FsrRoot {
    uint srv[FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_COUNT];
    uint uav[FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_COUNT];
    uint spdMips[6];
    uint pointClamp;
    uint linearClamp;
    AtomicPtr<uint> reconstructedDepth;
    AtomicPtr<uint> spdCounter;
    AtomicPtr<float> spdMip5;
    uint reconstructedDepthPitch;
    uint spdMip5Pitch;
    FsrConstants constants;
    uint4 secondary0;
    uint4 secondary1;
};

FsrRoot fsrRoot() { return rootArguments<FsrRoot>(); }
FsrConstants fsrConstants() { return rootArguments<FsrRoot>().constants; }
TextureHandle fsrSrv(uint resource) { return TextureHandle(rootArguments<FsrRoot>().srv[resource]); }
StorageTextureHandle fsrUav(uint resource) { return StorageTextureHandle(rootArguments<FsrRoot>().uav[resource]); }
SamplerHandle fsrLinearClamp() { return SamplerHandle(rootArguments<FsrRoot>().linearClamp); }
float4 fsrLoad(uint resource, int2 position) { return fsrSrv(resource).Load(position); }
float4 fsrSample(uint resource, float2 uv) { return fsrSrv(resource).SampleLevel(fsrLinearClamp(), uv, 0.0); }
int2 fsrSrvSize(uint resource) { return int2(fsrSrv(resource).GetDimensions()); }

FfxInt32x2 RenderSize() { return fsrConstants().renderSize; }
FfxInt32x2 PreviousFrameRenderSize() { return fsrConstants().previousRenderSize; }
FfxInt32x2 MaxRenderSize() { return fsrConstants().maxRenderSize; }
FfxInt32x2 UpscaleSize() { return fsrConstants().upscaleSize; }
FfxInt32x2 PreviousFrameUpscaleSize() { return fsrConstants().previousUpscaleSize; }
FfxInt32x2 MaxUpscaleSize() { return fsrConstants().maxUpscaleSize; }
FfxFloat32x2 Jitter() { return fsrConstants().jitter; }
FfxFloat32x2 PreviousFrameJitter() { return fsrConstants().previousJitter; }
FfxFloat32x4 DeviceToViewSpaceTransformFactors() { return fsrConstants().deviceToViewDepth; }
FfxFloat32x2 MotionVectorScale() { return fsrConstants().motionVectorScale; }
FfxFloat32x2 DownscaleFactor() { return fsrConstants().downscaleFactor; }
FfxFloat32x2 MotionVectorJitterCancellation() { return fsrConstants().motionVectorJitterCancellation; }
FfxFloat32 TanHalfFoV() { return fsrConstants().tanHalfFov; }
FfxFloat32 JitterSequenceLength() { return fsrConstants().jitterSequenceLength; }
FfxFloat32 DeltaTime() { return fsrConstants().deltaTime; }
FfxFloat32 DeltaPreExposure() { return fsrConstants().deltaPreExposure; }
FfxFloat32 ViewSpaceToMetersFactor() { return fsrConstants().viewSpaceToMetersFactor; }
FfxFloat32 FrameIndex() { return fsrConstants().frameIndex; }
FfxFloat32 VelocityFactor() { return fsrConstants().velocityFactor; }
FfxFloat32 AccumulationAddedPerFrame() { return fsrConstants().accumulationAddedPerFrame; }
FfxFloat32 MinDisocclusionAccumulation() { return fsrConstants().minDisocclusionAccumulation; }

FfxUInt32x4 RCASConfig() { return rootArguments<FsrRoot>().secondary0; }
FfxUInt32 MipCount() { return rootArguments<FsrRoot>().secondary0.x; }
FfxUInt32 NumWorkGroups() { return rootArguments<FsrRoot>().secondary0.y; }
FfxUInt32x2 WorkGroupOffset() { return rootArguments<FsrRoot>().secondary0.zw; }
FfxUInt32x2 SPD_RenderSize() { return rootArguments<FsrRoot>().secondary1.xy; }

//! SPD mips is one image per level, so a mip level selects a handle.
FfxInt32x2 GetSPDMipDimensions(FfxUInt32 uMipLevel) { return int2(TextureHandle(rootArguments<FsrRoot>().spdMips[min(uMipLevel, 5u)]).GetDimensions()); }
FfxFloat32x2 SampleSPDMipLevel(FfxFloat32x2 fUV, FfxUInt32 mipLevel) {
    return TextureHandle(rootArguments<FsrRoot>().spdMips[min(mipLevel, 5u)]).SampleLevel(fsrLinearClamp(), fUV, 0.0).xy;
}

FfxFloat32 LoadInputDepth(FfxUInt32x2 iPxPos) { return fsrLoad(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INPUT_DEPTH, int2(iPxPos)).x; }
FfxFloat32 SampleInputDepth(FfxFloat32x2 fUV) { return fsrSample(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INPUT_DEPTH, fUV).x; }

FfxFloat32 LoadReactiveMask(FfxUInt32x2 iPxPos) {
    return fsrLoad(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INPUT_REACTIVE_MASK, int2(iPxPos)).x * fsrConstants().reactivenessScale;
}
FfxInt32x2 GetReactiveMaskResourceDimensions() { return fsrSrvSize(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INPUT_REACTIVE_MASK); }
FfxFloat32 SampleReactiveMask(FfxFloat32x2 fUV) {
    return fsrSample(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INPUT_REACTIVE_MASK, fUV).x * fsrConstants().reactivenessScale;
}

FfxFloat32 LoadTransparencyAndCompositionMask(FfxUInt32x2 iPxPos) {
    return fsrLoad(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INPUT_TRANSPARENCY_AND_COMPOSITION_MASK, int2(iPxPos)).x;
}
FfxInt32x2 GetTransparencyAndCompositionMaskResourceDimensions() {
    return fsrSrvSize(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INPUT_TRANSPARENCY_AND_COMPOSITION_MASK);
}
FfxFloat32 SampleTransparencyAndCompositionMask(FfxFloat32x2 fUV) {
    return fsrSample(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INPUT_TRANSPARENCY_AND_COMPOSITION_MASK, fUV).x;
}

FfxFloat32x3 LoadInputColor(FfxUInt32x2 iPxPos) { return fsrLoad(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INPUT_COLOR, int2(iPxPos)).rgb; }
FfxFloat32x3 SampleInputColor(FfxFloat32x2 fUV) { return fsrSample(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INPUT_COLOR, fUV).rgb; }

FfxFloat32x2 LoadInputMotionVector(FfxUInt32x2 iPxDilatedMotionVectorPos) {
    FfxFloat32x2 fUvMotionVector = fsrLoad(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INPUT_MOTION_VECTORS, int2(iPxDilatedMotionVectorPos)).xy * MotionVectorScale();
#if FFX_FSR3UPSCALER_OPTION_JITTERED_MOTION_VECTORS
    fUvMotionVector -= MotionVectorJitterCancellation();
#endif
    return fUvMotionVector;
}

FfxFloat32x4 LoadHistory(FfxUInt32x2 iPxHistory) { return fsrLoad(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INTERNAL_UPSCALED_COLOR, int2(iPxHistory)); }
FfxFloat32x4 SampleHistory(FfxFloat32x2 fUV) { return fsrSample(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INTERNAL_UPSCALED_COLOR, fUV); }

void StoreLumaHistory(FfxUInt32x2 iPxPos, FfxFloat32x4 fLumaHistory) { fsrUav(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_LUMA_HISTORY)[iPxPos] = fLumaHistory; }
FfxFloat32x4 LoadLumaHistory(FfxInt32x2 iPxPos) { return fsrLoad(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_LUMA_HISTORY, iPxPos); }
FfxFloat32x4 SampleLumaHistory(FfxFloat32x2 fUV) { return fsrSample(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_LUMA_HISTORY, fUV); }

FfxFloat32x4 LoadRCAS_Input(FfxInt32x2 iPxPos) { return fsrLoad(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_RCAS_INPUT, iPxPos); }
FfxFloat32x3 SampleRCAS_Input(FfxFloat32x2 fUV) { return fsrSample(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_RCAS_INPUT, fUV).rgb; }

void StoreReprojectedHistory(FfxUInt32x2 iPxHistory, FfxFloat32x4 fHistory) {
    fsrUav(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INTERNAL_UPSCALED_COLOR)[iPxHistory] = fHistory;
}
void StoreInternalColorAndWeight(FfxUInt32x2 iPxPos, FfxFloat32x4 fColorAndWeight) {
    fsrUav(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INTERNAL_UPSCALED_COLOR)[iPxPos] = fColorAndWeight;
}
void StoreUpscaledOutput(FfxUInt32x2 iPxPos, FfxFloat32x3 fColor) { fsrUav(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_UPSCALED_OUTPUT)[iPxPos] = float4(fColor, 1.0); }

FfxFloat32 SampleAccumulation(FfxFloat32x2 fUV) { return fsrSample(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_ACCUMULATION, fUV).x; }
void StoreAccumulation(FfxUInt32x2 iPxPos, FfxFloat32 fAccumulation) { fsrUav(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_ACCUMULATION)[iPxPos] = float4(fAccumulation); }

FfxFloat32 LoadShadingChange(FfxUInt32x2 iPxPos) {
    return fsrLoad(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_SHADING_CHANGE, int2(iPxPos)).x * fsrConstants().shadingChangeScale;
}
FfxFloat32 SampleShadingChange(FfxFloat32x2 fUV) { return fsrSample(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_SHADING_CHANGE, fUV).x * fsrConstants().shadingChangeScale; }
void StoreShadingChange(FfxUInt32x2 iPxPos, FfxFloat32 fShadingChange) { fsrUav(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_SHADING_CHANGE)[iPxPos] = float4(fShadingChange); }

FfxInt32x2 GetFarthestDepthResourceDimensions() { return fsrSrvSize(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_FARTHEST_DEPTH); }
FfxFloat32 LoadFarthestDepth(FfxUInt32x2 iPxPos) { return fsrLoad(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_FARTHEST_DEPTH, int2(iPxPos)).x; }
FfxFloat32 SampleFarthestDepth(FfxFloat32x2 fUV) { return fsrSample(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_FARTHEST_DEPTH, fUV).x; }
void StoreFarthestDepth(FfxUInt32x2 iPxPos, FfxFloat32 fDepth) { fsrUav(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_FARTHEST_DEPTH)[iPxPos] = float4(fDepth); }

FfxInt32x2 GetFarthestDepthMip1ResourceDimensions() { return fsrSrvSize(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_FARTHEST_DEPTH_MIP1); }
FfxFloat32 LoadFarthestDepthMip1(FfxUInt32x2 iPxPos) { return fsrLoad(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_FARTHEST_DEPTH_MIP1, int2(iPxPos)).x; }
FfxFloat32 SampleFarthestDepthMip1(FfxFloat32x2 fUV) { return fsrSample(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_FARTHEST_DEPTH_MIP1, fUV).x; }
void StoreFarthestDepthMip1(FfxUInt32x2 iPxPos, FfxFloat32 fDepth) { fsrUav(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_FARTHEST_DEPTH_MIP1)[iPxPos] = float4(fDepth); }

FfxFloat32 LoadCurrentLuma(FfxUInt32x2 iPxPos) { return fsrLoad(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_CURRENT_LUMA, int2(iPxPos)).x; }
FfxFloat32 SampleCurrentLuma(FfxFloat32x2 uv) { return fsrSample(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_CURRENT_LUMA, uv).x; }
void StoreCurrentLuma(FfxUInt32x2 iPxPos, FfxFloat32 fLuma) { fsrUav(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_CURRENT_LUMA)[iPxPos] = float4(fLuma); }

FfxFloat32 SampleLumaInstability(FfxFloat32x2 uv) { return fsrSample(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_LUMA_INSTABILITY, uv).x; }
void StoreLumaInstability(FfxUInt32x2 iPxPos, FfxFloat32 fLumaInstability) {
    fsrUav(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_LUMA_INSTABILITY)[iPxPos] = float4(fLumaInstability);
}

FfxFloat32 LoadPreviousLuma(FfxUInt32x2 iPxPos) { return fsrLoad(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_PREVIOUS_LUMA, int2(iPxPos)).x; }
FfxFloat32 SamplePreviousLuma(FfxFloat32x2 uv) { return fsrSample(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_PREVIOUS_LUMA, uv).x; }

FfxFloat32 LoadNewLocks(FfxUInt32x2 iPxPos) { return fsrLoad(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_NEW_LOCKS, int2(iPxPos)).x; }
FfxFloat32 LoadRwNewLocks(FfxUInt32x2 iPxPos) { return fsrUav(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_NEW_LOCKS)[iPxPos].x; }
void StoreNewLocks(FfxUInt32x2 iPxPos, FfxFloat32 newLock) { fsrUav(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_NEW_LOCKS)[iPxPos] = float4(newLock); }

//! The reconstructed depth needs 32-bit atomics, so it lives in a buffer instead of an R32_UINT image.
AtomicPtr<uint> fsrReconstructedDepth(FfxUInt32x2 iPxPos) {
    const FsrRoot root = fsrRoot();
    return root.reconstructedDepth + (iPxPos.y * root.reconstructedDepthPitch + iPxPos.x);
}
FfxFloat32 LoadReconstructedPrevDepth(FfxUInt32x2 iPxPos) { return asfloat(atomicLoad(fsrReconstructedDepth(iPxPos))); }
void StoreReconstructedDepth(FfxUInt32x2 iPxSample, FfxFloat32 fDepth) {
#if FFX_FSR3UPSCALER_OPTION_INVERTED_DEPTH
    atomicMax(fsrReconstructedDepth(iPxSample), asuint(fDepth));
#else
    atomicMin(fsrReconstructedDepth(iPxSample), asuint(fDepth));
#endif
}
void SetReconstructedDepth(FfxUInt32x2 iPxSample, const FfxUInt32 uValue) { atomicStore(fsrReconstructedDepth(iPxSample), uValue); }

void StoreDilatedDepth(FfxUInt32x2 iPxPos, FfxFloat32 fDepth) { fsrUav(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_DILATED_DEPTH)[iPxPos] = float4(fDepth); }
void StoreDilatedMotionVector(FfxUInt32x2 iPxPos, FfxFloat32x2 fMotionVector) {
    fsrUav(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_DILATED_MOTION_VECTORS)[iPxPos] = float4(fMotionVector, 0.0, 0.0);
}
FfxFloat32x2 LoadDilatedMotionVector(FfxUInt32x2 iPxInput) { return fsrLoad(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_DILATED_MOTION_VECTORS, int2(iPxInput)).xy; }
FfxFloat32x2 SampleDilatedMotionVector(FfxFloat32x2 fUV) { return fsrSample(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_DILATED_MOTION_VECTORS, fUV).xy; }
FfxFloat32 LoadDilatedDepth(FfxUInt32x2 iPxInput) { return fsrLoad(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_DILATED_DEPTH, int2(iPxInput)).x; }
FfxFloat32 SampleDilatedDepth(FfxFloat32x2 fUV) { return fsrSample(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_DILATED_DEPTH, fUV).x; }

FfxFloat32 Exposure() {
    const FfxFloat32 exposure = fsrLoad(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INPUT_EXPOSURE, int2(0, 0)).x;
    return exposure == 0.0 ? 1.0 : exposure;
}

//! NoorRay compiles the reference Lanczos function, so the lookup table is never read.
FfxFloat32 SampleLanczos2Weight(FfxFloat32 x) { return 0.0; }

FfxFloat32x4 SampleDilatedReactiveMasks(FfxFloat32x2 fUV) { return fsrSample(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_DILATED_REACTIVE_MASKS, fUV); }
void StoreDilatedReactiveMasks(FfxUInt32x2 iPxPos, FfxFloat32x4 fDilatedReactiveMasks) {
    fsrUav(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_DILATED_REACTIVE_MASKS)[iPxPos] = fDilatedReactiveMasks;
}

FfxFloat32x4 LoadFrameInfo() { return fsrUav(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_FRAME_INFO)[uint2(0, 0)]; }
void StoreFrameInfo(FfxFloat32x4 fInfo) { fsrUav(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_FRAME_INFO)[uint2(0, 0)] = fInfo; }
FfxFloat32x4 FrameInfo() { return fsrLoad(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_FRAME_INFO, int2(0, 0)); }

//! SPD reads mip 5 back in the last workgroup of the same dispatch, so mip 5 also goes to a buffer with coherent atomics.
AtomicPtr<float> fsrSpdMip5(FfxInt32x2 iPxPos) {
    const FsrRoot root = fsrRoot();
    return root.spdMip5 + (uint(iPxPos.y) * root.spdMip5Pitch + uint(iPxPos.x)) * 2u;
}
FfxFloat32x2 RWLoadPyramid(FfxInt32x2 iPxPos, FfxUInt32 index) {
    if (index == 5u) {
        const AtomicPtr<float> texel = fsrSpdMip5(iPxPos);
        return float2(atomicLoad(texel), atomicLoad(texel + 1));
    }
    return fsrUav(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_SPD_MIPS_LEVEL_0 + index)[uint2(iPxPos)].xy;
}
void StorePyramid(FfxInt32x2 iPxPos, FfxFloat32x2 outValue, FfxUInt32 index) {
    fsrUav(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_SPD_MIPS_LEVEL_0 + index)[uint2(iPxPos)] = float4(outValue, 0.0, 0.0);
    if (index == 5u) {
        const AtomicPtr<float> texel = fsrSpdMip5(iPxPos);
        atomicStore(texel, outValue.x);
        atomicStore(texel + 1, outValue.y);
    }
}

void SPD_IncreaseAtomicCounter(inout FfxUInt32 spdCounter) { spdCounter = atomicAdd(rootArguments<FsrRoot>().spdCounter, 1u); }
void SPD_ResetAtomicCounter() { atomicStore(rootArguments<FsrRoot>().spdCounter, 0u); }
