import NoorRHI;

//! The NRD resources of a dispatch are bindless handles in the root arguments, in the order of the NRD resource ranges.
static const uint nrdMaxInputs = 32;
static const uint nrdMaxOutputs = 16;

T nrdFromFloat4<T>(float4 value) {
    if (T is float)
        return reinterpret<T>(value.x);
    if (T is float2)
        return reinterpret<T>(value.xy);
    if (T is float3)
        return reinterpret<T>(value.xyz);
    return reinterpret<T>(value);
}

float4 nrdToFloat4<T>(T value) {
    if (T is float)
        return float4(reinterpret<float>(value), 0.0, 0.0, 0.0);
    if (T is float2)
        return float4(reinterpret<float2>(value), 0.0, 0.0);
    if (T is float3)
        return float4(reinterpret<float3>(value), 0.0);
    return reinterpret<float4>(value);
}

//! The layout of NoorRay::Render::NrdRoot: sampled handles of the inputs, storage handles of the outputs, the two samplers, then the NRD constants.
struct NrdRoot<C> {
    uint inputs[nrdMaxInputs];
    uint outputs[nrdMaxOutputs];
    uint samplers[2];
    uint2 padding;
    C constants;
}

struct NrdSampler {
    uint slot;
}

struct NrdTexture<T> {
    uint slot;
    TextureHandle handle() { return TextureHandle(rootArguments<NrdRoot<uint4> >().inputs[slot]); }
    __subscript(uint2 coord) -> T {
        get { return nrdFromFloat4<T>(handle().Load(int2(coord))); }
    }
    T SampleLevel(NrdSampler sampler, float2 uv, float lod) {
        const SamplerHandle samplerHandle = SamplerHandle(rootArguments<NrdRoot<uint4> >().samplers[sampler.slot]);
        return nrdFromFloat4<T>(handle().SampleLevel(samplerHandle, uv, lod));
    }
    //! NRD gathers only with the nearest sampler, so four loads give the same texels on both backends.
    float4 GatherRed(NrdSampler sampler, float2 uv) {
        const TextureHandle texture = handle();
        const int2 size = int2(texture.GetDimensions());
        const int2 base = int2(floor(uv * float2(size) - 0.5));
        const int2 low = clamp(base, int2(0), size - 1);
        const int2 high = clamp(base + 1, int2(0), size - 1);
        return float4(texture.Load(int2(low.x, high.y)).x, texture.Load(high).x, texture.Load(int2(high.x, low.y)).x, texture.Load(low).x);
    }
}

struct NrdStorage<T> {
    uint slot;
    StorageTextureHandle handle() { return StorageTextureHandle(rootArguments<NrdRoot<uint4> >().outputs[slot]); }
    __subscript(uint2 coord) -> T {
        get { return nrdFromFloat4<T>(handle()[coord]); }
        [nonmutating] set { handle()[coord] = nrdToFloat4<T>(newValue); }
    }
}
