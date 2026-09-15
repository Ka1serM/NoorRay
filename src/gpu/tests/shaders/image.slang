// Writes a gradient into a storage image reached through the resource
// descriptor heap, then reads the sampled view of a second image back through
// the heap and a sampler from the sampler heap.
struct Arguments
{
    uint target;
    uint source;
    uint sampler;
    uint size;
};

struct Root
{
    Arguments* arguments;
};

[[vk::push_constant]] ConstantBuffer<Root> root;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    const Arguments arguments = *root.arguments;
    if (id.x >= arguments.size || id.y >= arguments.size)
        return;
    RWTexture2D<float4> target = ResourceDescriptorHeap[arguments.target];
    Texture2D<float4> source = ResourceDescriptorHeap[arguments.source];
    SamplerState sampler = SamplerDescriptorHeap[arguments.sampler];
    // The source is a 1x1 texel, so any sample position returns it.
    const float blue = source.SampleLevel(sampler, float2(0.5, 0.5), 0.0).b;
    target[id.xy] = float4(float(id.x * 16u) / 255.0, float(id.y * 16u) / 255.0,
        blue, 1.0);
}
