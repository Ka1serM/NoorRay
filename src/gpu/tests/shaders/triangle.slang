// A fixed triangle covering the middle of the target, used to prove that the
// raster path actually writes colour to a render target.
struct VertexOutput
{
    float4 position : SV_Position;
    float3 color : COLOR0;
};

[shader("vertex")]
VertexOutput vertMain(uint vertexId : SV_VertexID)
{
    // Wound counter-clockwise in Vulkan's Y-down framebuffer space so the
    // triangle is front-facing under the default CullMode::Back.
    const float2 positions[3] = {
        float2(-0.8, -0.8), float2(0.0, 0.8), float2(0.8, -0.8)};
    const float3 colors[3] = {
        float3(1, 0, 0), float3(0, 0, 1), float3(0, 1, 0)};
    VertexOutput output;
    output.position = float4(positions[vertexId], 0, 1);
    output.color = colors[vertexId];
    return output;
}

[shader("fragment")]
float4 fragMain(VertexOutput input) : SV_Target0
{
    return float4(input.color, 1);
}
