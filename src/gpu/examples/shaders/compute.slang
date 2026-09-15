struct Arguments
{
    float* lhs;
    float* rhs;
    float* result;
    uint count;
};

struct Root
{
    Arguments* arguments;
};

[[vk::push_constant]] ConstantBuffer<Root> root;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const Arguments arguments = *root.arguments;
    const uint index = dispatchThreadId.x;
    if (index < arguments.count)
        arguments.result[index] = arguments.lhs[index] + arguments.rhs[index];
}
