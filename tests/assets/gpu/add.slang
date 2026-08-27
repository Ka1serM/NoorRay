// Root-argument compute kernel: every buffer is reached through a GPU pointer,
// and the only shader input is the root pointer the library pushes.
struct Arguments
{
    float* a;
    float* b;
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
        arguments.result[index] = arguments.a[index] + arguments.b[index];
}
