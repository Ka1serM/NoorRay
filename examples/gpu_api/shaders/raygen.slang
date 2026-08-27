struct Arguments
{
    uint* result;
};

struct Root
{
    Arguments* arguments;
};

[[vk::push_constant]] ConstantBuffer<Root> root;

[shader("raygeneration")]
void main()
{
    const Arguments arguments = *root.arguments;
    if (DispatchRaysIndex().x == 3 && DispatchRaysIndex().y == 2
        && DispatchRaysIndex().z == 0)
        arguments.result[0] = 23;
}
