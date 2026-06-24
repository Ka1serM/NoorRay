[shader("miss")]
void main(inout uint shadowPayload)
{
    shadowPayload = 0u;
}
