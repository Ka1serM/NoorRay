import NoorRHI;
import Materials.Displacement;

[shader("compute")]
[numthreads(displacementGroupSize, 1, 1)]
void displaceFaceNormals(uint3 group : SV_GroupID, uint thread : SV_GroupIndex) {
    const DisplacementRecord record = rootArguments<DisplacementArguments>().record[0];
    const uint index = getDisplacementIndex(group, thread);
    if (index < record.faceCount)
        computeFaceNormal(record, index);
}
