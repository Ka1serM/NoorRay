import NoorRHI;
import SceneRecords;
import Materials.Graph;

//! The pipeline cache replaces this body with one alpha graph; its own name keeps it apart from the graph function of the kernels that it links into.
[noinline]
MaterialGraphOutput evaluateAlphaGraph(ShadingContext context, uint4 material) { return evaluateMaterialGraph(context, material); }

//! The pipeline cache copies this function, once for each alpha graph, into the kernels that call runAlphaGraph.
[noinline]
MaterialGraphCall alphaGraphCall(MaterialGraphCall call) {
    setGraphTrace(call.scene, call.instance, call.primitive, call.seed);
    call.output = evaluateAlphaGraph(call.context, call.material);
    return call;
}

//! Only the donor of alphaGraphCall: the pipeline cache never dispatches this kernel.
[shader("compute")]
[numthreads(64, 1, 1)]
void alphaGraph(uint3 id : SV_DispatchThreadID) {
    const Ptr<MaterialGraphCall> calls = rootArguments<Ptr<MaterialGraphCall>>();
    calls[id.x] = alphaGraphCall(calls[id.x]);
}
