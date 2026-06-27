#pragma once

#include <cuda.h>
#include <optix.h>

class OptiXSetup
{
public:
    OptiXSetup() = default;
    ~OptiXSetup();

    OptiXSetup(const OptiXSetup&) = delete;
    OptiXSetup& operator=(const OptiXSetup&) = delete;

    void create(CUcontext cudaContext);
    void destroy() noexcept;

    OptixDeviceContext getContext() const { return context; }
    OptixPipeline getPipeline() const { return pipeline; }
    const OptixShaderBindingTable& getExtendSbt() const { return extendSbt; }
    const OptixShaderBindingTable& getConnectSbt() const { return connectSbt; }

private:
    OptixDeviceContext context{};
    OptixModule module{};
    OptixProgramGroup extendGroup{};
    OptixProgramGroup connectGroup{};
    OptixProgramGroup triangleGroup{};
    OptixProgramGroup missGroup{};
    OptixPipeline pipeline{};
    CUdeviceptr extendRaygenRecord{};
    CUdeviceptr connectRaygenRecord{};
    CUdeviceptr hitgroupRecord{};
    CUdeviceptr missRecord{};
    OptixShaderBindingTable extendSbt{};
    OptixShaderBindingTable connectSbt{};
};
