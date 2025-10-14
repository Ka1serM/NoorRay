#pragma once

#include <memory>
#include <string>
#include <vector>
#include "Scene/Scene.h"
#include "../Shaders/SharedStructs.h"
#include "UI/ImGuiComponent.h"
#include "Vulkan/Accel.h"
#include "BVH/BVH.h"

class Scene;

class VolumeAsset : public ImGuiComponent {
public:
    static std::shared_ptr<VolumeAsset> CreateVolume(Scene& scene, const std::string& name, const vec3& dimensions);

    VolumeAsset(Scene& scene, std::string name, const vec3& dimensions, const std::vector<uint8_t>& voxelData);

    void renderUi() override;

    uint64_t getBlasAddress() const;
    uint64_t getTextureAddress() const;

    uint32_t getVolumeIndex() const;
    void setVolumeIndex(uint32_t newIndex);

    const std::vector<uint8_t>& getVoxelData() const { return voxels; }
    const vec3& getDimensions() const { return dims; }

    bool isDirty() const { return dirty; }
    void clearDirtyFlag() { dirty = false; }

private:
    void uploadToGpu();
    void buildBlas();

private:
    Scene& scene;
    std::string path;
    vec3 dims;
    uint32_t index = -1;

    std::vector<uint8_t> voxels;

    // GPU resources
    Buffer voxelBuffer;

    // BLAS for RTX
    Accel blasRtx;

    // CPU BLAS for non-RTX
    BVH blasCompute;
};