#include "VolumeAsset.h"
#include <imgui.h>
#include <utility>

std::shared_ptr<VolumeAsset> VolumeAsset::CreateVolume(Scene& scene, const std::string& name, const vec3& dimensions) {
    std::vector<uint8_t> voxels(dimensions.x * dimensions.y * dimensions.z, 0u);
    return std::make_shared<VolumeAsset>(scene, name, dimensions, voxels);
}

VolumeAsset::VolumeAsset(Scene& scene, std::string name, const vec3& dimensions, const std::vector<uint8_t>& voxelData)
    : scene(scene), path(std::move(name)), dims(dimensions), voxels(voxelData)
{
    uploadToGpu();
    buildBlas();
}

void VolumeAsset::uploadToGpu() {
    voxelBuffer = Buffer{
        scene.getContext(),
        Buffer::Type::Storage,
        sizeof(uint8_t) * voxels.size(),
        voxels.data()
    };
    dirty = false;
}

void VolumeAsset::buildBlas() {
    if (scene.getContext().isRtxSupported()) {
        vk::AccelerationStructureGeometryKHR geometry{};
        vk::AccelerationStructureGeometryAabbsDataKHR aabbData{};
        aabbData.setData(voxelBuffer.getDeviceAddress());
        aabbData.setStride(sizeof(vec3)); // One AABB per volume
        geometry.setGeometryType(vk::GeometryTypeKHR::eAabbs);
        geometry.setGeometry({aabbData});
        geometry.setFlags(vk::GeometryFlagBitsKHR::eOpaque);

        blasRtx.build(scene.getContext(), geometry, 1, vk::AccelerationStructureTypeKHR::eBottomLevel);
    } else {
        AABB aabbMinMax = AABB{};
        aabbMinMax.minBounds = vec3(0.0f);
        aabbMinMax.maxBounds = dims;
        blasCompute.build(scene.getContext(), aabbMinMax);
    }
}

uint64_t VolumeAsset::getBlasAddress() const {
    if (scene.getContext().isRtxSupported()) {
        return blasRtx.getBuffer().getDeviceAddress();
    } else {
        return 0; // CPU BLAS is used in software compute
    }
}

uint64_t VolumeAsset::getTextureAddress() const {
    return voxelBuffer.getDeviceAddress();
}

uint32_t VolumeAsset::getVolumeIndex() const {
    return index;
}

void VolumeAsset::setVolumeIndex(const uint32_t newIndex) {
    index = newIndex;
}

void VolumeAsset::renderUi() {
    ImGui::Text("Volume: %s", path.c_str());
    ImGui::Text("Dimensions: %.1f x %.1f x %.1f", dims.x, dims.y, dims.z);

    if (scene.getContext().isRtxSupported()) {
        ImGui::Text("BLAS: RTX");
    } else {
        ImGui::Text("BLAS: CPU compute");
    }
}