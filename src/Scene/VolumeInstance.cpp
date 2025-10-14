#pragma once
#include "SceneObject.h"
#include "Scene/Scene.h"
#include "VolumeAsset.h"  // you need a wrapper for your volume / DICOM data
#include <imgui.h>
#include <memory>

class VolumeInstance : public SceneObject {
public:
    std::shared_ptr<VolumeAsset> volumeAsset;
    vk::AccelerationStructureInstanceKHR instanceData;

    VolumeInstance(Scene& scene,
                   const std::string& name,
                   std::shared_ptr<VolumeAsset> asset,
                   const Transform& transf)
        : SceneObject(scene, name, transf),
          volumeAsset(std::move(asset))
    {
        setupInstance();
    }

    VolumeInstance(const VolumeInstance& other)
        : SceneObject(other),
          volumeAsset(other.volumeAsset)
    {
        if (volumeAsset) {
            setupInstance();
        }
    }

    std::unique_ptr<SceneObject> clone() const override {
        return std::make_unique<VolumeInstance>(*this);
    }

    void onTransformUpdated() override {
        SceneObject::onTransformUpdated();
        instanceData.setTransform(getWorldTransform().getVkTransformMatrix());
        scene.setDirtyFlag(TLAS); // mark TLAS dirty for rebuild
    }

    void renderUi() override {
        SceneObject::renderUi();
        ImGui::SeparatorText("Volume Asset");
        
        ImGui::TableNextColumn();
        if (volumeAsset)
            volumeAsset->renderUi();
        else
            ImGui::TextUnformatted("No Volume Assigned.");
    }

private:
    void setupInstance() {
        instanceData = vk::AccelerationStructureInstanceKHR{};
        instanceData.setTransform(getWorldTransform().getVkTransformMatrix());

        // Use volume index to fetch correct buffer in shader
        instanceData.setInstanceCustomIndex(volumeAsset->getVolumeIndex());

        instanceData.setMask(0xFF);
        instanceData.setInstanceShaderBindingTableRecordOffset(0);

        // Use AABB geometry (volumes are custom primitives)
        instanceData.setFlags(vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable);

        // Provide a "dummy" BLAS for RTX pipeline
        instanceData.setAccelerationStructureReference(volumeAsset->getBlasAddress());
    }
};