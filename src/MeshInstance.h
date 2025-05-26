#pragma once

#include <memory>

#include "MeshAsset.h"
#include "SceneObject.h"

class MeshInstance : public SceneObject {
public:
    const std::shared_ptr<MeshAsset> meshAsset;
    vk::AccelerationStructureInstanceKHR instanceData;

    MeshInstance(const std::string &name, std::shared_ptr<MeshAsset> asset, const Transform &transform);

    void renderUiOverride() override;
};