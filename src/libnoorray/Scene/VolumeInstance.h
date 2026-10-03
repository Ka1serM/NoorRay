#pragma once

#include <memory>
#include <span>
#include <string>
#include <vector>

#include <glm/vec3.hpp>

#include <noorrhi/noorrhi.hpp>

#include "Mesh/Assets/Mesh.h"
#include "Scene/SceneObject.h"

// The edges of a volume in its local space, as pairs of end points. Instances
// of the same volume shape share one outline and its GPU buffer.
class VolumeOutline {
public:
    // Keeps the mesh's boundary edges and the edges where neighbouring faces
    // meet at more than featureAngleDegrees. Edges inside flat faces are
    // always dropped, so an angle of zero keeps every other edge.
    VolumeOutline(const MeshGeometry& geometry, float featureAngleDegrees);

    std::span<const glm::vec3> segments() const { return segments_; }
    uint32_t segmentCount() const { return static_cast<uint32_t>(segments_.size() / 2); }
    // Uploads on first use. Call outside an open frame.
    const noorrhi::Buffer<glm::vec3>& upload(noorrhi::Device& device) const;

private:
    std::vector<glm::vec3> segments_;
    mutable noorrhi::Buffer<glm::vec3> buffer_;
};

// A volume the viewport draws as an outline. It is never traced, so it casts
// no shadows and is not picked in the render.
class VolumeInstance : public SceneObject {
    friend class Scene;
    std::shared_ptr<const VolumeOutline> outline;
    glm::vec3 color;
    // Index into Scene's volume slots, ~0u while outside a scene.
    uint32_t slot{~0u};

public:
    VolumeInstance(Scene& scene, const std::string& name,
        std::shared_ptr<const VolumeOutline> outline, const Transform& transform,
        const glm::vec3& color);
    VolumeInstance(const VolumeInstance& other);

    std::unique_ptr<SceneObject> clone() const override;

    std::string getType() const { return "Volume"; }
    const std::shared_ptr<const VolumeOutline>& getOutline() const { return outline; }
    glm::vec3 getColor() const { return color; }

    void onTransformUpdated() override;
    void onVisibilityChanged() override;
};
