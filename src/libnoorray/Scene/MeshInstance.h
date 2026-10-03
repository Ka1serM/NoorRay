#pragma once

#include <memory>
#include <span>
#include <string>
#include <vector>

#include <glm/mat4x4.hpp>

#include "Mesh/Assets/Mesh.h"
#include "Scene/SceneObject.h"

class Scene;
class Material;

// A mesh drawn with one material per slot, placed once at this object or,
// as an instanced Unreal component is, once per placement relative to it.
// Every placement shares the object's materials and colors and picks as the
// object.
class MeshInstance : public SceneObject {
    friend class Scene;
    Mesh* mesh{};
    std::vector<Material*> materials;
    // Optional FColor per mesh vertex, replacing the mesh's colors, kept
    // alive by colorOwner.
    std::span<const uint32_t> colors;
    std::shared_ptr<const void> colorOwner;
    // Optional floats materials read per instance, kept alive by customDataOwner.
    std::span<const float> customData;
    std::shared_ptr<const void> customDataOwner;
    std::vector<glm::mat4> placements;
    // Index into Scene's mesh instance slots, ~0u while outside a scene.
    uint32_t slot{~0u};
    uint32_t materialEntry{};
public:
    // How rays treat the instance, as Unreal's primitive flags decide it:
    // which rays find it, and whether its triangles face the other way.
    struct RayTracingFlags {
        bool camera = true;
        bool shadow = true;
        bool indirect = true;
        // What a hidden instance still casts: shadows and indirect light, never camera rays.
        bool shadowWhileHidden = false;
        bool indirectWhileHidden = false;
        // Triangles wound clockwise face forward instead of counterclockwise.
        bool reverseCulling = false;
        // The mesh is animated independently of scene publication.
        bool animated = false;
        bool operator==(const RayTracingFlags&) const = default;
    };
private:
    RayTracingFlags rayTracingFlags;
    uint32_t lightingChannels{1u};

    void markChanged();
public:
    // materials holds one material per mesh slot. materialEntry is where this
    // instance's entry starts in its material's parameters, for materials
    // holding one entry per instance.
    MeshInstance(Scene& scene, const std::string& name, Mesh* mesh,
        std::vector<Material*> materials, const Transform& transf, uint32_t materialEntry = 0);
    MeshInstance(const MeshInstance& other);

    std::unique_ptr<SceneObject> clone() const override;

    uint32_t getMeshIndex() const { return mesh->getMeshIndex(); }
    Mesh& getMesh() { return *mesh; }
    const Mesh& getMesh() const { return *mesh; }
    Mesh* getMeshPtr() const { return mesh; }
    bool hasMesh() const { return mesh != nullptr; }
    uint32_t getMaterialEntry() const { return materialEntry; }

    const std::vector<Material*>& getMaterials() const { return materials; }
    Material* getMaterial(uint32_t materialSlot) const { return materials[materialSlot]; }
    void setMaterial(uint32_t materialSlot, Material* material);
    void setMaterials(std::vector<Material*> value);

    std::span<const uint32_t> getColors() const { return colors; }
    const std::shared_ptr<const void>& getColorOwner() const { return colorOwner; }
    void setColors(std::span<const uint32_t> value, std::shared_ptr<const void> owner);

    std::span<const float> getCustomData() const { return customData; }
    const std::shared_ptr<const void>& getCustomDataOwner() const { return customDataOwner; }
    void setCustomData(std::span<const float> value, std::shared_ptr<const void> owner);

    // Transforms relative to this object, one per placement; empty places
    // the mesh once, at the object.
    const std::vector<glm::mat4>& getPlacements() const { return placements; }
    void setPlacements(std::vector<glm::mat4> value);

    RayTracingFlags getRayTracingFlags() const { return rayTracingFlags; }
    void setRayTracingFlags(RayTracingFlags value);
    // Bit n is Unreal's lighting channel n.
    uint32_t getLightingChannels() const { return lightingChannels; }
    void setLightingChannels(uint32_t value);
    uint32_t getPlacementCount() const
    {
        return placements.empty() ? 1u : static_cast<uint32_t>(placements.size());
    }

    void onTransformUpdated() override;
    void onVisibilityChanged() override { markChanged(); }
};
