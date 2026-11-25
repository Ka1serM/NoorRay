#pragma once

#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "NoorRHI/Resources/Buffer.h"
#include "NoorRay/Core/Error.h"
#include "NoorRay/Materials/GraphShader.h"
#include "NoorRay/Scene/Scene.h"
#include "NoorRay/Shaders/Common/GpuRecords.h"

namespace NoorRay::Gpu {

class TextureTable;

//! emission is the emitted radiance factor (colour times strength) for light sampling weights; an emission texture counts as one.
struct MaterialInfo {
    std::uint32_t flags = 0;
    Core::float3 emission;
    //! The graph program of a material whose emission varies, whose kernel bakes the emission of its triangle lights; invalidIndex otherwise.
    std::uint32_t emitterProgram = Shaders::invalidIndex;
    //! Whether light sampling picks the triangles; an emission that varies with the view or distance counts only where rays hit, as Emission Sampling None.
    bool sampled = true;
    //! The faces that light sampling skips, as lightFlagNoFront and lightFlagNoBack for an instance that keeps the winding of the mesh.
    std::uint32_t skippedFaces = 0;
};

Shaders::MaterialTextureRecord makeTextureRecord(const Scene::MaterialTexture& texture, const TextureTable& textures);

//! The GPU records of the scene materials in dense order, then a default material for empty slots at getCount(), and the texture records of their set slots.
class MaterialTable {
public:
    [[nodiscard]] std::expected<void, Core::Error> update(NoorRHI::Device& device, const Scene::Scene& scene, const TextureTable& textures);

    NoorRHI::Ptr<Shaders::MaterialRecord> getRecords() const { return records.getPtr(); }
    std::uint32_t getCount() const { return static_cast<std::uint32_t>(infos.size()); }
    std::span<const MaterialInfo> getInfos() const { return infos; }
    std::uint32_t getFlags() const { return flagUnion; }
    NoorRHI::Ptr<NoorRHI::float4> getGraphParameters() const { return graphParameters.getPtr(); }
    //! The code of each distinct graph of the materials, at the graph index of the material records.
    std::span<const std::shared_ptr<const Materials::GraphCode>> getGraphCodes() const { return graphCodes.codes; }
    //! The code of each distinct graph of the alpha-tested materials as the alpha graph function, at the alpha index of the material records.
    std::span<const std::shared_ptr<const Materials::GraphCode>> getAlphaGraphCodes() const { return alphaGraphCodes.codes; }

private:
    //! The code of graphs; the graphs stay, so that the code of a graph is written again only when its program changes.
    struct GraphCodes {
        std::vector<std::shared_ptr<const Materials::Graph>> graphs;
        std::vector<std::shared_ptr<const Materials::GraphCode>> codes;
    };

    [[nodiscard]] static std::expected<void, Core::Error> updateGraphCode(std::vector<std::shared_ptr<const Materials::Graph>> graphs, std::string_view name,
                                                                          GraphCodes& codes);

    std::vector<MaterialInfo> infos;
    std::uint32_t flagUnion = 0;
    NoorRHI::Buffer<Shaders::MaterialRecord> records;
    NoorRHI::Buffer<Shaders::MaterialTextureRecord> textureRecords;
    NoorRHI::Buffer<NoorRHI::float4> graphParameters;
    GraphCodes graphCodes;
    GraphCodes alphaGraphCodes;
};

}
