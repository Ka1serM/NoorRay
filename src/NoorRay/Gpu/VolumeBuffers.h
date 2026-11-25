#pragma once

#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <unordered_map>

#include "NoorRHI/Resources/Buffer.h"
#include "NoorRay/Core/Error.h"
#include "NoorRay/Gpu/GraphBakes.h"
#include "NoorRay/Materials/GraphShader.h"
#include "NoorRay/Scene/Scene.h"
#include "NoorRay/Shaders/Common/GpuRecords.h"

namespace NoorRay::Gpu {

class VolumeBuffers {
public:
    //! A density graph volume gets a sparse grid baked once per graph, values and bounds; with exact density, the first graph runs at each point.
    [[nodiscard]] std::expected<void, Core::Error> update(NoorRHI::Device& device, const Scene::Scene& scene, GraphBakes& bakes, const TextureTable& textures);

    //! The medium kernels run the density graph at each point, which is exact but several times slower; takes effect at the next update.
    void setExactDensity(bool exact);
    bool hasExactDensity() const { return exactDensity; }
    NoorRHI::Ptr<Shaders::VolumeRecord> getRecords() const { return records.getPtr(); }
    std::uint32_t getCount() const { return count; }
    //! The density program that the medium kernels run, or null when no volume has a density graph.
    const std::shared_ptr<const Materials::GraphCode>& getGraphCode() const { return graphCode; }

private:
    struct Grid {
        VolumeGrid device;
        bool used = false;
    };

    [[nodiscard]] std::expected<void, Core::Error> updateGraphCode(std::span<const Scene::Volume> volumes);
    [[nodiscard]] std::expected<const VolumeGrid*, Core::Error> getGrid(NoorRHI::Device& device, const Scene::Volume& volume, GraphBakes& bakes,
                                                                        const TextureTable& textures);

    //! The device grids by the hash of their graph and grid or of their file densities; a grid no volume uses at an update goes.
    std::unordered_map<std::uint64_t, Grid> grids;
    NoorRHI::Buffer<Shaders::VolumeRecord> records;
    bool exactDensity = false;
    std::shared_ptr<const Materials::Graph> codeGraph;
    std::shared_ptr<const Materials::GraphCode> graphCode;
    std::uint32_t count = 0;
};

}
