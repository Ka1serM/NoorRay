#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <unordered_map>
#include <vector>

#include "NoorRHI/Pipelines/Pipeline.h"
#include "NoorRay/Core/Error.h"
#include "NoorRay/Materials/GraphShader.h"

namespace NoorRay::Render {

//! Compiles each compute kernel once, the material kernel once per material graph, and each kernel that runs graphs once per volume graph and alpha graphs.
class PipelineCache {
public:
    //! The code of each material graph at its graph index and of each alpha graph at its alpha index; new material kernels compile in the background.
    [[nodiscard]] std::expected<void, Core::Error> setGraphCodes(NoorRHI::Device& device, std::span<const std::shared_ptr<const Materials::GraphCode>> codes,
                                                                 std::span<const std::shared_ptr<const Materials::GraphCode>> alphaCodes);
    //! The density graph that the medium kernels run from now on, or null when no volume has one.
    void setVolumeGraphCode(std::shared_ptr<const Materials::GraphCode> code);
    //! The material kernel of a material program: 0 for the materials without a graph, else 1 + the graph index; it waits for a graph still compiling.
    [[nodiscard]] std::expected<const NoorRHI::ComputePipeline*, Core::Error> getMaterial(NoorRHI::Device& device, std::uint32_t program);
    //! The pipeline of a compute shader, compiled on first use.
    [[nodiscard]] std::expected<const NoorRHI::ComputePipeline*, Core::Error> get(NoorRHI::Device& device, const NoorRHI::Shader& shader);
    //! Starts the compilation in the background on the first call; true once the kernel is compiled.
    [[nodiscard]] std::expected<bool, Core::Error> isReady(NoorRHI::Device& device, const NoorRHI::Shader& shader);

private:
    //! A kernel; a kernel that runs graphs has the code of the volume graph of volumeKey and of the alpha graphs of alphaKey, else zero.
    struct Compute {
        NoorRHI::ComputePipeline pipeline;
        std::chrono::steady_clock::time_point start;
        bool ready = false;
        std::array<std::vector<std::byte>, 4> code;
        NoorRHI::Shader shader;
        std::uint64_t volumeKey = 0;
        std::uint64_t alphaKey = 0;
    };

    //! The graphs that a kernel runs: the volume graph through the graph function, and the alpha graphs through the alpha graph switch.
    struct GraphCalls {
        bool volume = false;
        bool alpha = false;
    };

    //! The AlphaGraph kernel with the code of one alpha graph, which a link copies into the kernels that run alpha graphs.
    struct AlphaDonor {
        std::array<std::vector<std::byte>, 4> code;
        NoorRHI::Shader shader;
    };

    //! The material kernel with the code of one graph.
    struct Material {
        NoorRHI::ComputePipeline pipeline;
        std::array<std::vector<std::byte>, 4> code;
        NoorRHI::Shader shader;
        std::chrono::steady_clock::time_point start;
        bool ready = false;
    };

    //! The graph keys of a graph set and the key of the set, and the key of its alpha graphs.
    struct GraphSet {
        std::uint64_t key = 0;
        std::vector<std::uint64_t> graphs;
        std::uint64_t alphaKey = 0;
    };

    [[nodiscard]] std::expected<Compute*, Core::Error> openCompute(NoorRHI::Device& device, const NoorRHI::Shader& shader, bool async);
    [[nodiscard]] std::expected<GraphCalls, Core::Error> getGraphCalls(const NoorRHI::Shader& shader);
    [[nodiscard]] std::expected<Compute*, Core::Error> openGraphs(NoorRHI::Device& device, const NoorRHI::Shader& shader, GraphCalls calls, bool async);
    [[nodiscard]] std::expected<void, Core::Error> openAlphaDonors();
    [[nodiscard]] std::expected<Material*, Core::Error> openMaterial(NoorRHI::Device& device, const Materials::GraphCode& code);
    void dropUnusedSets();

    std::unordered_map<const NoorRHI::Shader*, std::unique_ptr<Compute>> computes;
    std::unordered_map<const NoorRHI::Shader*, GraphCalls> graphCallers;
    //! The kernels that run graphs, by their shader, volume graph and alpha graphs.
    std::unordered_map<std::uint64_t, std::unique_ptr<Compute>> graphKernels;
    std::shared_ptr<const Materials::GraphCode> volumeCode;
    Materials::GraphCode emptyVolumeCode;
    //! The volume graphs that have medium kernels, the current one last.
    std::vector<std::uint64_t> recentVolumeCodes;
    std::unordered_map<std::uint64_t, std::unique_ptr<Material>> materials;
    std::vector<std::shared_ptr<const Materials::GraphCode>> graphCodes;
    std::vector<std::shared_ptr<const Materials::GraphCode>> alphaGraphCodes;
    //! The donors of the alpha graphs by their code key.
    std::unordered_map<std::uint64_t, std::unique_ptr<AlphaDonor>> alphaDonors;
    //! The graph sets that have material kernels, the current one last.
    std::vector<GraphSet> recentSets;
};

}
