#pragma once

#include <chrono>
#include <expected>
#include <memory>
#include <unordered_map>

#include "NoorRHI/Pipelines/Pipeline.h"
#include "NoorRay/Core/Error.h"

namespace NoorRay::Render {

//! One compute pipeline per shader variant, opened on first use and kept until the cache closes.
class PipelineCache {
public:
    [[nodiscard]] std::expected<const NoorRHI::ComputePipeline*, Core::Error> get(NoorRHI::Device& device, const NoorRHI::Shader& shader);
    //! Starts the compilation in the background on the first call; true once the pipeline is compiled.
    [[nodiscard]] std::expected<bool, Core::Error> isReady(NoorRHI::Device& device, const NoorRHI::Shader& shader);

private:
    struct Entry {
        NoorRHI::ComputePipeline pipeline;
        std::chrono::steady_clock::time_point start;
        bool ready = false;
    };

    [[nodiscard]] std::expected<Entry*, Core::Error> open(NoorRHI::Device& device, const NoorRHI::Shader& shader, bool async);
    static void markReady(Entry& entry, const NoorRHI::Shader& shader);

    std::unordered_map<const NoorRHI::Shader*, std::unique_ptr<Entry>> pipelines;
};

}
