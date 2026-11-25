#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "NoorRHI/Commands/CommandList.h"
#include "NoorRHI/Commands/TimestampQuery.h"
#include "NoorRay/Core/Error.h"

namespace NoorRay::Render {

//! The GPU time of named scopes over many frames, one timestamp pair per scope; for measurements only, as each pair waits for the work before it.
class GpuProfiler {
public:
    struct Total {
        std::string name;
        double milliseconds = 0.0;
        std::uint32_t calls = 0;
    };

    [[nodiscard]] std::expected<void, Core::Error> open(NoorRHI::Device& device);
    void begin(NoorRHI::CommandList& list, std::string_view name);
    void end(NoorRHI::CommandList& list);
    //! Adds the scopes of the last frame to the totals; the device must have finished the frame.
    [[nodiscard]] std::expected<void, Core::Error> collect();
    //! The totals by name, the largest first.
    std::vector<Total> getTotals() const;
    std::uint32_t getFrameCount() const { return frames; }
    void clear();

private:
    NoorRHI::Device* device = nullptr;
    std::vector<NoorRHI::TimestampQuery> queries;
    std::vector<std::string> names;
    std::vector<std::size_t> openScopes;
    std::unordered_map<std::string, Total> totals;
    std::uint32_t frames = 0;
    std::expected<void, Core::Error> result;
};

//! Times the scope of its lifetime when there is a profiler.
class GpuScope {
public:
    GpuScope(GpuProfiler* profiler, NoorRHI::CommandList& list, const std::string_view name) : profiler(profiler), list(list) {
        if (profiler)
            profiler->begin(list, name);
    }
    ~GpuScope() {
        if (profiler)
            profiler->end(list);
    }
    GpuScope(const GpuScope&) = delete;
    GpuScope& operator=(const GpuScope&) = delete;

private:
    GpuProfiler* profiler;
    NoorRHI::CommandList& list;
};

}
