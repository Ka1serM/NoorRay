#include "NoorRay/Render/GpuProfiler.h"

#include <algorithm>
#include <utility>

namespace NoorRay::Render {

std::expected<void, Core::Error> GpuProfiler::open(NoorRHI::Device& openedDevice) {
    device = &openedDevice;
    clear();
    return {};
}

void GpuProfiler::begin(NoorRHI::CommandList& list, const std::string_view name) {
    if (!result)
        return;
    const std::size_t index = names.size();
    if (index == queries.size()) {
        NoorRHI::TimestampQuery query;
        if (auto opened = Core::check(query.open(*device)); !opened) {
            result = std::move(opened);
            return;
        }
        queries.push_back(std::move(query));
    }
    names.emplace_back(name);
    openScopes.push_back(index);
    list.beginTimestamp(queries[index]);
}

void GpuProfiler::end(NoorRHI::CommandList& list) {
    if (!result || openScopes.empty())
        return;
    list.endTimestamp(queries[openScopes.back()]);
    openScopes.pop_back();
}

std::expected<void, Core::Error> GpuProfiler::collect() {
    if (!result)
        return std::exchange(result, {});
    if (!openScopes.empty())
        return Core::fail(Core::ErrorCode::InvalidState, "a GPU scope of the frame is still open");
    for (std::size_t index = 0; index < names.size(); ++index) {
        Total& total = totals[names[index]];
        total.name = names[index];
        total.milliseconds += queries[index].getMilliseconds();
        ++total.calls;
    }
    names.clear();
    ++frames;
    return {};
}

std::vector<GpuProfiler::Total> GpuProfiler::getTotals() const {
    std::vector<Total> sorted;
    sorted.reserve(totals.size());
    for (const auto& [name, total] : totals)
        sorted.push_back(total);
    std::ranges::sort(sorted, [](const Total& left, const Total& right) { return left.milliseconds > right.milliseconds; });
    return sorted;
}

void GpuProfiler::clear() {
    names.clear();
    openScopes.clear();
    totals.clear();
    frames = 0;
    result = {};
}

}
