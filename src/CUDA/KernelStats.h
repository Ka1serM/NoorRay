#pragma once

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

// Lightweight named GPU kernel timer (pbrt-style --stats), opt-in and
// near-zero cost when disabled. time() records a start/stop cudaEvent pair
// around a launch without synchronizing, so profiling doesn't serialize the
// pipeline beyond whatever sync the caller already does. harvestFrame() reads
// all pending pairs back — call it once per frame, after the stream is known
// to have completed (e.g. right after the caller's own per-frame sync).
class KernelStats
{
public:
    void setEnabled(const bool value) { enabled = value; }
    bool isEnabled() const { return enabled; }

    template <typename Fn>
    void time(const char* name, const cudaStream_t stream, Fn&& launch)
    {
        if (!enabled)
        {
            launch();
            return;
        }
        const auto pair = acquirePair();
        cudaEventRecord(pair.first, stream);
        launch();
        cudaEventRecord(pair.second, stream);
        pending.push_back({name, pair.first, pair.second});
    }

    void harvestFrame()
    {
        if (!enabled)
            return;
        for (const auto& entry : pending)
        {
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, entry.start, entry.stop);
            Accum& accum = totals[entry.name];
            accum.totalMs += ms;
            accum.count++;
            freePairs.emplace_back(entry.start, entry.stop);
        }
        pending.clear();
        ++frameCount;
    }

    void printReport(std::ostream& os = std::cout) const
    {
        if (totals.empty())
            return;
        std::vector<std::pair<std::string, Accum>> rows(totals.begin(), totals.end());
        std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
            return a.second.totalMs > b.second.totalMs;
        });
        double grandTotal = 0.0;
        for (const auto& [name, accum] : rows)
            grandTotal += accum.totalMs;

        os << "\n--- Kernel timing (" << frameCount << " frame" << (frameCount == 1 ? "" : "s") << ") ---\n";
        os << std::left << std::setw(16) << "Kernel"
           << std::right << std::setw(12) << "Total ms"
           << std::setw(12) << "Avg ms/f"
           << std::setw(10) << "Calls"
           << std::setw(8) << "%" << "\n";
        for (const auto& [name, accum] : rows)
        {
            os << std::left << std::setw(16) << name
               << std::right << std::setw(12) << std::fixed << std::setprecision(3) << accum.totalMs
               << std::setw(12) << std::setprecision(4) << (frameCount ? accum.totalMs / static_cast<double>(frameCount) : 0.0)
               << std::setw(10) << accum.count
               << std::setw(7) << std::setprecision(1) << (grandTotal > 0 ? 100.0 * accum.totalMs / grandTotal : 0.0) << "%"
               << "\n";
        }
        os << std::left << std::setw(16) << "TOTAL"
           << std::right << std::setw(12) << std::fixed << std::setprecision(3) << grandTotal
           << std::setw(12) << std::setprecision(4) << (frameCount ? grandTotal / static_cast<double>(frameCount) : 0.0)
           << "\n";
    }

    KernelStats() = default;
    KernelStats(const KernelStats&) = delete;
    KernelStats& operator=(const KernelStats&) = delete;

    ~KernelStats()
    {
        for (const auto& entry : pending)
        {
            cudaEventDestroy(entry.start);
            cudaEventDestroy(entry.stop);
        }
        for (const auto& [start, stop] : freePairs)
        {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
    }

private:
    struct PendingEntry { std::string name; cudaEvent_t start; cudaEvent_t stop; };
    struct Accum { double totalMs = 0.0; uint64_t count = 0; };

    std::pair<cudaEvent_t, cudaEvent_t> acquirePair()
    {
        if (!freePairs.empty())
        {
            const auto pair = freePairs.back();
            freePairs.pop_back();
            return pair;
        }
        cudaEvent_t start{}, stop{};
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        return {start, stop};
    }

    bool enabled = false;
    uint64_t frameCount = 0;
    std::vector<PendingEntry> pending;
    std::vector<std::pair<cudaEvent_t, cudaEvent_t>> freePairs;
    std::unordered_map<std::string, Accum> totals;
};
