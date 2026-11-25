#pragma once

#include <algorithm>
#include <cstddef>
#include <thread>

namespace NoorRay::Python {

inline constexpr std::size_t minimumChunk = 1 << 16;

//! The number of chunks parallelFor splits count into: one per hardware thread, and none smaller than minimumChunk.
inline std::size_t chunkCount(const std::size_t count) {
    const std::size_t threads = std::max(1u, std::thread::hardware_concurrency());
    return std::clamp<std::size_t>(count / minimumChunk, 1, threads);
}

//! Runs run(context, chunk) for each chunk on a pool of hardware threads that lives as long as the module; the caller takes chunks too and waits for all.
void runChunks(std::size_t chunks, void (*run)(void*, std::size_t), void* context);

//! Calls body(chunk, begin, end) for each chunk of [0, count) and waits for all of them.
template <class Body>
void parallelFor(const std::size_t count, Body&& body) {
    const std::size_t chunks = chunkCount(count);
    if (chunks == 1) {
        body(std::size_t(0), std::size_t(0), count);
        return;
    }
    struct Context {
        Body& body;
        std::size_t count;
        std::size_t chunks;
    } context{body, count, chunks};
    const auto run = [](void* opaque, const std::size_t chunk) {
        Context& context = *static_cast<Context*>(opaque);
        context.body(chunk, context.count * chunk / context.chunks, context.count * (chunk + 1) / context.chunks);
    };
    runChunks(chunks, run, &context);
}

}
