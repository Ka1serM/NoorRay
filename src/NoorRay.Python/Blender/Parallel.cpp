#include "NoorRay.Python/Blender/Parallel.h"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <vector>

namespace NoorRay::Python {

namespace {

class Pool {
public:
    Pool() {
        const std::size_t threads = std::max(1u, std::thread::hardware_concurrency()) - 1;
        for (std::size_t index = 0; index < threads; ++index)
            workers.emplace_back([this](const std::stop_token stop) { work(stop); });
    }

    ~Pool() {
        for (std::jthread& worker : workers)
            worker.request_stop();
        wake.notify_all();
    }

    void run(const std::size_t chunks, void (*task)(void*, std::size_t), void* context) {
        //! One job at a time: a second caller waits for the first job to end.
        const std::scoped_lock job(jobMutex);
        {
            //! A worker that woke late for the last job may still be reading it.
            std::unique_lock lock(mutex);
            done.wait(lock, [&] { return active == 0; });
            this->task = task;
            this->context = context;
            this->chunks = chunks;
            next.store(0, std::memory_order_relaxed);
            remaining = chunks;
            ++epoch;
        }
        wake.notify_all();
        take();
        std::unique_lock lock(mutex);
        done.wait(lock, [&] { return remaining == 0 && active == 0; });
    }

private:
    void take() {
        std::size_t finished = 0;
        for (std::size_t chunk = next.fetch_add(1, std::memory_order_relaxed); chunk < chunks; chunk = next.fetch_add(1, std::memory_order_relaxed)) {
            task(context, chunk);
            ++finished;
        }
        const std::scoped_lock lock(mutex);
        remaining -= finished;
    }

    void work(const std::stop_token stop) {
        std::uint64_t seen = 0;
        while (true) {
            {
                std::unique_lock lock(mutex);
                wake.wait(lock, [&] { return stop.stop_requested() || epoch != seen; });
                if (stop.stop_requested())
                    return;
                seen = epoch;
                ++active;
            }
            take();
            const std::scoped_lock lock(mutex);
            --active;
            done.notify_all();
        }
    }

    std::mutex jobMutex;
    std::mutex mutex;
    std::condition_variable_any wake;
    std::condition_variable done;
    void (*task)(void*, std::size_t) = nullptr;
    void* context = nullptr;
    std::size_t chunks = 0;
    std::atomic<std::size_t> next{0};
    std::size_t remaining = 0;
    std::size_t active = 0;
    std::uint64_t epoch = 0;
    std::vector<std::jthread> workers;
};

}

void runChunks(const std::size_t chunks, void (*const run)(void*, std::size_t), void* const context) {
    static Pool pool;
    pool.run(chunks, run, context);
}

}
