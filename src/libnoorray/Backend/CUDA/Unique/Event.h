#pragma once

#include <cuda_runtime_api.h>

#include "Backend/CUDA/Checks.h"
#include "Backend/CUDA/Unique/Handle.h"

namespace nr::cuda
{

class UniqueEvent
{
public:
    UniqueEvent() = default;
    ~UniqueEvent() noexcept = default;

    UniqueEvent(const UniqueEvent&) = delete;
    UniqueEvent& operator=(const UniqueEvent&) = delete;

    UniqueEvent(UniqueEvent&&) noexcept = default;
    UniqueEvent& operator=(UniqueEvent&&) noexcept = default;

    void create()
    {
        reset();
        NR_GPU_CHECK(cudaEventCreate(event.put()));
    }

    void reset() noexcept
    {
        event.reset();
    }

    cudaEvent_t get() const noexcept { return event.get(); }
    explicit operator bool() const noexcept { return static_cast<bool>(event); }

private:
    UniqueHandle<cudaEvent_t, cudaEventDestroy> event;
};

}
