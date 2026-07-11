#pragma once

#include <utility>

#include <cuda_runtime_api.h>

#include "CUDA/Unique/Check.h"

namespace nr::cuda
{

class UniqueEvent
{
public:
    UniqueEvent() = default;
    ~UniqueEvent() noexcept { reset(); }

    UniqueEvent(const UniqueEvent&) = delete;
    UniqueEvent& operator=(const UniqueEvent&) = delete;

    UniqueEvent(UniqueEvent&& other) noexcept
        : event(std::exchange(other.event, nullptr))
    {
    }

    UniqueEvent& operator=(UniqueEvent&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            event = std::exchange(other.event, nullptr);
        }
        return *this;
    }

    void create()
    {
        reset();
        NR_CUDA_UNIQUE_CHECK(cudaEventCreate(&event));
    }

    void reset() noexcept
    {
        if (event != nullptr)
            cudaEventDestroy(event);
        event = nullptr;
    }

    cudaEvent_t get() const noexcept { return event; }
    explicit operator bool() const noexcept { return event != nullptr; }

private:
    cudaEvent_t event{};
};

}
