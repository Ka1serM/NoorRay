#pragma once

#include <cuda_runtime_api.h>

#include "CUDA/Unique/Handle.h"

namespace nr::cuda
{

class UniqueExternalSemaphore
{
public:
    UniqueExternalSemaphore() = default;
    explicit UniqueExternalSemaphore(cudaExternalSemaphore_t semaphore) noexcept
        : semaphore(semaphore)
    {
    }
    ~UniqueExternalSemaphore() noexcept = default;

    UniqueExternalSemaphore(const UniqueExternalSemaphore&) = delete;
    UniqueExternalSemaphore& operator=(const UniqueExternalSemaphore&) = delete;

    UniqueExternalSemaphore(UniqueExternalSemaphore&&) noexcept = default;
    UniqueExternalSemaphore& operator=(UniqueExternalSemaphore&&) noexcept = default;

    void reset(cudaExternalSemaphore_t newSemaphore = nullptr) noexcept
    {
        semaphore.reset(newSemaphore);
    }

    cudaExternalSemaphore_t get() const noexcept { return semaphore.get(); }
    cudaExternalSemaphore_t* put() noexcept
    {
        return semaphore.put();
    }
    explicit operator bool() const noexcept { return static_cast<bool>(semaphore); }

private:
    UniqueHandle<cudaExternalSemaphore_t, cudaDestroyExternalSemaphore> semaphore;
};

}
