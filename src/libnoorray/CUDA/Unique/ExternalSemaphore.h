#pragma once

#include <utility>

#include <cuda_runtime_api.h>

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
    ~UniqueExternalSemaphore() noexcept { reset(); }

    UniqueExternalSemaphore(const UniqueExternalSemaphore&) = delete;
    UniqueExternalSemaphore& operator=(const UniqueExternalSemaphore&) = delete;

    UniqueExternalSemaphore(UniqueExternalSemaphore&& other) noexcept
        : semaphore(std::exchange(other.semaphore, nullptr))
    {
    }

    UniqueExternalSemaphore& operator=(UniqueExternalSemaphore&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            semaphore = std::exchange(other.semaphore, nullptr);
        }
        return *this;
    }

    void reset(cudaExternalSemaphore_t newSemaphore = nullptr) noexcept
    {
        if (semaphore != nullptr)
            cudaDestroyExternalSemaphore(semaphore);
        semaphore = newSemaphore;
    }

    cudaExternalSemaphore_t get() const noexcept { return semaphore; }
    cudaExternalSemaphore_t* put() noexcept
    {
        reset();
        return &semaphore;
    }
    explicit operator bool() const noexcept { return semaphore != nullptr; }

private:
    cudaExternalSemaphore_t semaphore{};
};

}
