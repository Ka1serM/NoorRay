#pragma once

#include <utility>

#include <optix.h>
#include <optix_stubs.h>

namespace nr::cuda
{

class UniqueOptixDeviceContext
{
public:
    UniqueOptixDeviceContext() = default;
    ~UniqueOptixDeviceContext() noexcept { reset(); }

    UniqueOptixDeviceContext(const UniqueOptixDeviceContext&) = delete;
    UniqueOptixDeviceContext& operator=(const UniqueOptixDeviceContext&) = delete;

    UniqueOptixDeviceContext(UniqueOptixDeviceContext&& other) noexcept
        : context(std::exchange(other.context, nullptr))
    {
    }

    UniqueOptixDeviceContext& operator=(UniqueOptixDeviceContext&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            context = std::exchange(other.context, nullptr);
        }
        return *this;
    }

    void reset(OptixDeviceContext newContext = nullptr) noexcept
    {
        if (context != nullptr)
            optixDeviceContextDestroy(context);
        context = newContext;
    }

    OptixDeviceContext get() const noexcept { return context; }
    OptixDeviceContext* put() noexcept
    {
        reset();
        return &context;
    }
    explicit operator bool() const noexcept { return context != nullptr; }

private:
    OptixDeviceContext context{};
};

}
