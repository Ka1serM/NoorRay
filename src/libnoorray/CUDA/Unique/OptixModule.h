#pragma once

#include <utility>

#include <optix.h>
#include <optix_stubs.h>

namespace nr::cuda
{

class UniqueOptixModule
{
public:
    UniqueOptixModule() = default;
    ~UniqueOptixModule() noexcept { reset(); }

    UniqueOptixModule(const UniqueOptixModule&) = delete;
    UniqueOptixModule& operator=(const UniqueOptixModule&) = delete;

    UniqueOptixModule(UniqueOptixModule&& other) noexcept
        : module(std::exchange(other.module, nullptr))
    {
    }

    UniqueOptixModule& operator=(UniqueOptixModule&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            module = std::exchange(other.module, nullptr);
        }
        return *this;
    }

    void reset(OptixModule newModule = nullptr) noexcept
    {
        if (module != nullptr)
            optixModuleDestroy(module);
        module = newModule;
    }

    OptixModule get() const noexcept { return module; }
    OptixModule* put() noexcept
    {
        reset();
        return &module;
    }
    explicit operator bool() const noexcept { return module != nullptr; }

private:
    OptixModule module{};
};

}
