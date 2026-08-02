#pragma once

#include <utility>

#include <optix.h>
#include <optix_stubs.h>

namespace nr::cuda
{

class UniqueOptixProgramGroup
{
public:
    UniqueOptixProgramGroup() = default;
    ~UniqueOptixProgramGroup() noexcept { reset(); }

    UniqueOptixProgramGroup(const UniqueOptixProgramGroup&) = delete;
    UniqueOptixProgramGroup& operator=(const UniqueOptixProgramGroup&) = delete;

    UniqueOptixProgramGroup(UniqueOptixProgramGroup&& other) noexcept
        : group(std::exchange(other.group, nullptr))
    {
    }

    UniqueOptixProgramGroup& operator=(UniqueOptixProgramGroup&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            group = std::exchange(other.group, nullptr);
        }
        return *this;
    }

    void reset(OptixProgramGroup newGroup = nullptr) noexcept
    {
        if (group != nullptr)
            optixProgramGroupDestroy(group);
        group = newGroup;
    }

    OptixProgramGroup get() const noexcept { return group; }
    OptixProgramGroup* put() noexcept
    {
        reset();
        return &group;
    }
    explicit operator bool() const noexcept { return group != nullptr; }

private:
    OptixProgramGroup group{};
};

}
