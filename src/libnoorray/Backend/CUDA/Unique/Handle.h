#pragma once

#include <utility>

namespace nr::cuda
{

template <typename Handle, auto Destroy>
class UniqueHandle
{
public:
    UniqueHandle() = default;
    explicit UniqueHandle(Handle handle) noexcept : handle(handle) {}
    ~UniqueHandle() noexcept { reset(); }

    UniqueHandle(const UniqueHandle&) = delete;
    UniqueHandle& operator=(const UniqueHandle&) = delete;

    UniqueHandle(UniqueHandle&& other) noexcept
        : handle(std::exchange(other.handle, Handle{})) {}

    UniqueHandle& operator=(UniqueHandle&& other) noexcept
    {
        if (this != &other)
            reset(std::exchange(other.handle, Handle{}));
        return *this;
    }

    void reset(Handle replacement = Handle{}) noexcept
    {
        if (handle != Handle{})
            (void)Destroy(handle);
        handle = replacement;
    }

    Handle get() const noexcept { return handle; }
    Handle* put() noexcept
    {
        reset();
        return &handle;
    }
    explicit operator bool() const noexcept { return handle != Handle{}; }

private:
    Handle handle{};
};

}
