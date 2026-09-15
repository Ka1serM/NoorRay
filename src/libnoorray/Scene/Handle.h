#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>

class SceneObject;

// Stable, generation-checked reference to a scene object.
//
// The index addresses a slot in Scene's object table, so it stays valid as
// objects move within the dense array behind it. The generation counter is
// bumped whenever a slot is released, which makes a handle that outlived its
// object detectable instead of silently aliasing whichever object was
// allocated into the recycled slot -- this is what keeps a stale UI selection
// safe.
class SceneObjectHandle
{
public:
    static constexpr uint32_t InvalidIndex = ~0u;

    constexpr SceneObjectHandle() = default;
    constexpr SceneObjectHandle(const uint32_t index, const uint32_t generation)
        : index_(index), generation_(generation)
    {
    }

    constexpr uint32_t index() const { return index_; }
    constexpr uint32_t generation() const { return generation_; }
    constexpr bool isValid() const { return index_ != InvalidIndex; }
    constexpr explicit operator bool() const { return isValid(); }

    friend constexpr bool operator==(
        const SceneObjectHandle&, const SceneObjectHandle&) = default;

private:
    uint32_t index_{InvalidIndex};
    uint32_t generation_{};
};

template<>
struct std::hash<SceneObjectHandle>
{
    std::size_t operator()(const SceneObjectHandle& handle) const noexcept
    {
        return std::hash<uint64_t>{}(
            static_cast<uint64_t>(handle.index()) << 32 | handle.generation());
    }
};
