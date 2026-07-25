#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>

namespace nr {

// Opaque, strongly typed reference to a scene resource.
//
// The index addresses a registry slot and is the value GPU-side structures
// store, so it stays stable for as long as the resource lives. The generation
// counter is bumped whenever a slot is released, which makes a handle that
// outlived its resource detectable instead of silently aliasing whichever
// resource was allocated into the recycled slot.
template<class Tag>
class Handle
{
public:
    static constexpr uint32_t InvalidIndex = ~0u;

    constexpr Handle() = default;
    constexpr Handle(const uint32_t index, const uint32_t generation)
        : index_(index), generation_(generation)
    {
    }

    constexpr uint32_t index() const { return index_; }
    constexpr uint32_t generation() const { return generation_; }
    constexpr bool isValid() const { return index_ != InvalidIndex; }
    constexpr explicit operator bool() const { return isValid(); }

    friend constexpr bool operator==(const Handle&, const Handle&) = default;

private:
    uint32_t index_{InvalidIndex};
    uint32_t generation_{};
};

}

template<class Tag>
struct std::hash<nr::Handle<Tag>>
{
    std::size_t operator()(const nr::Handle<Tag>& handle) const noexcept
    {
        return std::hash<uint64_t>{}(
            static_cast<uint64_t>(handle.index()) << 32
            | handle.generation());
    }
};

class GaussianAsset;
class MeshAsset;
class SceneObject;
class Texture;
struct Material;

using MeshAssetHandle = nr::Handle<MeshAsset>;
using GaussianAssetHandle = nr::Handle<GaussianAsset>;
using MaterialHandle = nr::Handle<Material>;
using TextureHandle = nr::Handle<Texture>;
using SceneObjectHandle = nr::Handle<SceneObject>;
