#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "Scene/Handle.h"

namespace nr {

// Called when the last reference to a resource goes away. The default drops
// whatever the resource owns and leaves an empty placeholder behind; resources
// that are plain GPU-side structs instead get an overload next to their
// registry typedef, so their layout stays free of host-only members.
template<class T>
void releaseResource(T& value)
{
    value.releaseResources();
}

// A reference-counted slot map for scene resources.
//
// Storage stays dense and slot indices stay stable, because GPU-side data
// (GpuInstance::meshIndex, Material texture indices, ...) addresses resources
// by raw index and must keep working across unrelated add/remove traffic.
// Dropping the last reference does not erase the element -- that would shift
// every later index -- it calls releaseResources() on it, which frees the
// device memory the resource owns and leaves behind an empty placeholder. The
// slot then goes on the free list and the next add move-assigns into it.
//
// Resources are therefore reclaimed as soon as nothing refers to them, while
// the index space only grows to the high-water mark of simultaneously live
// resources rather than to the total ever created.
template<class T, class Storage>
class ResourceRegistry
{
public:
    using Handle = nr::Handle<T>;
    using ValueType = T;

    ResourceRegistry() = default;
    ResourceRegistry(const ResourceRegistry&) = delete;
    ResourceRegistry& operator=(const ResourceRegistry&) = delete;

    template<class... Args>
    Handle emplace(Args&&... args)
    {
        if (freeSlots_.empty()) {
            const auto slot = static_cast<uint32_t>(slots_.size());
            storage_.emplace_back(std::forward<Args>(args)...);
            slots_.push_back(SlotState{.generation = 0, .refCount = 0, .live = true});
            return Handle(slot, 0);
        }
        const uint32_t slot = freeSlots_.back();
        freeSlots_.pop_back();
        storage_[slot] = T(std::forward<Args>(args)...);
        slots_[slot].live = true;
        slots_[slot].refCount = 0;
        return Handle(slot, slots_[slot].generation);
    }

    bool isValid(const Handle handle) const
    {
        const uint32_t slot = handle.index();
        return slot < slots_.size() && slots_[slot].live
            && slots_[slot].generation == handle.generation();
    }

    T* find(const Handle handle)
    {
        return isValid(handle) ? &storage_[handle.index()] : nullptr;
    }

    const T* find(const Handle handle) const
    {
        return isValid(handle) ? &storage_[handle.index()] : nullptr;
    }

    // Unchecked access for hot paths that have already validated the handle.
    T& operator[](const Handle handle) { return storage_[handle.index()]; }
    const T& operator[](const Handle handle) const { return storage_[handle.index()]; }

    void retain(const Handle handle)
    {
        if (isValid(handle))
            ++slots_[handle.index()].refCount;
    }

    // Returns true when this call reclaimed the resource.
    bool release(const Handle handle)
    {
        if (!isValid(handle))
            return false;
        SlotState& state = slots_[handle.index()];
        if (state.refCount > 0)
            --state.refCount;
        if (state.refCount != 0)
            return false;
        releaseResource(storage_[handle.index()]);
        state.live = false;
        ++state.generation;
        freeSlots_.push_back(handle.index());
        return true;
    }

    uint32_t useCount(const Handle handle) const
    {
        return isValid(handle) ? slots_[handle.index()].refCount : 0;
    }

    // Slot-level queries, for consumers that walk the dense storage directly
    // and must skip the placeholders left behind by released resources.
    uint32_t slotCount() const { return static_cast<uint32_t>(slots_.size()); }
    bool isLiveSlot(const uint32_t slot) const
    {
        return slot < slots_.size() && slots_[slot].live;
    }
    Handle handleAt(const uint32_t slot) const
    {
        return isLiveSlot(slot) ? Handle(slot, slots_[slot].generation) : Handle();
    }

    Storage& storage() { return storage_; }
    const Storage& storage() const { return storage_; }

    // Reclaims every live resource. Slot bookkeeping is kept so that refs which
    // outlive the clear still resolve as stale instead of aliasing whatever is
    // allocated into their slot next.
    void clear()
    {
        for (uint32_t slot = 0; slot < slots_.size(); ++slot) {
            if (!slots_[slot].live)
                continue;
            releaseResource(storage_[slot]);
            slots_[slot].live = false;
            slots_[slot].refCount = 0;
            ++slots_[slot].generation;
            freeSlots_.push_back(slot);
        }
    }

private:
    struct SlotState
    {
        uint32_t generation{};
        uint32_t refCount{};
        bool live{};
    };

    Storage storage_;
    std::vector<SlotState> slots_;
    std::vector<uint32_t> freeSlots_;
};

// An owning handle. Copying shares ownership, destruction gives it up, and the
// resource is reclaimed when the last reference goes away -- the same contract
// as shared_ptr, expressed over registry slots so the index stays GPU-visible.
template<class Registry>
class ResourceRef
{
public:
    using HandleType = typename Registry::Handle;

    ResourceRef() = default;

    ResourceRef(Registry& registry, const HandleType handle)
        : registry_(&registry), handle_(handle)
    {
        registry_->retain(handle_);
    }

    ResourceRef(const ResourceRef& other)
        : registry_(other.registry_), handle_(other.handle_)
    {
        if (registry_)
            registry_->retain(handle_);
    }

    ResourceRef(ResourceRef&& other) noexcept
        : registry_(other.registry_), handle_(other.handle_)
    {
        other.registry_ = nullptr;
        other.handle_ = HandleType();
    }

    ResourceRef& operator=(const ResourceRef& other)
    {
        if (this != &other) {
            if (other.registry_)
                other.registry_->retain(other.handle_);
            reset();
            registry_ = other.registry_;
            handle_ = other.handle_;
        }
        return *this;
    }

    ResourceRef& operator=(ResourceRef&& other) noexcept
    {
        if (this != &other) {
            reset();
            registry_ = other.registry_;
            handle_ = other.handle_;
            other.registry_ = nullptr;
            other.handle_ = HandleType();
        }
        return *this;
    }

    ~ResourceRef() { reset(); }

    void reset()
    {
        if (registry_)
            registry_->release(handle_);
        registry_ = nullptr;
        handle_ = HandleType();
    }

    HandleType handle() const { return handle_; }
    uint32_t index() const { return handle_.index(); }
    bool isValid() const { return registry_ != nullptr && registry_->isValid(handle_); }
    explicit operator bool() const { return isValid(); }

    typename Registry::ValueType* get() const
    {
        return registry_ ? registry_->find(handle_) : nullptr;
    }

    friend bool operator==(const ResourceRef& a, const ResourceRef& b)
    {
        return a.registry_ == b.registry_ && a.handle_ == b.handle_;
    }

private:
    Registry* registry_{};
    HandleType handle_{};
};

}
