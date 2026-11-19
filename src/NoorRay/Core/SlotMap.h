#pragma once

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "NoorRay/Core/Handle.h"

namespace NoorRay::Core {

//! Values stay dense: removal moves the last value into the hole, so iteration and upload are linear.
template <class T>
class SlotMap {
public:
    using Key = Handle<T>;

    Key insert(T value) {
        const std::uint32_t denseIndex = static_cast<std::uint32_t>(values.size());
        values.push_back(std::move(value));
        std::uint32_t slot = 0;
        if (freeSlots.empty()) {
            slot = static_cast<std::uint32_t>(slots.size());
            slots.push_back({denseIndex, 0});
        } else {
            slot = freeSlots.back();
            freeSlots.pop_back();
            slots[slot].dense = denseIndex;
        }
        owners.push_back(slot);
        return {slot, slots[slot].generation};
    }

    bool erase(const Key key) {
        if (!contains(key))
            return false;
        const std::uint32_t hole = slots[key.index].dense;
        const std::uint32_t last = static_cast<std::uint32_t>(values.size() - 1);
        if (hole != last) {
            values[hole] = std::move(values[last]);
            owners[hole] = owners[last];
            slots[owners[hole]].dense = hole;
        }
        values.pop_back();
        owners.pop_back();
        Slot& slot = slots[key.index];
        slot.dense = Key::invalidIndex;
        ++slot.generation;
        freeSlots.push_back(key.index);
        return true;
    }

    void clear() {
        for (std::uint32_t owner : owners) {
            slots[owner].dense = Key::invalidIndex;
            ++slots[owner].generation;
            freeSlots.push_back(owner);
        }
        values.clear();
        owners.clear();
    }

    bool contains(const Key key) const {
        return key.index < slots.size() && slots[key.index].generation == key.generation && slots[key.index].dense != Key::invalidIndex;
    }

    T* find(const Key key) { return contains(key) ? &values[slots[key.index].dense] : nullptr; }
    const T* find(const Key key) const { return contains(key) ? &values[slots[key.index].dense] : nullptr; }

    std::uint32_t denseIndex(const Key key) const { return contains(key) ? slots[key.index].dense : Key::invalidIndex; }
    Key keyAt(const std::uint32_t dense) const { return {owners[dense], slots[owners[dense]].generation}; }

    std::span<T> getValues() { return values; }
    std::span<const T> getValues() const { return values; }
    std::size_t size() const { return values.size(); }
    bool empty() const { return values.empty(); }

private:
    struct Slot {
        std::uint32_t dense = Key::invalidIndex;
        std::uint32_t generation = 0;
    };

    std::vector<T> values;
    std::vector<std::uint32_t> owners;
    std::vector<Slot> slots;
    std::vector<std::uint32_t> freeSlots;
};

}
