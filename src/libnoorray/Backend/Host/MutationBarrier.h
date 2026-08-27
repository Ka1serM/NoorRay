#pragma once

namespace nr
{
// Scene mutations are serialized by Scene's publication barrier. The Vulkan
// renderer owns synchronization, so host edits do not require a device-wide
// vendor runtime call here.
inline bool synchronizeBeforeManagedMutation(const char*) noexcept { return true; }
}
