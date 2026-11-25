#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "NoorRay/Materials/NodeTree.h"
#include "NoorRay/Scene/Mesh.h"

namespace NoorRay::Materials {

//! The two values that a UV set holds of a named attribute: its first two, its last two, or the mean of its first three and its alpha.
enum class UvPart : std::uint8_t { Low, High, Mean };

//! Streams that the mesh reader derives, as the Pointiness and the Random Per Island of Cycles; no attribute has such a name.
inline const std::string pointinessStream = "\x01pointiness";
inline const std::string islandStream = "\x01island";

struct UvStream {
    std::string name;
    UvPart part = UvPart::Low;

    auto operator<=>(const UvStream&) const = default;
};

//! The named stream of each UV set; set 0 is always the render UV map.
using UvStreams = std::array<std::optional<UvStream>, Scene::maxUvSets>;

//! Gives each stream the same UV set in every material where that is free, so that the materials of one mesh agree.
class UvStreamTable {
public:
    //! The set of the stream in streams, or none when all sets hold other streams.
    std::optional<std::size_t> assign(UvStreams& streams, const UvStream& stream) {
        const std::size_t preferred = preferences.try_emplace(stream, 1 + preferences.size() % (Scene::maxUvSets - 1)).first->second;
        for (std::size_t offset = 0; offset < Scene::maxUvSets - 1; ++offset) {
            const std::size_t set = 1 + (preferred - 1 + offset) % (Scene::maxUvSets - 1);
            if (streams[set] && streams[set] != stream)
                continue;
            streams[set] = stream;
            return set;
        }
        return std::nullopt;
    }
    //! Makes set the first choice of the stream, as an earlier compile chose it.
    void prefer(const UvStream& stream, const std::size_t set) { preferences.insert_or_assign(stream, set); }

private:
    std::map<UvStream, std::size_t> preferences;
};

enum class AttributeScope : std::uint8_t { Object, Instancer, ViewLayer };

//! An attribute that is constant over an instance.
struct InstanceAttribute {
    std::string name;
    AttributeScope scope = AttributeScope::Object;

    bool operator==(const InstanceAttribute&) const = default;
};

//! The slots of the instance attributes that the materials read; a slot keeps its attribute.
class InstanceAttributeTable {
public:
    std::size_t assign(const InstanceAttribute& attribute) {
        const auto found = std::ranges::find(attributes, attribute);
        const auto slot = static_cast<std::size_t>(found - attributes.begin());
        if (found == attributes.end())
            attributes.push_back(attribute);
        return slot;
    }
    //! Puts the attribute in slot, as an earlier compile placed it; a slot that holds another attribute keeps it.
    void place(const InstanceAttribute& attribute, const std::size_t slot) {
        if (std::ranges::contains(attributes, attribute))
            return;
        if (attributes.size() <= slot)
            attributes.resize(slot + 1, {std::string(1, '\0'), AttributeScope::Object});
        if (attributes[slot].name == std::string(1, '\0'))
            attributes[slot] = attribute;
    }
    const std::vector<InstanceAttribute>& getAttributes() const { return attributes; }

private:
    std::vector<InstanceAttribute> attributes;
};

//! Where the attributes that the materials read are.
struct AttributeTables {
    UvStreamTable uvStreams;
    InstanceAttributeTable instances;
};

//! The records that a compile stores in the properties of its root tree: the UV set of each stream of the material and the slot of each instance attribute.
std::vector<NodeProperty> attributeRecords(const UvStreams& streams, const InstanceAttributeTable& instances);
//! Gives the streams and instance attributes of the records the sets and slots that the compile that stored them chose.
void seedAttributes(std::span<const NodeProperty> records, AttributeTables& tables);

}
