#include "NoorRay/Materials/Nodes/NodeAttributes.h"

#include <string_view>

namespace NoorRay::Materials {

namespace {

constexpr std::string_view uvStreamName = "uv_stream";
constexpr std::string_view instanceAttributeName = "instance_attribute";

}

std::vector<NodeProperty> attributeRecords(const UvStreams& streams, const InstanceAttributeTable& instances) {
    std::vector<NodeProperty> records;
    for (std::size_t set = 0; set < streams.size(); ++set) {
        if (!streams[set])
            continue;
        NodeProperty& record = records.emplace_back();
        record.name = uvStreamName;
        record.kind = NodePropertyKind::Text;
        record.text = streams[set]->name;
        record.integer = static_cast<std::int64_t>(set);
        record.number.x = static_cast<float>(streams[set]->part);
    }
    const std::vector<InstanceAttribute>& attributes = instances.getAttributes();
    for (std::size_t slot = 0; slot < attributes.size(); ++slot) {
        if (attributes[slot].name == std::string(1, '\0'))
            continue;
        NodeProperty& record = records.emplace_back();
        record.name = instanceAttributeName;
        record.kind = NodePropertyKind::Text;
        record.text = attributes[slot].name;
        record.integer = static_cast<std::int64_t>(slot);
        record.number.x = static_cast<float>(attributes[slot].scope);
    }
    return records;
}

void seedAttributes(const std::span<const NodeProperty> records, AttributeTables& tables) {
    for (const NodeProperty& record : records) {
        if (record.name == uvStreamName)
            tables.uvStreams.prefer({record.text, static_cast<UvPart>(record.number.x)}, static_cast<std::size_t>(record.integer));
        else if (record.name == instanceAttributeName)
            tables.instances.place({record.text, static_cast<AttributeScope>(record.number.x)}, static_cast<std::size_t>(record.integer));
    }
}

}
