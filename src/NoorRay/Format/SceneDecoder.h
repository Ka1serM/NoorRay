#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "NoorRay/Core/Error.h"
#include "NoorRay/Format/BinaryReader.h"
#include "NoorRay/Format/SceneCodec.h"
#include "NoorRay/Format/SceneIndex.h"

namespace NoorRay::Format {

//! Reads and checks what SceneEncoder writes, with the handles and shared objects of index.
class SceneDecoder {
public:
    SceneDecoder(BinaryReader& reader, const SceneIndex& index) : reader(reader), index(index) {}

    template <class... Values>
    void operator()(Values&... values) {
        (decode(values), ...);
    }
    void transferSize(std::size_t& size);
    void transferTopology(Scene::Mesh& mesh);

    template <class Object>
    [[nodiscard]] std::expected<Object, Core::Error> read(const std::string_view part) {
        Object object;
        decode(object);
        if (reader.isValid())
            validate(object);
        if (problem)
            return std::unexpected(std::move(*problem));
        if (std::expected<void, Core::Error> checked = reader.check(part); !checked)
            return std::unexpected(std::move(checked.error()));
        return object;
    }

private:
    template <class T>
    void decode(T& value) {
        if constexpr (!StoredAsBytes<T>) {
            transfer(*this, value);
        } else {
            reader.read(value);
            if constexpr (std::is_enum_v<T>)
                if (value > lastValue<T>)
                    refuse("an unknown value of an enumeration");
        }
    }
    void decode(std::string& text) { reader.read(text); }
    template <class T>
    void decode(std::vector<T>& values) {
        if constexpr (StoredAsBytes<T>) {
            reader.readArray(values);
        } else {
            const std::uint64_t count = reader.readCount();
            for (std::uint64_t element = 0; element < count && reader.isValid(); ++element)
                decode(values.emplace_back());
        }
    }
    template <class T, std::size_t count>
    void decode(std::array<T, count>& values) {
        for (T& value : values)
            decode(value);
    }
    template <class T>
    void decode(std::optional<T>& value) {
        if (reader.read<bool>())
            decode(value.emplace());
    }
    template <class Object>
    void decode(Core::Handle<Object>& handle) {
        const std::uint32_t object = reader.read<std::uint32_t>();
        if (object == Core::Handle<Object>::invalidIndex)
            return;
        if (object < index.get<Object>().size())
            handle = index.get<Object>()[object];
        else
            refuse("a reference to a scene object that does not exist");
    }
    template <class Object>
    void decode(std::shared_ptr<const Object>& object) {
        const std::uint32_t entry = reader.read<std::uint32_t>();
        if (entry == UINT32_MAX)
            return;
        if (entry < index.getShared<Object>().size())
            object = index.getShared<Object>()[entry];
        else
            refuse("a reference to a graph, program or node tree that does not exist");
    }

    template <class Object>
    void validate(const Object&) {}
    void validate(const Scene::Texture& texture);
    void validate(const Scene::Material& material);
    void validate(const Scene::Mesh& mesh);
    void validate(const Scene::Light& light);
    void validate(const Scene::GaussianCloud& cloud);
    void validate(const Scene::Volume& volume);
    void validate(const Scene::Environment& environment);
    void validate(const Materials::NodeTree& tree);
    void validateGraph(std::string_view owner, const std::shared_ptr<const Materials::Graph>& graph, std::size_t values, std::size_t textures);
    //! The first problem counts, and nothing reads on after it.
    void refuse(std::string_view content);

    BinaryReader& reader;
    const SceneIndex& index;
    std::optional<Core::Error> problem;
};

}
