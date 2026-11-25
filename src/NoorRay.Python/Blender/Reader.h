#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <Python.h>

#include "NoorRay/Core/Error.h"
#include "NoorRay/Core/Math.h"

namespace NoorRay::Python {

class Object {
public:
    Object() = default;
    explicit Object(PyObject* owned) : value(owned) {}
    Object(const Object& other) : value(other.value) { Py_XINCREF(value); }
    Object(Object&& other) noexcept : value(std::exchange(other.value, nullptr)) {}
    Object& operator=(Object other) noexcept {
        std::swap(value, other.value);
        return *this;
    }
    ~Object() { Py_XDECREF(value); }

    static Object borrow(PyObject* borrowed) {
        Py_XINCREF(borrowed);
        return Object(borrowed);
    }

    PyObject* get() const { return value; }
    explicit operator bool() const { return value != nullptr && value != Py_None; }

private:
    PyObject* value = nullptr;
};

//! A socket value or a property array: one to four floats.
struct Values {
    std::array<float, 4> values{};
    std::size_t count = 0;
};

//! Reads Blender data through bpy. The first failure sticks: later reads give empty values, and takeError reports it.
class Reader {
public:
    //! Attribute names must be string literals: the reader keeps one interned Python string per name address.
    Object get(const Object& object, const char* name);
    //! A missing or None attribute gives an empty object and no failure.
    Object find(const Object& object, const char* name);
    //! An attribute by a name that is not a literal; a missing or None attribute gives an empty object and no failure.
    Object attribute(const Object& object, std::string_view name);
    Object item(const Object& collection, const char* key);
    Object at(const Object& sequence, std::size_t index);
    //! collection.get(key): an empty object when the key is missing.
    Object findItem(const Object& collection, std::string_view key);
    std::vector<Object> items(const Object& iterable);
    //! Visits each item while the iterator holds it; the items of depsgraph.object_instances are valid only then.
    template <class Visit>
    void forEach(const Object& iterable, Visit&& visit) {
        if (failed())
            return;
        Object iterator = check(PyObject_GetIter(iterable.get()), "an iteration");
        if (!iterator)
            return;
        while (PyObject* next = PyIter_Next(iterator.get())) {
            visit(Object(next));
            if (failed())
                return;
        }
        if (PyErr_Occurred())
            check(nullptr, "an iteration");
    }
    Object call(const Object& object, const char* method);
    Object call(const Object& function, const Object& argument, const char* keyword, const Object& keywordValue);
    //! object.method(arguments...).
    Object call(const Object& object, const char* method, std::span<const Object> arguments);
    Object module(const char* name);
    //! Calls a function of the add-on's helpers module.
    Object helper(const char* function, std::span<const Object> arguments);
    Object string(std::string_view value);
    //! A writable memory view of float32 values, valid while the values live.
    Object view(std::span<float> values);
    static void setHelpers(Object module);
    std::size_t size(const Object& collection);

    float number(const Object& object, const char* name);
    int integer(const Object& object, const char* name);
    bool flag(const Object& object, const char* name);
    std::string text(const Object& object, const char* name);
    std::string text(const Object& value);
    //! The bytes of a Python bytes object, valid while the object lives.
    std::span<const std::byte> bytes(const Object& value);
    //! as_pointer() of a bpy struct: a stable identity while Blender keeps the data.
    std::uintptr_t pointer(const Object& object);
    Values values(const Object& value);
    Values values(const Object& object, const char* name);
    std::vector<std::int32_t> integers(const Object& object, const char* name);
    //! The first three rows of a 4x4 mathutils matrix.
    Core::Transform matrix(const Object& matrix);

    //! The element type follows the property: Blender takes no int32 buffer for an unsigned property such as loop_start, and no uint32 one for a signed.
    template <class T>
        requires(std::is_same_v<T, float> || std::is_same_v<T, std::int32_t> || std::is_same_v<T, std::uint32_t> || std::is_same_v<T, bool>)
    void foreachGet(const Object& collection, const char* attribute, std::span<T> values) {
        const char* format = std::is_same_v<T, float> ? "f" : std::is_same_v<T, std::int32_t> ? "i" : std::is_same_v<T, bool> ? "?" : "I";
        foreachGet(collection, attribute, values.data(), values.size_bytes(), format);
    }

    void fail(std::string message);
    bool failed() const { return error.has_value(); }
    const std::optional<Core::Error>& getError() const { return error; }
    [[nodiscard]] std::expected<void, Core::Error> takeError();

private:
    void foreachGet(const Object& collection, const char* attribute, void* data, std::size_t size, const char* format);
    Object check(PyObject* result, std::string_view context);

    std::optional<Core::Error> error;
};

}
