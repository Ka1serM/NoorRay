#include "NoorRay.Python/Blender/Reader.h"

#include <algorithm>
#include <cstring>
#include <format>
#include <unordered_map>

namespace NoorRay::Python {

namespace {

std::string toString(PyObject* value) {
    Object text(PyObject_Str(value));
    if (!text) {
        PyErr_Clear();
        return {};
    }
    Py_ssize_t size = 0;
    const char* data = PyUnicode_AsUTF8AndSize(text.get(), &size);
    if (!data) {
        PyErr_Clear();
        return {};
    }
    return {data, static_cast<std::size_t>(size)};
}

//! Attribute names are string literals, so their address names them; each becomes one interned Python string for the life of the process.
PyObject* attributeName(const char* name) {
    static std::unordered_map<const char*, PyObject*> names;
    PyObject*& interned = names[name];
    if (!interned)
        interned = PyUnicode_InternFromString(name);
    return interned;
}

Object& helpers() {
    static Object module;
    return module;
}

}

void Reader::setHelpers(Object module) {
    helpers() = std::move(module);
}

Object Reader::helper(const char* function, const std::span<const Object> arguments) {
    if (!helpers())
        fail("the helpers module of the add-on is not set");
    return call(helpers(), function, arguments);
}

Object Reader::string(const std::string_view value) {
    if (failed())
        return {};
    return check(PyUnicode_FromStringAndSize(value.data(), static_cast<Py_ssize_t>(value.size())), "a string");
}

Object Reader::view(const std::span<float> values) {
    if (failed())
        return {};
    const Object memory =
        check(PyMemoryView_FromMemory(reinterpret_cast<char*>(values.data()), static_cast<Py_ssize_t>(values.size_bytes()), PyBUF_WRITE), "a view");
    if (!memory)
        return {};
    return check(PyObject_CallMethod(memory.get(), "cast", "s", "f"), "a view");
}

Object Reader::check(PyObject* result, const std::string_view context) {
    if (result)
        return Object(result);
    PyObject* type = nullptr;
    PyObject* value = nullptr;
    PyObject* traceback = nullptr;
    PyErr_Fetch(&type, &value, &traceback);
    const Object owned[] = {Object(type), Object(value), Object(traceback)};
    fail(std::format("Blender data: {}: {}", context, value ? toString(value) : std::string("unknown error")));
    return {};
}

Object Reader::get(const Object& object, const char* name) {
    if (failed())
        return {};
    if (!object) {
        fail(std::format("Blender data: '{}' of None", name));
        return {};
    }
    PyObject* key = attributeName(name);
    if (!key)
        return check(nullptr, name);
    return check(PyObject_GetAttr(object.get(), key), name);
}

Object Reader::find(const Object& object, const char* name) {
    if (failed() || !object)
        return {};
    PyObject* key = attributeName(name);
    if (!key)
        return check(nullptr, name);
    PyObject* value = PyObject_GetAttr(object.get(), key);
    if (!value && PyErr_ExceptionMatches(PyExc_AttributeError)) {
        PyErr_Clear();
        return {};
    }
    return check(value, name);
}

Object Reader::attribute(const Object& object, const std::string_view name) {
    if (failed() || !object)
        return {};
    const Object key = string(name);
    if (!key)
        return {};
    PyObject* value = PyObject_GetAttr(object.get(), key.get());
    if (!value && PyErr_ExceptionMatches(PyExc_AttributeError)) {
        PyErr_Clear();
        return {};
    }
    return check(value, name);
}

Object Reader::item(const Object& collection, const char* key) {
    if (failed())
        return {};
    return check(PyMapping_GetItemString(collection.get(), key), key);
}

Object Reader::at(const Object& sequence, const std::size_t index) {
    if (failed())
        return {};
    return check(PySequence_GetItem(sequence.get(), static_cast<Py_ssize_t>(index)), "an item");
}

Object Reader::findItem(const Object& collection, const std::string_view key) {
    if (failed())
        return {};
    Object name(PyUnicode_FromStringAndSize(key.data(), static_cast<Py_ssize_t>(key.size())));
    Object method = get(collection, "get");
    if (failed())
        return {};
    return check(PyObject_CallFunctionObjArgs(method.get(), name.get(), nullptr), key);
}

std::vector<Object> Reader::items(const Object& iterable) {
    std::vector<Object> result;
    if (failed())
        return result;
    Object iterator = check(PyObject_GetIter(iterable.get()), "an iteration");
    if (!iterator)
        return result;
    while (PyObject* next = PyIter_Next(iterator.get()))
        result.emplace_back(next);
    if (PyErr_Occurred())
        check(nullptr, "an iteration");
    return result;
}

Object Reader::call(const Object& object, const char* method) {
    if (failed())
        return {};
    Object function = get(object, method);
    if (failed())
        return {};
    return check(PyObject_CallNoArgs(function.get()), method);
}

Object Reader::call(const Object& function, const Object& argument, const char* keyword, const Object& keywordValue) {
    if (failed())
        return {};
    const Object arguments(PyTuple_Pack(1, argument.get()));
    const Object keywords(Py_BuildValue("{sO}", keyword, keywordValue.get()));
    if (!arguments || !keywords)
        return check(nullptr, "a call");
    return check(PyObject_Call(function.get(), arguments.get(), keywords.get()), "a call");
}

Object Reader::call(const Object& object, const char* method, const std::span<const Object> arguments) {
    if (failed())
        return {};
    Object function = get(object, method);
    const Object tuple(PyTuple_New(static_cast<Py_ssize_t>(arguments.size())));
    if (failed() || !tuple)
        return check(nullptr, method);
    for (std::size_t index = 0; index < arguments.size(); ++index) {
        Py_XINCREF(arguments[index].get());
        PyTuple_SetItem(tuple.get(), static_cast<Py_ssize_t>(index), arguments[index].get());
    }
    return check(PyObject_Call(function.get(), tuple.get(), nullptr), method);
}

Object Reader::module(const char* name) {
    if (failed())
        return {};
    return check(PyImport_ImportModule(name), name);
}

std::size_t Reader::size(const Object& collection) {
    if (failed())
        return 0;
    const Py_ssize_t length = PyObject_Size(collection.get());
    if (length < 0) {
        check(nullptr, "a length");
        return 0;
    }
    return static_cast<std::size_t>(length);
}

float Reader::number(const Object& object, const char* name) {
    const Object value = get(object, name);
    if (failed())
        return 0.0f;
    const double result = PyFloat_AsDouble(value.get());
    if (result == -1.0 && PyErr_Occurred())
        check(nullptr, name);
    return static_cast<float>(result);
}

int Reader::integer(const Object& object, const char* name) {
    const Object value = get(object, name);
    if (failed())
        return 0;
    const long result = PyLong_AsLong(value.get());
    if (result == -1 && PyErr_Occurred())
        check(nullptr, name);
    return static_cast<int>(result);
}

bool Reader::flag(const Object& object, const char* name) {
    const Object value = get(object, name);
    if (failed())
        return false;
    const int result = PyObject_IsTrue(value.get());
    if (result < 0)
        check(nullptr, name);
    return result > 0;
}

std::string Reader::text(const Object& object, const char* name) {
    return text(get(object, name));
}

std::string Reader::text(const Object& value) {
    if (failed() || !value)
        return {};
    Py_ssize_t size = 0;
    const char* data = PyUnicode_AsUTF8AndSize(value.get(), &size);
    if (!data) {
        check(nullptr, "a string");
        return {};
    }
    return {data, static_cast<std::size_t>(size)};
}

std::span<const std::byte> Reader::bytes(const Object& value) {
    if (failed())
        return {};
    char* data = nullptr;
    Py_ssize_t size = 0;
    if (PyBytes_AsStringAndSize(value.get(), &data, &size) != 0) {
        check(nullptr, "bytes");
        return {};
    }
    return std::as_bytes(std::span(data, static_cast<std::size_t>(size)));
}

std::uintptr_t Reader::pointer(const Object& object) {
    if (!object)
        return 0;
    const Object value = call(object, "as_pointer");
    if (failed())
        return 0;
    const unsigned long long result = PyLong_AsUnsignedLongLong(value.get());
    if (PyErr_Occurred())
        check(nullptr, "as_pointer");
    return static_cast<std::uintptr_t>(result);
}

Values Reader::values(const Object& value) {
    Values result;
    if (failed())
        return result;
    if (PyFloat_Check(value.get()) || PyLong_Check(value.get())) {
        result.values[0] = static_cast<float>(PyFloat_AsDouble(value.get()));
        result.count = 1;
        return result;
    }
    const std::size_t length = size(value);
    result.count = std::min<std::size_t>(length, result.values.size());
    for (std::size_t index = 0; index < result.count; ++index) {
        const Object component = at(value, index);
        if (failed())
            return {};
        result.values[index] = static_cast<float>(PyFloat_AsDouble(component.get()));
    }
    if (PyErr_Occurred())
        check(nullptr, "a vector");
    return result;
}

Values Reader::values(const Object& object, const char* name) {
    return values(get(object, name));
}

std::vector<std::int32_t> Reader::integers(const Object& object, const char* name) {
    std::vector<std::int32_t> result;
    for (const Object& value : items(get(object, name))) {
        const long number = PyLong_AsLong(value.get());
        if (number == -1 && PyErr_Occurred()) {
            check(nullptr, name);
            return {};
        }
        result.push_back(static_cast<std::int32_t>(number));
    }
    return result;
}

Core::Transform Reader::matrix(const Object& matrix) {
    Core::Transform transform;
    if (failed())
        return transform;
    Py_buffer view{};
    if (PyObject_GetBuffer(matrix.get(), &view, PyBUF_STRIDES | PyBUF_FORMAT) != 0) {
        check(nullptr, "a matrix");
        return transform;
    }
    const bool valid = view.ndim == 2 && view.shape[0] == 4 && view.shape[1] == 4 && view.format && std::string_view(view.format) == "f";
    if (valid) {
        const auto* bytes = static_cast<const char*>(view.buf);
        for (int row = 0; row < 3; ++row)
            for (int column = 0; column < 4; ++column)
                std::memcpy(&transform.rows[row][column], bytes + row * view.strides[0] + column * view.strides[1], sizeof(float));
    }
    PyBuffer_Release(&view);
    if (!valid)
        fail("Blender data: a matrix is not 4x4 floats");
    return transform;
}

void Reader::foreachGet(const Object& collection, const char* attribute, void* data, const std::size_t size, const char* format) {
    if (failed())
        return;
    const Object memory = check(PyMemoryView_FromMemory(static_cast<char*>(data), static_cast<Py_ssize_t>(size), PyBUF_WRITE), "foreach_get");
    if (!memory)
        return;
    const Object cast = check(PyObject_CallMethod(memory.get(), "cast", "s", format), "foreach_get");
    if (!cast)
        return;
    if (!attribute) {
        check(PyObject_CallMethod(collection.get(), "foreach_get", "O", cast.get()), "foreach_get");
        return;
    }
    check(PyObject_CallMethod(collection.get(), "foreach_get", "sO", attribute, cast.get()), attribute);
}

void Reader::fail(std::string message) {
    if (!error)
        error = Core::Error{Core::ErrorCode::InvalidFile, std::move(message)};
}

std::expected<void, Core::Error> Reader::takeError() {
    if (!error)
        return {};
    Core::Error taken = std::move(*error);
    error.reset();
    return std::unexpected(std::move(taken));
}

}
