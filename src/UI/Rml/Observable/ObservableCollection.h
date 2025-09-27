#pragma once

#include <RmlUi/Core.h>
#include <vector>

/**
 * @class ObservableCollection
 * @brief A wrapper for std::vector that automatically notifies RmlUi on modification.
 *
 * This class is used for data-bound collections in a ViewModel. When you add,
 * remove, or clear elements, it automatically calls DirtyVariable on the data model,
 * causing the UI (e.g., a 'data-for' loop) to refresh.
 */
template <typename T>
class ObservableCollection {
public:
    // Expose the vector's iterator types so we can use range-based for loops.
    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;

    ObservableCollection() = default;

    /**
     * @brief Binds the collection to an RmlUi data model.
     * @param constructor The DataModelConstructor for the target data model.
     * @param name The name this collection will have in the RML.
     */
    void Bind(Rml::DataModelConstructor& constructor, Rml::String name) {
        this->model_handle = constructor.GetModelHandle();
        this->variable_name = std::move(name);

        // Bind a pointer to our internal vector. This is the correct way
        // to bind non-scalar types like arrays.
        constructor.Bind(this->variable_name, &collection);
    }

    // Modifying Methods (with automatic notification) 

    void push_back(const T& value) {
        collection.push_back(value);
        Notify();
    }

    void push_back(T&& value) {
        collection.push_back(std::move(value));
        Notify();
    }

    iterator erase(const_iterator position) {
        auto result = collection.erase(position);
        Notify();
        return result;
    }

    void clear() {
        collection.clear();
        Notify();
    }

    // Non-Modifying (Container-like) Methods 
    iterator begin() { return collection.begin(); }
    const_iterator begin() const { return collection.begin(); }
    iterator end() { return collection.end(); }
    const_iterator end() const { return collection.end(); }

    size_t size() const { return collection.size(); }
    bool empty() const { return collection.empty(); }

    T& operator[](size_t index) { return collection[index]; }
    const T& operator[](size_t index) const { return collection[index]; }

private:
    void Notify() {
        if (model_handle)
            model_handle.DirtyVariable(variable_name);
    }

    std::vector<T> collection;
    Rml::DataModelHandle model_handle;
    Rml::String variable_name;
};