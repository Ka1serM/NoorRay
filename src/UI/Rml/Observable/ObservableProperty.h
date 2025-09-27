#pragma once

#include <RmlUi/Core.h>
#include <functional>

/**
 * @class ObservableProperty
 * @brief A template class that wraps a value and automatically notifies RmlUi when it changes.
 *
 * This class is the core of the MVVM pattern. It holds a value and connects it to the
 * RmlUi data model, handling the "dirtying" process automatically. It also provides
 * an optional callback to notify the C++ Model of changes originating from the UI.
 */
template <typename T>
class ObservableProperty {
public:
    using ChangeCallback = std::function<void(const T&)>;

    ObservableProperty() = default;
    explicit ObservableProperty(T initial_value) : value(std::move(initial_value)) {}

    /**
     * @brief Binds the property to an RmlUi data model.
     *
     * This method is the primary way to connect the property to the UI. It encapsulates
     * all the necessary RmlUi boilerplate, setting up the internal handle and registering
     * the Get/Set functions with the data model.
     *
     * @param constructor The DataModelConstructor for the target data model.
     * @param name The name this property will have in the RML (e.g., "player_name").
     * @param on_change An optional callback to execute when the value changes. If provided,
     * it will overwrite any callback set previously via SetOnChange.
     */
    void Bind(Rml::DataModelConstructor& constructor, Rml::String name, ChangeCallback on_change = nullptr) {
        // Store the handle and name for internal use (e.g., for DirtyVariable).
        this->model_handle = constructor.GetModelHandle();
        this->property_name = std::move(name);

        if (on_change)
            this->on_change_callback = on_change;

        // Automatically call BindFunc to expose this property's Get/Set methods to RmlUi.
        constructor.BindFunc(this->property_name, [this](Rml::Variant& variant) { variant = this->Get(); }, [this](const Rml::Variant& variant) { this->Set(variant.Get<T>()); }
        );
    }
    
    /**
     * @brief Sets or replaces the callback to be executed when the value changes.
     * @param on_change The function to call.
     */
    void OnChange(ChangeCallback on_change) {
        this->on_change_callback = on_change;
    }

    /**
     * @brief Gets the current value of the property.
     * @return A const reference to the stored value.
     */
    const T& Get() const {
        return value;
    }

    /**
     * @brief Sets a new value for the property.
     *
     * If the new value is different from the current one, this method updates the internal
     * value, notifies RmlUi by dirtying the data model variable, and executes the
     * OnChange callback if one is set.
     *
     * @param new_value The new value to set.
     */
    void Set(const T& new_value) {
        if (value == new_value)
            return;
        
        value = new_value;

        // Notify RmlUi that this variable has changed so the UI can update.
        if (model_handle)
            model_handle.DirtyVariable(property_name);

        // Execute the C++ callback to notify the Model of the change.
        if (on_change_callback)
            on_change_callback(value);
    }

private:
    T value{};
    Rml::DataModelHandle model_handle;
    Rml::String property_name;
    ChangeCallback on_change_callback = nullptr;
};