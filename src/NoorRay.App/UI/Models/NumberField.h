#pragma once

#include <RmlUi/Core/DataModelHandle.h>
#include <RmlUi/Core/Types.h>

namespace NoorRay::App {

//! A float that a text field edits. It keeps the typed text, so the value binding does not rewrite a half-typed number such as "0." while the user types.
class NumberField {
public:
    NumberField(const float value = 0.0f) { *this = value; }
    NumberField& operator=(float value);
    operator float() const { return value; }

    const Rml::String& getText() const { return text; }
    //! Text that does not parse keeps the last value.
    void setText(const Rml::String& typed);

private:
    float value = 0.0f;
    Rml::String text;
};

//! Registers NumberField as a scalar of the data model; call it before any struct with a NumberField member is registered.
void registerNumberField(Rml::DataModelConstructor& model);

}
