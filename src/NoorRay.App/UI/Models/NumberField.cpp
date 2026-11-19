#include "NoorRay.App/UI/Models/NumberField.h"

#include <charconv>
#include <cmath>
#include <format>

namespace NoorRay::App {

NumberField& NumberField::operator=(const float newValue) {
    value = newValue;
    text = std::format("{:.3f}", newValue);
    if (const std::size_t dot = text.find('.'); dot != Rml::String::npos) {
        const std::size_t last = text.find_last_not_of('0');
        text.erase(last == dot ? dot : last + 1);
    }
    if (text == "-0")
        text = "0";
    return *this;
}

void NumberField::setText(const Rml::String& typed) {
    text = typed;
    const std::size_t first = typed.find_first_not_of(' ');
    if (first == Rml::String::npos)
        return;
    const char* begin = typed.data() + first + (typed[first] == '+' ? 1 : 0);
    float parsed = 0.0f;
    if (const auto [end, error] = std::from_chars(begin, typed.data() + typed.size(), parsed); error == std::errc{} && std::isfinite(parsed))
        value = parsed;
}

void registerNumberField(Rml::DataModelConstructor& model) {
    model.RegisterScalar<NumberField>([](const NumberField& field, Rml::Variant& variant) { variant = field.getText(); },
                                      [](NumberField& field, const Rml::Variant& variant) { field.setText(variant.Get<Rml::String>()); });
}

}
