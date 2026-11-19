#include "NoorRay.App/UI/Controls/FieldCommit.h"

namespace NoorRay::App {

namespace {

bool isTextField(const Rml::Element* element) {
    return element && element->GetTagName() == "input" && element->GetAttribute<Rml::String>("type", "text") == "text";
}

}

FieldCommit::FieldCommit(Rml::Element& root) : root(root) {
    for (const Rml::EventId id : {Rml::EventId::Focus, Rml::EventId::Change, Rml::EventId::Blur})
        root.AddEventListener(id, this, true);
}

FieldCommit::~FieldCommit() {
    for (const Rml::EventId id : {Rml::EventId::Focus, Rml::EventId::Change, Rml::EventId::Blur})
        root.RemoveEventListener(id, this, true);
}

void FieldCommit::ProcessEvent(Rml::Event& event) {
    Rml::Element* field = event.GetTargetElement();
    if (!isTextField(field))
        return;
    const Rml::String value = field->GetAttribute<Rml::String>("value", "");
    switch (event.GetId()) {
    case Rml::EventId::Focus:
        committed = value;
        break;
    case Rml::EventId::Change:
        if (event.GetParameter<bool>("linebreak", false))
            committed = value;
        break;
    case Rml::EventId::Blur:
        if (value != committed) {
            committed = value;
            field->DispatchEvent(Rml::EventId::Change, {{"value", Rml::Variant(value)}, {"linebreak", Rml::Variant(true)}});
        }
        break;
    default:
        break;
    }
}

bool FieldCommit::isCommit(Rml::Event& event) {
    return event.GetId() != Rml::EventId::Change || !isTextField(event.GetTargetElement()) || event.GetParameter<bool>("linebreak", false);
}

}
