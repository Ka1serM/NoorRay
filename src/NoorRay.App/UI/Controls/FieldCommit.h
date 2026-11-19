#pragma once

#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/Event.h>
#include <RmlUi/Core/EventListener.h>

namespace NoorRay::App {

//! Text fields apply on Enter or on blur, so a half typed number such as "0." is not applied. A blur after an edit sends the change event of Enter.
class FieldCommit final : public Rml::EventListener {
public:
    explicit FieldCommit(Rml::Element& root);
    ~FieldCommit() override;
    FieldCommit(const FieldCommit&) = delete;
    FieldCommit& operator=(const FieldCommit&) = delete;

    void ProcessEvent(Rml::Event& event) override;

    //! True for every event except the change events of a text field while the user types.
    static bool isCommit(Rml::Event& event);

private:
    Rml::Element& root;
    Rml::String committed;
};

}
