#pragma once
#include "IViewModel.h"
#include "../Observable/ObservableProperty.h"

class TestViewModel : public IViewModel {
public:
    TestViewModel();
    void BindToModel(const Rml::String& model_name, Rml::Context* context) override;
private:
    ObservableProperty<Rml::String> player_name;
    ObservableProperty<float> health;
    ObservableProperty<bool> magic_enabled;
};