#include "TestViewModel.h"
#include <iostream>

#include "../Observable/ObservableProperty.h"

TestViewModel::TestViewModel() {
    player_name = ObservableProperty<Rml::String>("Sir Lancelot");
    health = ObservableProperty(85.0f);
    magic_enabled = ObservableProperty(true);

    player_name.OnChange([](const Rml::String& new_name) {
        std::cout << "Model's player name updated to: " << new_name.c_str() << std::endl;
    });

    health.OnChange([](const float& new_health) {
        std::cout << "Model's health updated to: " << new_health << std::endl;
    });

    magic_enabled.OnChange([](const bool& is_enabled) {
        std::cout << "Model's magic_enabled updated to: " << (is_enabled ? "true" : "false") << std::endl;
    });
}

void TestViewModel::BindToModel(const Rml::String& model_name, Rml::Context* context) {
    Rml::DataModelConstructor constructor = context->CreateDataModel(model_name);
    if (!constructor) return;

    player_name.Bind(constructor, "player_name");
    health.Bind(constructor, "health");
    magic_enabled.Bind(constructor, "magic_enabled");
}