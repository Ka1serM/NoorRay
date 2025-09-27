#pragma once
#include <RmlUi/Core.h>

class IViewModel {
public:
    virtual ~IViewModel() = default;
    virtual void BindToModel(const Rml::String& model_name, Rml::Context* context) = 0;
};