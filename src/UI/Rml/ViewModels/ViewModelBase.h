#pragma once
#include "RmlUi/Core/DataModelHandle.h"
#include "RmlUi/Core/Context.h"
#include <utility>
#include "RmlUi/Core/EventListener.h"
#include "UI/Rml/Observable/ObservableCollection.h"
#include "UI/Rml/Observable/ObservableProperty.h"

class ViewModelBase : public Rml::EventListener {
protected:
    Rml::DataModelConstructor data_model_;

public:   
    ViewModelBase(Rml::Context* ctx, const Rml::String& name) {
        data_model_ = ctx->CreateDataModel(name);
    }

    virtual ~ViewModelBase() = default;
    virtual void Update() {}

    void ProcessEvent(Rml::Event&) override {}

    // Bind an ObservableProperty<T>
    template <typename T>
    void Bind(const char* name, ObservableProperty<T>& prop) {
        if (!data_model_)
            return;
            
        data_model_.Bind(name, prop.Ptr());

        // When the property's value changes, automatically notify the data model
        auto handle = data_model_.GetModelHandle();
        prop.OnChange([handle, name_str = Rml::String(name)](const T&) mutable {
            if (handle) handle.DirtyVariable(name_str);
        });
    }

    // Bind an ObservableCollection<T>
    template <typename T>
    void Bind(const char* name, ObservableCollection<T>& coll) {
        if (!data_model_)
            return;
        
        data_model_.Bind(name, coll.Ptr());

        // Capture the handle by moving into the lambda
        auto handle = data_model_.GetModelHandle(); // rvalue
        coll.SetOnChanged([handle = std::move(handle), name_str = Rml::String(name)]() mutable {
            if (handle) handle.DirtyVariable(name_str);
        });
    }
    
    void BindAction(const char* name, std::function<void(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList&)> func) {
        if (!data_model_)
            return;
        data_model_.BindEventCallback(name, func);
    }

    // Binds a non-static member function automatically
    template <typename T, typename Ret, typename... Args>
    void BindAction(const char* name, T* obj, Ret(T::*func)(Args...)) {
        BindAction(name, [obj, func](Rml::DataModelHandle, Rml::Event&, const Rml::VariantList&) {
            (obj->*func)(); // call member function with no args
        });
    }
};