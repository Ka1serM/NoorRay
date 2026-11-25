#include "NoorRay.App/UI/Models/AppModel.h"

namespace NoorRay::App {

AppModel::AppModel(RmlKit::UI::Model& model) {
    model.bindData([this](Rml::DataModelConstructor& constructor) {
        registerNumberField(constructor);
        bind(constructor, outliner);
        bind(constructor, inspector, environment);
        bind(constructor, materialEditor);
        bind(constructor, nodeGraph);
        bind(constructor, renderSettings);
        bind(constructor, status);
        bind(constructor, timeline);
        if (auto fields = constructor.RegisterStruct<ViewportSourceModel>()) {
            fields.RegisterMember("shown", &ViewportSourceModel::shown);
            fields.RegisterMember("title", &ViewportSourceModel::title);
            fields.RegisterMember("detail", &ViewportSourceModel::detail);
            fields.RegisterMember("action", &ViewportSourceModel::action);
            fields.RegisterMember("actionEnabled", &ViewportSourceModel::actionEnabled);
        }
        constructor.Bind("viewportSource", &viewportSource);
        constructor.Bind("canSave", &canSave);
        constructor.Bind("menu", &menu);
        constructor.Bind("tool", &tool);
        constructor.Bind("addShape", &addShape);
        constructor.Bind("addLight", &addLight);
        constructor.Bind("materialEditorOpen", &materialEditorOpen);
        constructor.Bind("nodeEditorOpen", &nodeEditorOpen);
    });
}

}
