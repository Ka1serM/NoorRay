#include "NoorRay.App/UI/Models/AppModel.h"

namespace NoorRay::App {

AppModel::AppModel(RmlKit::UI::Model& model) {
    model.bindData([this](Rml::DataModelConstructor& constructor) {
        registerNumberField(constructor);
        bind(constructor, outliner);
        bind(constructor, inspector, environment);
        bind(constructor, materialEditor);
        bind(constructor, renderSettings);
        bind(constructor, status);
        bind(constructor, timeline);
        bind(constructor, training);
        constructor.Bind("canSave", &canSave);
        constructor.Bind("menu", &menu);
        constructor.Bind("tool", &tool);
        constructor.Bind("addShape", &addShape);
        constructor.Bind("addLight", &addLight);
        constructor.Bind("materialEditorOpen", &materialEditorOpen);
    });
}

}
