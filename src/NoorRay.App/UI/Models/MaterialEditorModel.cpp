#include "NoorRay.App/UI/Models/MaterialEditorModel.h"

namespace NoorRay::App {

void bind(Rml::DataModelConstructor& model, MaterialEditorModel& editor) {
    if (auto handle = model.RegisterStruct<MaterialParameterRow>()) {
        handle.RegisterMember("name", &MaterialParameterRow::name);
        handle.RegisterMember("components", &MaterialParameterRow::components);
        handle.RegisterMember("x", &MaterialParameterRow::x);
        handle.RegisterMember("y", &MaterialParameterRow::y);
        handle.RegisterMember("z", &MaterialParameterRow::z);
        handle.RegisterMember("hue", &MaterialParameterRow::hue);
        handle.RegisterMember("saturation", &MaterialParameterRow::saturation);
        handle.RegisterMember("value", &MaterialParameterRow::value);
        handle.RegisterMember("color", &MaterialParameterRow::color);
        handle.RegisterMember("hueColor", &MaterialParameterRow::hueColor);
        handle.RegisterMember("textureSlot", &MaterialParameterRow::textureSlot);
        handle.RegisterMember("texture", &MaterialParameterRow::texture);
        handle.RegisterMember("channel", &MaterialParameterRow::channel);
    }
    model.RegisterArray<std::vector<MaterialParameterRow>>();
    auto fields = model.RegisterStruct<MaterialEditorModel>();
    fields.RegisterMember("sceneIndex", &MaterialEditorModel::sceneIndex);
    fields.RegisterMember("name", &MaterialEditorModel::name);
    fields.RegisterMember("alphaMode", &MaterialEditorModel::alphaMode);
    fields.RegisterMember("parameters", &MaterialEditorModel::parameters);
    fields.RegisterMember("parameterIndex", &MaterialEditorModel::parameterIndex);
    fields.RegisterMember("pickerParameter", &MaterialEditorModel::pickerParameter);
    model.Bind("material", &editor);
}

}
