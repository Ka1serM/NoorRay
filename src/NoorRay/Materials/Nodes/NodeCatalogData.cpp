//! tools/generate_node_catalog.py made this file from Blender 5.2.0 LTS.
#include "NoorRay/Materials/Nodes/NodeCatalogData.h"

namespace NoorRay::Materials {

namespace {

constexpr NodePropertyData table0[] = {{"shrink", "Shrink", NodePropertyKind::Boolean, 0.0f, 1, "", {}, {}, 1.0f, 0.0f, true, true},
                                       {"label_size", "Label Font Size", NodePropertyKind::Integer, 0.0f, 20, "", {}, {}, 8.0f, 64.0f, true, true}};
constexpr NodeTreeKind table1[] = {NodeTreeKind::Material, NodeTreeKind::Light, NodeTreeKind::World};
constexpr NodeSocketData table2[] = {{"Input", "Input", NodeSocketType::Color, 4, true, false, false, {0.0f, 0.0f, 0.0f, 1.0f}, "", 1.0f, 0.0f, ""}};
constexpr NodeSocketData table3[] = {{"Output", "Output", NodeSocketType::Color, 4, true, false, false, {0.0f, 0.0f, 0.0f, 1.0f}, "", 1.0f, 0.0f, ""}};
constexpr NodePropertyData table4[] = {
    {"socket_idname", "Type of socket", NodePropertyKind::Text, 0.0f, 0, "NodeSocketColor", {}, {}, 1.0f, 0.0f, false, true}};
constexpr NodeMuting table5[] = {{0x0, 0}};
constexpr NodeSocketData table6[] = {{"Shader", "Shader", NodeSocketType::Shader, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", 1.0f, 0.0f, ""},
                                     {"Shader_001", "Shader", NodeSocketType::Shader, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", 1.0f, 0.0f, ""}};
constexpr NodeSocketData table7[] = {{"Shader", "Shader", NodeSocketType::Shader, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", 1.0f, 0.0f, ""}};
constexpr NodeMuting table8[] = {{0x3, 0}};
constexpr NodeSocketData table9[] = {
    {"Color", "Color", NodeSocketType::Color, 4, true, false, false, {1.0f, 1.0f, 1.0f, 1.0f}, "", 1.0f, 0.0f, ""},
    {"Distance", "Distance", NodeSocketType::Float, 1, true, false, false, {1.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1000.0f, "NONE"},
    {"Normal", "Normal", NodeSocketType::Vector, 3, true, false, true, {0.0f, 0.0f, 0.0f, 0.0f}, "", -1.0f, 1.0f, "NONE"}};
constexpr NodeSocketData table10[] = {
    {"Color", "Color", NodeSocketType::Color, 4, true, false, false, {0.800000012f, 0.800000012f, 0.800000012f, 1.0f}, "", 1.0f, 0.0f, ""},
    {"AO", "AO", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"}};
constexpr NodePropertyData table11[] = {{"samples", "Samples", NodePropertyKind::Integer, 0.0f, 16, "", {}, {}, 1.0f, 128.0f, true, true},
                                        {"inside", "Inside", NodePropertyKind::Boolean, 0.0f, 0, "", {}, {}, 1.0f, 0.0f, true, true},
                                        {"only_local", "Only Local", NodePropertyKind::Boolean, 0.0f, 0, "", {}, {}, 1.0f, 0.0f, true, true},
                                        {"world_distance", "", NodePropertyKind::Number, 10.0f, 0, "", {}, {}, 1.0f, 0.0f, false, true}};
constexpr NodeTreeKind table12[] = {NodeTreeKind::Material};
constexpr NodeMuting table13[] = {{0x3, 0}, {0x6, 1}};
constexpr NodeSocketData table14[] = {
    {"Color", "Color", NodeSocketType::Color, 4, true, false, false, {0.800000012f, 0.800000012f, 0.800000012f, 1.0f}, "", 1.0f, 0.0f, ""},
    {"Vector", "Vector", NodeSocketType::Vector, 3, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Fac", "Factor", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Alpha", "Alpha", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"}};
constexpr NodeEnumData table15[] = {{"GEOMETRY", "Geometry"}, {"OBJECT", "Object"}, {"INSTANCER", "Instancer"}, {"VIEW_LAYER", "View Layer"}};
constexpr NodePropertyData table16[] = {{"attribute_type", "Attribute Type", NodePropertyKind::Text, 0.0f, 0, "GEOMETRY", {}, table15, 1.0f, 0.0f, true, true},
                                        {"attribute_name", "Attribute Name", NodePropertyKind::Text, 0.0f, 0, "", {}, {}, 1.0f, 0.0f, true, true}};
constexpr NodeMuting table17[] = {{0x0, -1}, {0x0, -1}, {0x0, -1}, {0x0, -1}};
constexpr NodeSocketData table18[] = {
    {"Color", "Color", NodeSocketType::Color, 4, true, false, false, {0.800000012f, 0.800000012f, 0.800000012f, 1.0f}, "", 1.0f, 0.0f, ""},
    {"Strength", "Strength", NodeSocketType::Float, 1, true, false, false, {1.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1000000.0f, "NONE"},
    {"Weight", "Weight", NodeSocketType::Float, 1, false, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"}};
constexpr NodeSocketData table19[] = {
    {"Background", "Background", NodeSocketType::Shader, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", 1.0f, 0.0f, ""}};
constexpr NodeTreeKind table20[] = {NodeTreeKind::World};
constexpr NodeMuting table21[] = {{0x0, -1}};
constexpr NodeSocketData table22[] = {
    {"Radius", "Radius", NodeSocketType::Float, 1, true, false, false, {0.0500000007f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1000.0f, "NONE"},
    {"Normal", "Normal", NodeSocketType::Vector, 3, true, false, true, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"}};
constexpr NodeSocketData table23[] = {
    {"Normal", "Normal", NodeSocketType::Vector, 3, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"}};
constexpr NodePropertyData table24[] = {{"samples", "Samples", NodePropertyKind::Integer, 0.0f, 4, "", {}, {}, 2.0f, 16.0f, true, true}};
constexpr NodeMuting table25[] = {{0x2, 1}};
constexpr NodeSocketData table26[] = {
    {"Temperature", "Temperature", NodeSocketType::Float, 1, true, false, false, {6500.0f, 0.0f, 0.0f, 0.0f}, "", 800.0f, 12000.0f, "COLOR_TEMPERATURE"}};
constexpr NodeSocketData table27[] = {
    {"Color", "Color", NodeSocketType::Color, 4, true, false, false, {0.800000012f, 0.800000012f, 0.800000012f, 1.0f}, "", 1.0f, 0.0f, ""}};
constexpr NodeMuting table28[] = {{0x1, 0}};
constexpr NodeSocketData table29[] = {
    {"Color", "Color", NodeSocketType::Color, 4, true, false, false, {1.0f, 1.0f, 1.0f, 1.0f}, "", 1.0f, 0.0f, ""},
    {"Bright", "Brightness", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -100.0f, 100.0f, "NONE"},
    {"Contrast", "Contrast", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -100.0f, 100.0f, "NONE"}};
constexpr NodeMuting table30[] = {{0x7, 0}};
constexpr NodeSocketData table31[] = {
    {"Color", "Color", NodeSocketType::Color, 4, true, false, false, {0.800000012f, 0.800000012f, 0.800000012f, 1.0f}, "", 1.0f, 0.0f, ""},
    {"Roughness", "Roughness", NodeSocketType::Float, 1, true, false, false, {0.5f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
    {"Anisotropy", "Anisotropy", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -1.0f, 1.0f, "NONE"},
    {"Rotation", "Rotation", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
    {"Normal", "Normal", NodeSocketType::Vector, 3, true, false, true, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Tangent", "Tangent", NodeSocketType::Vector, 3, true, false, true, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Weight", "Weight", NodeSocketType::Float, 1, false, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"}};
constexpr NodeSocketData table32[] = {{"BSDF", "BSDF", NodeSocketType::Shader, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", 1.0f, 0.0f, ""}};
constexpr NodeEnumData table33[] = {{"BECKMANN", "Beckmann"}, {"GGX", "GGX"}, {"ASHIKHMIN_SHIRLEY", "Ashikhmin-Shirley"}, {"MULTI_GGX", "Multiscatter GGX"}};
constexpr NodePropertyData table34[] = {{"distribution", "Distribution", NodePropertyKind::Text, 0.0f, 0, "MULTI_GGX", {}, table33, 1.0f, 0.0f, true, true}};
constexpr NodeSocketData table35[] = {
    {"Color", "Color", NodeSocketType::Color, 4, true, false, false, {0.800000012f, 0.800000012f, 0.800000012f, 1.0f}, "", 1.0f, 0.0f, ""},
    {"Roughness", "Roughness", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
    {"Normal", "Normal", NodeSocketType::Vector, 3, true, false, true, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Weight", "Weight", NodeSocketType::Float, 1, false, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"}};
constexpr NodeSocketData table36[] = {
    {"Color", "Color", NodeSocketType::Color, 4, true, false, false, {1.0f, 1.0f, 1.0f, 1.0f}, "", 1.0f, 0.0f, ""},
    {"Roughness", "Roughness", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
    {"IOR", "IOR", NodeSocketType::Float, 1, true, false, false, {1.5f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1000.0f, "NONE"},
    {"Normal", "Normal", NodeSocketType::Vector, 3, true, false, true, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Weight", "Weight", NodeSocketType::Float, 1, false, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Thin Film Thickness", "Thin Film Thickness", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 100000.0f, "WAVELENGTH"},
    {"Thin Film IOR", "Thin Film IOR", NodeSocketType::Float, 1, true, false, false, {1.33000004f, 0.0f, 0.0f, 0.0f}, "", 1.0f, 1000.0f, "NONE"}};
constexpr NodeEnumData table37[] = {{"BECKMANN", "Beckmann"}, {"GGX", "GGX"}, {"MULTI_GGX", "Multiscatter GGX"}};
constexpr NodePropertyData table38[] = {{"distribution", "Distribution", NodePropertyKind::Text, 0.0f, 0, "MULTI_GGX", {}, table37, 1.0f, 0.0f, true, true}};
constexpr NodeSocketData table39[] = {
    {"Color", "Color", NodeSocketType::Color, 4, true, false, false, {0.800000012f, 0.800000012f, 0.800000012f, 1.0f}, "", 1.0f, 0.0f, ""},
    {"Offset", "Offset", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -1.57079637f, 1.57079637f, "ANGLE"},
    {"RoughnessU", "RoughnessU", NodeSocketType::Float, 1, true, false, false, {0.100000001f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
    {"RoughnessV", "RoughnessV", NodeSocketType::Float, 1, true, false, false, {1.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
    {"Tangent", "Tangent", NodeSocketType::Vector, 3, true, false, true, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Weight", "Weight", NodeSocketType::Float, 1, false, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"}};
constexpr NodeEnumData table40[] = {{"Reflection", "Reflection"}, {"Transmission", "Transmission"}};
constexpr NodePropertyData table41[] = {{"component", "Component", NodePropertyKind::Text, 0.0f, 0, "Reflection", {}, table40, 1.0f, 0.0f, true, true}};
constexpr NodeSocketData table42[] = {
    {"Color", "Color", NodeSocketType::Color, 4, true, false, false, {0.0175129995f, 0.00576299988f, 0.00205900008f, 1.0f}, "", 1.0f, 0.0f, ""},
    {"Melanin", "Melanin", NodeSocketType::Float, 1, false, false, false, {0.800000012f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
    {"Melanin Redness", "Melanin Redness", NodeSocketType::Float, 1, false, false, false, {1.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
    {"Tint", "Tint", NodeSocketType::Color, 4, false, false, false, {1.0f, 1.0f, 1.0f, 1.0f}, "", 1.0f, 0.0f, ""},
    {"Absorption Coefficient",
     "Absorption Coefficient",
     NodeSocketType::Vector,
     3,
     false,
     false,
     false,
     {0.245530993f, 0.519999981f, 1.36500001f, 0.0f},
     "",
     0.0f,
     1000.0f,
     "NONE"},
    {"Aspect Ratio", "Aspect Ratio", NodeSocketType::Float, 1, false, false, false, {0.850000024f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
    {"Roughness", "Roughness", NodeSocketType::Float, 1, true, false, false, {0.300000012f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
    {"Radial Roughness", "Radial Roughness", NodeSocketType::Float, 1, true, false, false, {0.300000012f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
    {"Coat", "Coat", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
    {"IOR", "IOR", NodeSocketType::Float, 1, true, false, false, {1.54999995f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1000.0f, "NONE"},
    {"Offset", "Offset", NodeSocketType::Float, 1, true, false, false, {0.0349065848f, 0.0f, 0.0f, 0.0f}, "", -1.57079637f, 1.57079637f, "ANGLE"},
    {"Random Color", "Random Color", NodeSocketType::Float, 1, false, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
    {"Random Roughness", "Random Roughness", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
    {"Random", "Random", NodeSocketType::Float, 1, true, false, true, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Weight", "Weight", NodeSocketType::Float, 1, false, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"R lobe", "Reflection", NodeSocketType::Float, 1, false, false, false, {1.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
    {"TT lobe", "Transmission", NodeSocketType::Float, 1, false, false, false, {1.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
    {"TRT lobe", "Secondary Reflection", NodeSocketType::Float, 1, false, false, false, {1.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"}};
constexpr NodeEnumData table43[] = {{"CHIANG", "Chiang"}, {"HUANG", "Huang"}};
constexpr NodeEnumData table44[] = {{"ABSORPTION", "Absorption Coefficient"}, {"MELANIN", "Melanin Concentration"}, {"COLOR", "Direct Coloring"}};
constexpr NodePropertyData table45[] = {
    {"model", "Scattering model", NodePropertyKind::Text, 0.0f, 0, "CHIANG", {}, table43, 1.0f, 0.0f, true, true},
    {"parametrization", "Color Parametrization", NodePropertyKind::Text, 0.0f, 0, "COLOR", {}, table44, 1.0f, 0.0f, true, true}};
constexpr std::string_view table46[] = {"model", "parametrization"};
constexpr std::uint8_t table47[] = {0, 1, 2, 3, 4, 5};
constexpr NodeSocketState table48[] = {{0x37d0, 0x1, 0x2000, {}},  {0x3fce, 0x1, 0x2000, {}},  {0x37c1, 0x1, 0x2000, {}},
                                       {0x3b670, 0x1, 0x2000, {}}, {0x3be6e, 0x1, 0x2000, {}}, {0x3b661, 0x1, 0x2000, {}}};
constexpr NodeSocketData table49[] = {
    {"Base Color", "Base Color", NodeSocketType::Color, 4, true, false, false, {0.616999984f, 0.577000022f, 0.540000021f, 1.0f}, "", 1.0f, 0.0f, ""},
    {"Edge Tint", "Edge Tint", NodeSocketType::Color, 4, true, false, false, {0.694999993f, 0.726000011f, 0.769999981f, 1.0f}, "", 1.0f, 0.0f, ""},
    {"IOR", "IOR", NodeSocketType::Vector, 3, false, false, false, {2.75699997f, 2.51300001f, 2.23099995f, 0.0f}, "", 0.0f, 100.0f, "NONE"},
    {"Extinction", "Extinction", NodeSocketType::Vector, 3, false, false, false, {3.8670001f, 3.40400004f, 3.00900006f, 0.0f}, "", 0.0f, 100.0f, "NONE"},
    {"Roughness", "Roughness", NodeSocketType::Float, 1, true, false, false, {0.5f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
    {"Anisotropy", "Anisotropy", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
    {"Rotation", "Rotation", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
    {"Normal", "Normal", NodeSocketType::Vector, 3, true, false, true, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Tangent", "Tangent", NodeSocketType::Vector, 3, true, false, true, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Weight", "Weight", NodeSocketType::Float, 1, false, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Thin Film Thickness", "Thin Film Thickness", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 100000.0f, "WAVELENGTH"},
    {"Thin Film IOR", "Thin Film IOR", NodeSocketType::Float, 1, true, false, false, {1.33000004f, 0.0f, 0.0f, 0.0f}, "", 1.0f, 1000.0f, "NONE"}};
constexpr NodeEnumData table50[] = {{"PHYSICAL_CONDUCTOR", "Physical Conductor"}, {"F82", "F82 Tint"}};
constexpr NodePropertyData table51[] = {{"distribution", "Distribution", NodePropertyKind::Text, 0.0f, 0, "MULTI_GGX", {}, table37, 1.0f, 0.0f, true, true},
                                        {"fresnel_type", "Fresnel Type", NodePropertyKind::Text, 0.0f, 0, "F82", {}, table50, 1.0f, 0.0f, true, true}};
constexpr std::string_view table52[] = {"fresnel_type"};
constexpr std::uint8_t table53[] = {0, 1};
constexpr NodeSocketState table54[] = {{0xdfc, 0x1, 0x180, {}}, {0xdf3, 0x1, 0x180, {}}};
constexpr NodeSocketData table55[] = {
    {"Base Color", "Base Color", NodeSocketType::Color, 4, true, false, false, {0.800000012f, 0.800000012f, 0.800000012f, 1.0f}, "", 1.0f, 0.0f, ""},
    {"Metallic", "Metallic", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
    {"Roughness", "Roughness", NodeSocketType::Float, 1, true, false, false, {0.5f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
    {"IOR", "IOR", NodeSocketType::Float, 1, true, false, false, {1.5f, 0.0f, 0.0f, 0.0f}, "", 1.0f, 1000.0f, "NONE"},
    {"Alpha", "Alpha", NodeSocketType::Float, 1, true, false, false, {1.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
    {"Thin Wall", "Thin Wall", NodeSocketType::Bool, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", 1.0f, 0.0f, ""},
    {"Normal", "Normal", NodeSocketType::Vector, 3, true, false, true, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Weight", "Weight", NodeSocketType::Float, 1, false, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Diffuse Roughness", "Diffuse Roughness", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
    {"Subsurface Weight", "Subsurface Weight", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
    {"Subsurface Radius",
     "Subsurface Radius",
     NodeSocketType::Vector,
     3,
     true,
     false,
     false,
     {1.0f, 0.200000003f, 0.100000001f, 0.0f},
     "",
     0.0f,
     100.0f,
     "NONE"},
    {"Subsurface Scale", "Subsurface Scale", NodeSocketType::Float, 1, true, false, false, {0.00499999989f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 10.0f, "DISTANCE"},
    {"Subsurface IOR",
     "Subsurface IOR",
     NodeSocketType::Float,
     1,
     false,
     false,
     false,
     {1.39999998f, 0.0f, 0.0f, 0.0f},
     "",
     1.00999999f,
     3.79999995f,
     "FACTOR"},
    {"Subsurface Anisotropy", "Subsurface Anisotropy", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -1.0f, 1.0f, "FACTOR"},
    {"Specular IOR Level", "Specular IOR Level", NodeSocketType::Float, 1, true, false, false, {0.5f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
    {"Specular Tint", "Specular Tint", NodeSocketType::Color, 4, true, false, false, {1.0f, 1.0f, 1.0f, 1.0f}, "", 1.0f, 0.0f, ""},
    {"Anisotropic", "Anisotropic", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
    {"Anisotropic Rotation", "Anisotropic Rotation", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
    {"Tangent", "Tangent", NodeSocketType::Vector, 3, true, false, true, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Transmission Weight", "Transmission Weight", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
    {"Coat Weight", "Coat Weight", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
    {"Coat Roughness", "Coat Roughness", NodeSocketType::Float, 1, true, false, false, {0.0299999993f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
    {"Coat IOR", "Coat IOR", NodeSocketType::Float, 1, true, false, false, {1.5f, 0.0f, 0.0f, 0.0f}, "", 1.0f, 4.0f, "NONE"},
    {"Coat Tint", "Coat Tint", NodeSocketType::Color, 4, true, false, false, {1.0f, 1.0f, 1.0f, 1.0f}, "", 1.0f, 0.0f, ""},
    {"Coat Normal", "Coat Normal", NodeSocketType::Vector, 3, true, false, true, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Sheen Weight", "Sheen Weight", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
    {"Sheen Roughness", "Sheen Roughness", NodeSocketType::Float, 1, true, false, false, {0.5f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
    {"Sheen Tint", "Sheen Tint", NodeSocketType::Color, 4, true, false, false, {1.0f, 1.0f, 1.0f, 1.0f}, "", 1.0f, 0.0f, ""},
    {"Emission Color", "Emission Color", NodeSocketType::Color, 4, true, false, false, {1.0f, 1.0f, 1.0f, 1.0f}, "", 1.0f, 0.0f, ""},
    {"Emission Strength", "Emission Strength", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1000000.0f, "NONE"},
    {"Thin Film Thickness", "Thin Film Thickness", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 100000.0f, "WAVELENGTH"},
    {"Thin Film IOR", "Thin Film IOR", NodeSocketType::Float, 1, true, false, false, {1.33000004f, 0.0f, 0.0f, 0.0f}, "", 1.0f, 1000.0f, "NONE"}};
constexpr NodeEnumData table56[] = {{"GGX", "GGX"}, {"MULTI_GGX", "Multiscatter GGX"}};
constexpr NodeEnumData table57[] = {{"BURLEY", "Christensen-Burley"},
                                    {"RANDOM_WALK", "Random Walk"},
                                    {"RANDOM_WALK_SKIN", "Random Walk (Skin)"},
                                    {"RANDOM_WALK_LEGACY", "Random Walk (Legacy)"}};
constexpr NodePropertyData table58[] = {
    {"distribution", "Distribution", NodePropertyKind::Text, 0.0f, 0, "MULTI_GGX", {}, table56, 1.0f, 0.0f, true, true},
    {"subsurface_method", "Subsurface Method", NodePropertyKind::Text, 0.0f, 0, "RANDOM_WALK", {}, table57, 1.0f, 0.0f, true, true}};
constexpr std::string_view table59[] = {"subsurface_method"};
constexpr std::uint8_t table60[] = {0, 0, 1, 0};
constexpr NodeSocketState table61[] = {{0xffffef7f, 0x1, 0x1040040, {}}, {0xffffff7f, 0x1, 0x1040040, {}}};
constexpr NodeSocketData table62[] = {
    {"Color", "Color", NodeSocketType::Color, 4, true, false, false, {1.0f, 1.0f, 1.0f, 1.0f}, "", 1.0f, 0.0f, ""},
    {"Position", "Position", NodeSocketType::Vector, 3, true, false, true, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Direction", "Direction", NodeSocketType::Vector, 3, true, false, true, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Weight", "Weight", NodeSocketType::Float, 1, false, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"}};
constexpr NodeSocketData table63[] = {
    {"Color", "Color", NodeSocketType::Color, 4, true, false, false, {1.0f, 1.0f, 1.0f, 1.0f}, "", 1.0f, 0.0f, ""},
    {"Roughness", "Roughness", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
    {"IOR", "IOR", NodeSocketType::Float, 1, true, false, false, {1.45000005f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1000.0f, "NONE"},
    {"Normal", "Normal", NodeSocketType::Vector, 3, true, false, true, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Weight", "Weight", NodeSocketType::Float, 1, false, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"}};
constexpr NodeEnumData table64[] = {{"BECKMANN", "Beckmann"}, {"GGX", "GGX"}};
constexpr NodePropertyData table65[] = {{"distribution", "Distribution", NodePropertyKind::Text, 0.0f, 0, "BECKMANN", {}, table64, 1.0f, 0.0f, true, true}};
constexpr NodeSocketData table66[] = {
    {"Color", "Color", NodeSocketType::Color, 4, true, false, false, {0.800000012f, 0.800000012f, 0.800000012f, 1.0f}, "", 1.0f, 0.0f, ""},
    {"Roughness", "Roughness", NodeSocketType::Float, 1, true, false, false, {0.5f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
    {"Normal", "Normal", NodeSocketType::Vector, 3, true, false, true, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Weight", "Weight", NodeSocketType::Float, 1, false, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"}};
constexpr NodeEnumData table67[] = {{"ASHIKHMIN", "Ashikhmin"}, {"MICROFIBER", "Microfiber"}};
constexpr NodePropertyData table68[] = {{"distribution", "Distribution", NodePropertyKind::Text, 0.0f, 0, "MICROFIBER", {}, table67, 1.0f, 0.0f, true, true}};
constexpr NodeSocketData table69[] = {
    {"Color", "Color", NodeSocketType::Color, 4, true, false, false, {0.800000012f, 0.800000012f, 0.800000012f, 1.0f}, "", 1.0f, 0.0f, ""},
    {"Size", "Size", NodeSocketType::Float, 1, true, false, false, {0.5f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
    {"Smooth", "Smooth", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
    {"Normal", "Normal", NodeSocketType::Vector, 3, true, false, true, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Weight", "Weight", NodeSocketType::Float, 1, false, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"}};
constexpr NodeEnumData table70[] = {{"DIFFUSE", "Diffuse"}, {"GLOSSY", "Glossy"}};
constexpr NodePropertyData table71[] = {{"component", "Component", NodePropertyKind::Text, 0.0f, 0, "DIFFUSE", {}, table70, 1.0f, 0.0f, true, true}};
constexpr NodeSocketData table72[] = {
    {"Color", "Color", NodeSocketType::Color, 4, true, false, false, {0.800000012f, 0.800000012f, 0.800000012f, 1.0f}, "", 1.0f, 0.0f, ""},
    {"Normal", "Normal", NodeSocketType::Vector, 3, true, false, true, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Weight", "Weight", NodeSocketType::Float, 1, false, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"}};
constexpr NodeSocketData table73[] = {
    {"Color", "Color", NodeSocketType::Color, 4, true, false, false, {1.0f, 1.0f, 1.0f, 1.0f}, "", 1.0f, 0.0f, ""},
    {"Weight", "Weight", NodeSocketType::Float, 1, false, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"}};
constexpr NodeSocketData table74[] = {
    {"Strength", "Strength", NodeSocketType::Float, 1, true, false, false, {1.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
    {"Distance", "Distance", NodeSocketType::Float, 1, true, false, false, {0.00100000005f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1000.0f, "NONE"},
    {"Filter Width", "Filter Width", NodeSocketType::Float, 1, true, false, false, {0.100000001f, 0.0f, 0.0f, 0.0f}, "", 0.00100000005f, 10.0f, "PIXEL"},
    {"Height", "Height", NodeSocketType::Float, 1, true, false, true, {1.0f, 0.0f, 0.0f, 0.0f}, "", -1000.0f, 1000.0f, "NONE"},
    {"Normal", "Normal", NodeSocketType::Vector, 3, true, false, true, {0.0f, 0.0f, 0.0f, 0.0f}, "", -1.0f, 1.0f, "NONE"}};
constexpr NodePropertyData table75[] = {{"invert", "Invert", NodePropertyKind::Boolean, 0.0f, 0, "", {}, {}, 1.0f, 0.0f, true, true}};
constexpr NodeMuting table76[] = {{0x10, 4}};
constexpr NodeSocketData table77[] = {
    {"View Vector", "View Vector", NodeSocketType::Vector, 3, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"View Z Depth", "View Z Depth", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"View Distance", "View Distance", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"}};
constexpr NodeMuting table78[] = {{0x0, -1}, {0x0, -1}, {0x0, -1}};
constexpr NodeSocketData table79[] = {
    {"Value", "Value", NodeSocketType::Float, 1, true, false, false, {1.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Min", "Min", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -10000.0f, 10000.0f, "NONE"},
    {"Max", "Max", NodeSocketType::Float, 1, true, false, false, {1.0f, 0.0f, 0.0f, 0.0f}, "", -10000.0f, 10000.0f, "NONE"}};
constexpr NodeSocketData table80[] = {
    {"Result", "Result", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"}};
constexpr NodeEnumData table81[] = {{"MINMAX", "Min Max"}, {"RANGE", "Range"}};
constexpr NodePropertyData table82[] = {{"clamp_type", "Clamp Type", NodePropertyKind::Text, 0.0f, 0, "MINMAX", {}, table81, 1.0f, 0.0f, true, true}};
constexpr NodeSocketData table83[] = {{"Red", "Red", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
                                      {"Green", "Green", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
                                      {"Blue", "Blue", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"}};
constexpr NodeEnumData table84[] = {{"RGB", "RGB"}, {"HSV", "HSV"}, {"HSL", "HSL"}};
constexpr NodePropertyData table85[] = {{"mode", "Mode", NodePropertyKind::Text, 0.0f, 0, "RGB", {}, table84, 1.0f, 0.0f, true, true}};
constexpr NodeSocketData table86[] = {{"X", "X", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -10000.0f, 10000.0f, "NONE"},
                                      {"Y", "Y", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -10000.0f, 10000.0f, "NONE"},
                                      {"Z", "Z", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -10000.0f, 10000.0f, "NONE"}};
constexpr NodeSocketData table87[] = {
    {"Vector", "Vector", NodeSocketType::Vector, 3, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"}};
constexpr NodeSocketData table88[] = {
    {"Height", "Height", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1000.0f, "NONE"},
    {"Midlevel", "Midlevel", NodeSocketType::Float, 1, true, false, false, {0.5f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1000.0f, "NONE"},
    {"Scale", "Scale", NodeSocketType::Float, 1, true, false, false, {0.00999999978f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1000.0f, "NONE"},
    {"Normal", "Normal", NodeSocketType::Vector, 3, true, false, true, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"}};
constexpr NodeSocketData table89[] = {
    {"Displacement", "Displacement", NodeSocketType::Vector, 3, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"}};
constexpr NodeEnumData table90[] = {{"OBJECT", "Object Space"}, {"WORLD", "World Space"}};
constexpr NodePropertyData table91[] = {{"space", "Space", NodePropertyKind::Text, 0.0f, 0, "OBJECT", {}, table90, 1.0f, 0.0f, true, true}};
constexpr NodeMuting table92[] = {{0x8, 3}};
constexpr NodeSocketData table93[] = {
    {"Color", "Color", NodeSocketType::Color, 4, true, false, false, {1.0f, 1.0f, 1.0f, 1.0f}, "", 1.0f, 0.0f, ""},
    {"Strength", "Strength", NodeSocketType::Float, 1, true, false, false, {1.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1000000.0f, "NONE"},
    {"Weight", "Weight", NodeSocketType::Float, 1, false, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"}};
constexpr NodeSocketData table94[] = {{"Emission", "Emission", NodeSocketType::Shader, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", 1.0f, 0.0f, ""}};
constexpr NodeSocketData table95[] = {
    {"Factor", "Factor", NodeSocketType::Float, 1, true, false, false, {1.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
    {"Value", "Value", NodeSocketType::Float, 1, true, false, false, {1.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"}};
constexpr NodeSocketData table96[] = {
    {"Value", "Value", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"}};
constexpr NodeEnumData table97[] = {{"HORIZONTAL", "Horizontal"}, {"EXTRAPOLATED", "Extrapolated"}};
constexpr float table98[] = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f};
constexpr float table99[] = {0.0f, 0.0f, 1.0f, 1.0f};
constexpr float table100[] = {0.0f, 0.0f};
constexpr NodePropertyData table101[] = {{"mapping.curve_count", "Curves", NodePropertyKind::Integer, 0.0f, 1, "", {}, {}, 1.0f, 0.0f, false, true},
                                         {"mapping.extend", "Extend", NodePropertyKind::Text, 0.0f, 0, "EXTRAPOLATED", {}, table97, 1.0f, 0.0f, true, true},
                                         {"mapping.clip", "Clip", NodePropertyKind::Numbers, 0.0f, 0, "", table98, {}, 1.0f, 0.0f, false, true},
                                         {"mapping.curve0", "Value", NodePropertyKind::Numbers, 0.0f, 0, "", table99, {}, 1.0f, 0.0f, true, true},
                                         {"mapping.handles0", "Value Handles", NodePropertyKind::Numbers, 0.0f, 0, "", table100, {}, 1.0f, 0.0f, false, true}};
constexpr NodeSocketData table102[] = {
    {"IOR", "IOR", NodeSocketType::Float, 1, true, false, false, {1.5f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1000.0f, "NONE"},
    {"Normal", "Normal", NodeSocketType::Vector, 3, true, false, true, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"}};
constexpr NodeSocketData table103[] = {
    {"Fac", "Factor", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"}};
constexpr NodeSocketData table104[] = {
    {"Color", "Color", NodeSocketType::Color, 4, true, false, true, {1.0f, 1.0f, 1.0f, 1.0f}, "", 1.0f, 0.0f, ""},
    {"Gamma", "Gamma", NodeSocketType::Float, 1, true, false, false, {1.0f, 0.0f, 0.0f, 0.0f}, "", 0.00100000005f, 10.0f, "NONE"}};
constexpr NodeSocketData table105[] = {
    {"Is Strand", "Is Strand", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Intercept", "Intercept", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Length", "Length", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Thickness", "Thickness", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Tangent Normal",
     "Tangent Normal",
     NodeSocketType::Vector,
     3,
     true,
     false,
     false,
     {0.0f, 0.0f, 0.0f, 0.0f},
     "",
     -3.40282347e+38f,
     3.40282347e+38f,
     "NONE"},
    {"Random", "Random", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"}};
constexpr NodeMuting table106[] = {{0x0, -1}, {0x0, -1}, {0x0, -1}, {0x0, -1}, {0x0, -1}, {0x0, -1}};
constexpr NodeSocketData table107[] = {
    {"Weight", "Weight", NodeSocketType::Float, 1, false, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"}};
constexpr NodeSocketData table108[] = {{"Holdout", "Holdout", NodeSocketType::Shader, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", 1.0f, 0.0f, ""}};
constexpr NodeSocketData table109[] = {
    {"Hue", "Hue", NodeSocketType::Float, 1, true, false, false, {0.5f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "NONE"},
    {"Saturation", "Saturation", NodeSocketType::Float, 1, true, false, false, {1.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 2.0f, "NONE"},
    {"Value", "Value", NodeSocketType::Float, 1, true, false, false, {1.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 2.0f, "NONE"},
    {"Fac", "Factor", NodeSocketType::Float, 1, true, false, false, {1.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
    {"Color", "Color", NodeSocketType::Color, 4, true, false, false, {0.800000012f, 0.800000012f, 0.800000012f, 1.0f}, "", 1.0f, 0.0f, ""}};
constexpr NodeSocketData table110[] = {{"Fac", "Factor", NodeSocketType::Float, 1, true, false, false, {1.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
                                       {"Color", "Color", NodeSocketType::Color, 4, true, false, false, {0.0f, 0.0f, 0.0f, 1.0f}, "", 1.0f, 0.0f, ""}};
constexpr NodeSocketData table111[] = {
    {"Blend", "Blend", NodeSocketType::Float, 1, true, false, false, {0.5f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "NONE"},
    {"Normal", "Normal", NodeSocketType::Vector, 3, true, false, true, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"}};
constexpr NodeSocketData table112[] = {
    {"Fresnel", "Fresnel", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Facing", "Facing", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"}};
constexpr NodeMuting table113[] = {{0x3, 0}, {0x3, 0}};
constexpr NodeSocketData table114[] = {
    {"Strength", "Strength", NodeSocketType::Float, 1, true, false, false, {100.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1000000.0f, "NONE"},
    {"Smooth", "Smooth", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1000.0f, "NONE"}};
constexpr NodeSocketData table115[] = {
    {"Quadratic", "Quadratic", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Linear", "Linear", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Constant", "Constant", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"}};
constexpr NodeMuting table116[] = {{0x3, 0}, {0x3, 0}, {0x3, 0}};
constexpr NodeSocketData table117[] = {
    {"Is Camera Ray", "Is Camera Ray", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Is Shadow Ray", "Is Shadow Ray", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Is Diffuse Ray", "Is Diffuse Ray", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Is Glossy Ray", "Is Glossy Ray", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Is Singular Ray",
     "Is Singular Ray",
     NodeSocketType::Float,
     1,
     true,
     false,
     false,
     {0.0f, 0.0f, 0.0f, 0.0f},
     "",
     -3.40282347e+38f,
     3.40282347e+38f,
     "NONE"},
    {"Is Reflection Ray",
     "Is Reflection Ray",
     NodeSocketType::Float,
     1,
     true,
     false,
     false,
     {0.0f, 0.0f, 0.0f, 0.0f},
     "",
     -3.40282347e+38f,
     3.40282347e+38f,
     "NONE"},
    {"Is Transmission Ray",
     "Is Transmission Ray",
     NodeSocketType::Float,
     1,
     true,
     false,
     false,
     {0.0f, 0.0f, 0.0f, 0.0f},
     "",
     -3.40282347e+38f,
     3.40282347e+38f,
     "NONE"},
    {"Is Volume Scatter Ray",
     "Is Volume Scatter Ray",
     NodeSocketType::Float,
     1,
     true,
     false,
     false,
     {0.0f, 0.0f, 0.0f, 0.0f},
     "",
     -3.40282347e+38f,
     3.40282347e+38f,
     "NONE"},
    {"Ray Length", "Ray Length", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Ray Depth", "Ray Depth", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Diffuse Depth", "Diffuse Depth", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Glossy Depth", "Glossy Depth", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Transparent Depth",
     "Transparent Depth",
     NodeSocketType::Float,
     1,
     true,
     false,
     false,
     {0.0f, 0.0f, 0.0f, 0.0f},
     "",
     -3.40282347e+38f,
     3.40282347e+38f,
     "NONE"},
    {"Transmission Depth",
     "Transmission Depth",
     NodeSocketType::Float,
     1,
     true,
     false,
     false,
     {0.0f, 0.0f, 0.0f, 0.0f},
     "",
     -3.40282347e+38f,
     3.40282347e+38f,
     "NONE"},
    {"Portal Depth", "Portal Depth", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"}};
constexpr NodeMuting table118[] = {{0x0, -1}, {0x0, -1}, {0x0, -1}, {0x0, -1}, {0x0, -1}, {0x0, -1}, {0x0, -1}, {0x0, -1},
                                   {0x0, -1}, {0x0, -1}, {0x0, -1}, {0x0, -1}, {0x0, -1}, {0x0, -1}, {0x0, -1}};
constexpr NodeSocketData table119[] = {
    {"Value", "Value", NodeSocketType::Float, 1, true, false, false, {1.0f, 0.0f, 0.0f, 0.0f}, "", -10000.0f, 10000.0f, "NONE"},
    {"From Min", "From Min", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -10000.0f, 10000.0f, "NONE"},
    {"From Max", "From Max", NodeSocketType::Float, 1, true, false, false, {1.0f, 0.0f, 0.0f, 0.0f}, "", -10000.0f, 10000.0f, "NONE"},
    {"To Min", "To Min", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -10000.0f, 10000.0f, "NONE"},
    {"To Max", "To Max", NodeSocketType::Float, 1, true, false, false, {1.0f, 0.0f, 0.0f, 0.0f}, "", -10000.0f, 10000.0f, "NONE"},
    {"Steps", "Steps", NodeSocketType::Float, 1, false, false, false, {4.0f, 0.0f, 0.0f, 0.0f}, "", -10000.0f, 10000.0f, "NONE"},
    {"Vector", "Vector", NodeSocketType::Vector, 3, false, false, true, {0.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "NONE"},
    {"From_Min_FLOAT3", "From Min", NodeSocketType::Vector, 3, false, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"From_Max_FLOAT3", "From Max", NodeSocketType::Vector, 3, false, false, false, {1.0f, 1.0f, 1.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"To_Min_FLOAT3", "To Min", NodeSocketType::Vector, 3, false, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"To_Max_FLOAT3", "To Max", NodeSocketType::Vector, 3, false, false, false, {1.0f, 1.0f, 1.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Steps_FLOAT3", "Steps", NodeSocketType::Vector, 3, false, false, false, {4.0f, 4.0f, 4.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"}};
constexpr NodeSocketData table120[] = {
    {"Result", "Result", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Vector", "Vector", NodeSocketType::Vector, 3, false, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"}};
constexpr NodeEnumData table121[] = {{"LINEAR", "Linear"}, {"STEPPED", "Stepped Linear"}, {"SMOOTHSTEP", "Smooth Step"}, {"SMOOTHERSTEP", "Smoother Step"}};
constexpr NodeEnumData table122[] = {{"FLOAT", "Float"}, {"FLOAT_VECTOR", "Vector"}};
constexpr NodePropertyData table123[] = {
    {"clamp", "Clamp", NodePropertyKind::Boolean, 0.0f, 1, "", {}, {}, 1.0f, 0.0f, true, true},
    {"interpolation_type", "Interpolation Type", NodePropertyKind::Text, 0.0f, 0, "LINEAR", {}, table121, 1.0f, 0.0f, true, true},
    {"data_type", "Data Type", NodePropertyKind::Text, 0.0f, 0, "FLOAT", {}, table122, 1.0f, 0.0f, true, true}};
constexpr NodeMuting table124[] = {{0x3f, 0}, {0xfc0, 6}};
constexpr std::string_view table125[] = {"interpolation_type", "data_type"};
constexpr std::uint8_t table126[] = {0, 1, 2, 3, 0, 1, 0, 1};
constexpr NodeSocketState table127[] = {{0x1f, 0x1, 0x40, {}}, {0x7c0, 0x2, 0x40, {}}, {0x3f, 0x1, 0x40, {}}, {0xfc0, 0x2, 0x40, {}}};
constexpr NodeSocketData table128[] = {
    {"Vector", "Vector", NodeSocketType::Vector, 3, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Location", "Location", NodeSocketType::Vector, 3, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "TRANSLATION"},
    {"Rotation", "Rotation", NodeSocketType::Vector, 3, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "EULER"},
    {"Scale", "Scale", NodeSocketType::Vector, 3, true, false, false, {1.0f, 1.0f, 1.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "XYZ"}};
constexpr NodeEnumData table129[] = {{"POINT", "Point"}, {"TEXTURE", "Texture"}, {"VECTOR", "Vector"}, {"NORMAL", "Normal"}};
constexpr NodePropertyData table130[] = {{"vector_type", "Type", NodePropertyKind::Text, 0.0f, 0, "POINT", {}, table129, 1.0f, 0.0f, true, true}};
constexpr NodeMuting table131[] = {{0xf, 0}};
constexpr std::string_view table132[] = {"vector_type"};
constexpr std::uint8_t table133[] = {0, 0, 1, 1};
constexpr NodeSocketState table134[] = {{0xf, 0x1, 0x0, {}}, {0xd, 0x1, 0x0, {}}};
constexpr NodeSocketData table135[] = {
    {"Value", "Value", NodeSocketType::Float, 1, true, false, false, {0.5f, 0.0f, 0.0f, 0.0f}, "", -10000.0f, 10000.0f, "NONE"},
    {"Value_001", "Value", NodeSocketType::Float, 1, true, false, false, {0.5f, 0.0f, 0.0f, 0.0f}, "", -10000.0f, 10000.0f, "NONE"},
    {"Value_002", "Value", NodeSocketType::Float, 1, false, false, false, {0.5f, 0.0f, 0.0f, 0.0f}, "", -10000.0f, 10000.0f, "NONE"}};
constexpr NodeEnumData table136[] = {{"ADD", "Add"},
                                     {"SUBTRACT", "Subtract"},
                                     {"MULTIPLY", "Multiply"},
                                     {"DIVIDE", "Divide"},
                                     {"MULTIPLY_ADD", "Multiply Add"},
                                     {"POWER", "Power"},
                                     {"LOGARITHM", "Logarithm"},
                                     {"SQRT", "Square Root"},
                                     {"INVERSE_SQRT", "Inverse Square Root"},
                                     {"ABSOLUTE", "Absolute"},
                                     {"EXPONENT", "Exponent"},
                                     {"MINIMUM", "Minimum"},
                                     {"MAXIMUM", "Maximum"},
                                     {"LESS_THAN", "Less Than"},
                                     {"GREATER_THAN", "Greater Than"},
                                     {"SIGN", "Sign"},
                                     {"COMPARE", "Compare"},
                                     {"SMOOTH_MIN", "Smooth Minimum"},
                                     {"SMOOTH_MAX", "Smooth Maximum"},
                                     {"ROUND", "Round"},
                                     {"FLOOR", "Floor"},
                                     {"CEIL", "Ceil"},
                                     {"TRUNC", "Truncate"},
                                     {"FRACT", "Fraction"},
                                     {"MODULO", "Truncated Modulo"},
                                     {"FLOORED_MODULO", "Floored Modulo"},
                                     {"WRAP", "Wrap"},
                                     {"SNAP", "Snap"},
                                     {"PINGPONG", "Ping-Pong"},
                                     {"SINE", "Sine"},
                                     {"COSINE", "Cosine"},
                                     {"TANGENT", "Tangent"},
                                     {"ARCSINE", "Arcsine"},
                                     {"ARCCOSINE", "Arccosine"},
                                     {"ARCTANGENT", "Arctangent"},
                                     {"ARCTAN2", "Arctan2"},
                                     {"SINH", "Hyperbolic Sine"},
                                     {"COSH", "Hyperbolic Cosine"},
                                     {"TANH", "Hyperbolic Tangent"},
                                     {"RADIANS", "To Radians"},
                                     {"DEGREES", "To Degrees"}};
constexpr NodePropertyData table137[] = {{"operation", "Operation", NodePropertyKind::Text, 0.0f, 0, "ADD", {}, table136, 1.0f, 0.0f, true, true},
                                         {"use_clamp", "Clamp", NodePropertyKind::Boolean, 0.0f, 0, "", {}, {}, 1.0f, 0.0f, true, true}};
constexpr std::string_view table138[] = {"operation"};
constexpr std::uint8_t table139[] = {0, 0, 0, 0, 1, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 2, 1, 1, 1, 2, 2, 2, 2, 2, 0, 0, 1, 0, 0, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2};
constexpr NodeSocketState table140[] = {{0x3, 0x1, 0x0, {}}, {0x7, 0x1, 0x0, {}}, {0x1, 0x1, 0x0, {}}};
constexpr NodeSocketData table141[] = {
    {"Factor_Float", "Factor", NodeSocketType::Float, 1, true, false, false, {1.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
    {"Factor_Vector", "Factor", NodeSocketType::Vector, 3, false, false, false, {0.5f, 0.5f, 0.5f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
    {"A_Float", "A", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -10000.0f, 10000.0f, "NONE"},
    {"B_Float", "B", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -10000.0f, 10000.0f, "NONE"},
    {"A_Vector", "A", NodeSocketType::Vector, 3, false, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"B_Vector", "B", NodeSocketType::Vector, 3, false, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"A_Color", "A", NodeSocketType::Color, 4, false, false, false, {0.5f, 0.5f, 0.5f, 1.0f}, "", 1.0f, 0.0f, ""},
    {"B_Color", "B", NodeSocketType::Color, 4, false, false, false, {0.5f, 0.5f, 0.5f, 1.0f}, "", 1.0f, 0.0f, ""},
    {"A_Rotation", "A", NodeSocketType::Rotation, 3, false, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", 1.0f, 0.0f, ""},
    {"B_Rotation", "B", NodeSocketType::Rotation, 3, false, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", 1.0f, 0.0f, ""}};
constexpr NodeSocketData table142[] = {
    {"Result_Float", "Result", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Result_Vector", "Result", NodeSocketType::Vector, 3, false, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Result_Color", "Result", NodeSocketType::Color, 4, false, false, false, {0.800000012f, 0.800000012f, 0.800000012f, 1.0f}, "", 1.0f, 0.0f, ""},
    {"Result_Rotation", "Result", NodeSocketType::Rotation, 3, false, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", 1.0f, 0.0f, ""}};
constexpr NodeEnumData table143[] = {{"FLOAT", "Float"}, {"VECTOR", "Vector"}, {"RGBA", "Color"}};
constexpr NodeEnumData table144[] = {{"UNIFORM", "Uniform"}, {"NON_UNIFORM", "Non-Uniform"}};
constexpr NodeEnumData table145[] = {{"MIX", "Mix"},
                                     {"DARKEN", "Darken"},
                                     {"MULTIPLY", "Multiply"},
                                     {"BURN", "Color Burn"},
                                     {"LIGHTEN", "Lighten"},
                                     {"SCREEN", "Screen"},
                                     {"DODGE", "Color Dodge"},
                                     {"ADD", "Add"},
                                     {"OVERLAY", "Overlay"},
                                     {"SOFT_LIGHT", "Soft Light"},
                                     {"LINEAR_LIGHT", "Linear Light"},
                                     {"DIFFERENCE", "Difference"},
                                     {"EXCLUSION", "Exclusion"},
                                     {"SUBTRACT", "Subtract"},
                                     {"DIVIDE", "Divide"},
                                     {"HUE", "Hue"},
                                     {"SATURATION", "Saturation"},
                                     {"COLOR", "Color"},
                                     {"VALUE", "Value"}};
constexpr NodePropertyData table146[] = {{"data_type", "Data Type", NodePropertyKind::Text, 0.0f, 0, "FLOAT", {}, table143, 1.0f, 0.0f, true, true},
                                         {"factor_mode", "Factor Mode", NodePropertyKind::Text, 0.0f, 0, "UNIFORM", {}, table144, 1.0f, 0.0f, true, true},
                                         {"blend_type", "Blending Mode", NodePropertyKind::Text, 0.0f, 0, "MIX", {}, table145, 1.0f, 0.0f, true, true},
                                         {"clamp_factor", "Clamp Factor", NodePropertyKind::Boolean, 0.0f, 1, "", {}, {}, 1.0f, 0.0f, true, true},
                                         {"clamp_result", "Clamp Result", NodePropertyKind::Boolean, 0.0f, 0, "", {}, {}, 1.0f, 0.0f, true, true}};
constexpr NodeMuting table147[] = {{0xc, 2}, {0x30, 4}, {0xc0, 6}, {0x0, -1}};
constexpr std::string_view table148[] = {"data_type", "factor_mode"};
constexpr std::uint8_t table149[] = {0, 0, 1, 2, 3, 3};
constexpr NodeSocketState table150[] = {{0xd, 0x1, 0x0, {}}, {0x31, 0x2, 0x0, {}}, {0x32, 0x2, 0x0, {}}, {0xc1, 0x4, 0x0, {}}};
constexpr NodeSocketData table151[] = {{"Fac", "Factor", NodeSocketType::Float, 1, true, false, false, {0.5f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
                                       {"Color1", "Color1", NodeSocketType::Color, 4, true, false, false, {0.5f, 0.5f, 0.5f, 1.0f}, "", 1.0f, 0.0f, ""},
                                       {"Color2", "Color2", NodeSocketType::Color, 4, true, false, false, {0.5f, 0.5f, 0.5f, 1.0f}, "", 1.0f, 0.0f, ""}};
constexpr NodePropertyData table152[] = {{"blend_type", "Blending Mode", NodePropertyKind::Text, 0.0f, 0, "MIX", {}, table145, 1.0f, 0.0f, true, true},
                                         {"use_alpha", "Alpha", NodePropertyKind::Boolean, 0.0f, 0, "", {}, {}, 1.0f, 0.0f, true, true},
                                         {"use_clamp", "Clamp", NodePropertyKind::Boolean, 0.0f, 0, "", {}, {}, 1.0f, 0.0f, true, true}};
constexpr NodeMuting table153[] = {{0x6, 1}};
constexpr NodeSocketData table154[] = {{"Fac", "Factor", NodeSocketType::Float, 1, true, false, false, {0.5f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
                                       {"Shader", "Shader", NodeSocketType::Shader, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", 1.0f, 0.0f, ""},
                                       {"Shader_001", "Shader", NodeSocketType::Shader, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", 1.0f, 0.0f, ""}};
constexpr NodeSocketData table155[] = {
    {"Position", "Position", NodeSocketType::Vector, 3, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Normal", "Normal", NodeSocketType::Vector, 3, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Tangent", "Tangent", NodeSocketType::Vector, 3, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"True Normal", "True Normal", NodeSocketType::Vector, 3, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Incoming", "Incoming", NodeSocketType::Vector, 3, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Parametric", "Parametric", NodeSocketType::Vector, 3, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Backfacing", "Backfacing", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Pointiness", "Pointiness", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Random Per Island",
     "Random Per Island",
     NodeSocketType::Float,
     1,
     true,
     false,
     false,
     {0.0f, 0.0f, 0.0f, 0.0f},
     "",
     -3.40282347e+38f,
     3.40282347e+38f,
     "NONE"}};
constexpr NodeMuting table156[] = {{0x0, -1}, {0x0, -1}, {0x0, -1}, {0x0, -1}, {0x0, -1}, {0x0, -1}, {0x0, -1}, {0x0, -1}, {0x0, -1}};
constexpr NodeSocketData table157[] = {
    {"Normal", "Normal", NodeSocketType::Vector, 3, true, false, false, {0.0f, 0.0f, 1.0f, 0.0f}, "", -1.0f, 1.0f, "DIRECTION"}};
constexpr NodeSocketData table158[] = {
    {"Normal", "Normal", NodeSocketType::Vector, 3, true, false, false, {0.0f, 0.0f, 1.0f, 0.0f}, "", -1.0f, 1.0f, "DIRECTION"},
    {"Dot", "Dot", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"}};
constexpr NodeMuting table159[] = {{0x1, 0}, {0x1, 0}};
constexpr NodeSocketData table160[] = {
    {"Strength", "Strength", NodeSocketType::Float, 1, true, false, false, {1.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 10.0f, "NONE"},
    {"Color", "Color", NodeSocketType::Color, 4, true, false, false, {0.5f, 0.5f, 1.0f, 1.0f}, "", 1.0f, 0.0f, ""}};
constexpr NodeEnumData table161[] = {{"TANGENT", "Tangent Space"},
                                     {"OBJECT", "Object Space"},
                                     {"WORLD", "World Space"},
                                     {"BLENDER_OBJECT", "Blender Object Space"},
                                     {"BLENDER_WORLD", "Blender World Space"}};
constexpr NodeEnumData table162[] = {{"OPENGL", "OpenGL"}, {"DIRECTX", "DirectX"}};
constexpr NodeEnumData table163[] = {{"ORIGINAL", "Original Base"}, {"DISPLACED", "Displaced Base"}};
constexpr NodePropertyData table164[] = {{"space", "Space", NodePropertyKind::Text, 0.0f, 0, "TANGENT", {}, table161, 1.0f, 0.0f, true, true},
                                         {"uv_map", "UV Map", NodePropertyKind::Text, 0.0f, 0, "", {}, {}, 1.0f, 0.0f, true, true},
                                         {"convention", "Mode", NodePropertyKind::Text, 0.0f, 0, "OPENGL", {}, table162, 1.0f, 0.0f, true, true},
                                         {"base", "Base", NodePropertyKind::Text, 0.0f, 0, "DISPLACED", {}, table163, 1.0f, 0.0f, true, true}};
constexpr NodeSocketData table165[] = {
    {"Location", "Location", NodeSocketType::Vector, 3, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Color", "Color", NodeSocketType::Color, 4, true, false, false, {0.800000012f, 0.800000012f, 0.800000012f, 1.0f}, "", 1.0f, 0.0f, ""},
    {"Alpha", "Alpha", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Object Index", "Object Index", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Material Index", "Material Index", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Random", "Random", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"}};
constexpr NodeTreeKind table166[] = {NodeTreeKind::Material, NodeTreeKind::Light};
constexpr NodeSocketData table167[] = {{"Surface", "Surface", NodeSocketType::Shader, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", 1.0f, 0.0f, ""}};
constexpr NodeEnumData table168[] = {{"ALL", "All"}, {"EEVEE", "EEVEE"}, {"CYCLES", "Cycles"}};
constexpr NodePropertyData table169[] = {{"is_active_output", "Active Output", NodePropertyKind::Boolean, 0.0f, 1, "", {}, {}, 1.0f, 0.0f, true, true},
                                         {"target", "Target", NodePropertyKind::Text, 0.0f, 0, "ALL", {}, table168, 1.0f, 0.0f, true, true}};
constexpr NodeSocketData table170[] = {
    {"Surface", "Surface", NodeSocketType::Shader, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", 1.0f, 0.0f, ""},
    {"Volume", "Volume", NodeSocketType::Shader, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", 1.0f, 0.0f, ""},
    {"Displacement", "Displacement", NodeSocketType::Vector, 3, true, false, true, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Thickness", "Thickness", NodeSocketType::Float, 1, true, false, true, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"}};
constexpr NodeSocketData table171[] = {{"Surface", "Surface", NodeSocketType::Shader, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", 1.0f, 0.0f, ""},
                                       {"Volume", "Volume", NodeSocketType::Shader, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", 1.0f, 0.0f, ""}};
constexpr NodeSocketData table172[] = {
    {"Index", "Index", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Random", "Random", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Age", "Age", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Lifetime", "Lifetime", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Location", "Location", NodeSocketType::Vector, 3, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Size", "Size", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Velocity", "Velocity", NodeSocketType::Vector, 3, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Angular Velocity",
     "Angular Velocity",
     NodeSocketType::Vector,
     3,
     true,
     false,
     false,
     {0.0f, 0.0f, 0.0f, 0.0f},
     "",
     -3.40282347e+38f,
     3.40282347e+38f,
     "NONE"}};
constexpr NodeMuting table173[] = {{0x0, -1}, {0x0, -1}, {0x0, -1}, {0x0, -1}, {0x0, -1}, {0x0, -1}, {0x0, -1}, {0x0, -1}};
constexpr NodeSocketData table174[] = {
    {"Position", "Position", NodeSocketType::Vector, 3, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Radius", "Radius", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Random", "Random", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"}};
constexpr NodeSocketData table175[] = {{"Color", "Color", NodeSocketType::Color, 4, true, false, false, {0.5f, 0.5f, 0.5f, 1.0f}, "", 1.0f, 0.0f, ""}};
constexpr NodeSocketData table176[] = {{"Fac", "Factor", NodeSocketType::Float, 1, true, false, false, {1.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
                                       {"Color", "Color", NodeSocketType::Color, 4, true, false, false, {1.0f, 1.0f, 1.0f, 1.0f}, "", 1.0f, 0.0f, ""}};
constexpr NodePropertyData table177[] = {{"mapping.curve_count", "Curves", NodePropertyKind::Integer, 0.0f, 4, "", {}, {}, 1.0f, 0.0f, false, true},
                                         {"mapping.extend", "Extend", NodePropertyKind::Text, 0.0f, 0, "EXTRAPOLATED", {}, table97, 1.0f, 0.0f, true, true},
                                         {"mapping.clip", "Clip", NodePropertyKind::Numbers, 0.0f, 0, "", table98, {}, 1.0f, 0.0f, false, true},
                                         {"mapping.curve0", "R", NodePropertyKind::Numbers, 0.0f, 0, "", table99, {}, 1.0f, 0.0f, true, true},
                                         {"mapping.handles0", "R Handles", NodePropertyKind::Numbers, 0.0f, 0, "", table100, {}, 1.0f, 0.0f, false, true},
                                         {"mapping.curve1", "G", NodePropertyKind::Numbers, 0.0f, 0, "", table99, {}, 1.0f, 0.0f, true, true},
                                         {"mapping.handles1", "G Handles", NodePropertyKind::Numbers, 0.0f, 0, "", table100, {}, 1.0f, 0.0f, false, true},
                                         {"mapping.curve2", "B", NodePropertyKind::Numbers, 0.0f, 0, "", table99, {}, 1.0f, 0.0f, true, true},
                                         {"mapping.handles2", "B Handles", NodePropertyKind::Numbers, 0.0f, 0, "", table100, {}, 1.0f, 0.0f, false, true},
                                         {"mapping.curve3", "C", NodePropertyKind::Numbers, 0.0f, 0, "", table99, {}, 1.0f, 0.0f, true, true},
                                         {"mapping.handles3", "C Handles", NodePropertyKind::Numbers, 0.0f, 0, "", table100, {}, 1.0f, 0.0f, false, true}};
constexpr NodeSocketData table178[] = {
    {"Val", "Val", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"}};
constexpr NodeSocketData table179[] = {
    {"Vector", "Vector", NodeSocketType::Vector, 2, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Sides", "Sides", NodeSocketType::Float, 1, true, false, false, {5.0f, 0.0f, 0.0f, 0.0f}, "", 2.0f, 1000.0f, "NONE"},
    {"Roundness", "Roundness", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"}};
constexpr NodeSocketData table180[] = {
    {"Segment Coordinates",
     "Segment Coordinates",
     NodeSocketType::Vector,
     3,
     true,
     false,
     false,
     {0.0f, 0.0f, 0.0f, 0.0f},
     "",
     -3.40282347e+38f,
     3.40282347e+38f,
     "NONE"},
    {"Segment ID", "Segment ID", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Segment Width", "Segment Width", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Segment Rotation",
     "Segment Rotation",
     NodeSocketType::Float,
     1,
     true,
     false,
     false,
     {0.0f, 0.0f, 0.0f, 0.0f},
     "",
     -3.40282347e+38f,
     3.40282347e+38f,
     "NONE"}};
constexpr NodePropertyData table181[] = {{"normalize", "Normalize", NodePropertyKind::Boolean, 0.0f, 0, "", {}, {}, 1.0f, 0.0f, true, true}};
constexpr NodeSocketData table182[] = {
    {"Position", "Position", NodeSocketType::Vector, 3, true, false, true, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Direction", "Direction", NodeSocketType::Vector, 3, true, false, true, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Length", "Length", NodeSocketType::Float, 1, true, false, false, {1.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"__extend__", "", NodeSocketType::Custom, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", 1.0f, 0.0f, ""}};
constexpr NodeSocketData table183[] = {
    {"Is Hit", "Is Hit", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Self Hit", "Self Hit", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Hit Distance", "Hit Distance", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Hit Position", "Hit Position", NodeSocketType::Vector, 3, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Hit Normal", "Hit Normal", NodeSocketType::Vector, 3, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"__extend__", "", NodeSocketType::Custom, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", 1.0f, 0.0f, ""}};
constexpr NodePropertyData table184[] = {{"only_local", "Only Local", NodePropertyKind::Boolean, 0.0f, 0, "", {}, {}, 1.0f, 0.0f, true, true},
                                         {"sample_attribute_items", "Items", NodePropertyKind::Integer, 0.0f, 0, "", {}, {}, 1.0f, 0.0f, false, true},
                                         {"active_index", "Active Item Index", NodePropertyKind::Integer, 0.0f, 0, "", {}, {}, 0.0f, 10000.0f, false, true}};
constexpr NodeMuting table185[] = {{0x4, 2}, {0x4, 2}, {0x4, 2}, {0x7, 0}, {0x7, 0}, {0x8, 3}};
constexpr NodeSocketData table186[] = {
    {"Red", "Red", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Green", "Green", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Blue", "Blue", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"}};
constexpr NodeMuting table187[] = {{0x1, 0}, {0x1, 0}, {0x1, 0}};
constexpr NodeSocketData table188[] = {
    {"Vector", "Vector", NodeSocketType::Vector, 3, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -10000.0f, 10000.0f, "NONE"}};
constexpr NodeSocketData table189[] = {
    {"X", "X", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Y", "Y", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Z", "Z", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"}};
constexpr NodeSocketData table190[] = {
    {"Color", "Color", NodeSocketType::Color, 4, true, false, false, {0.800000012f, 0.800000012f, 0.800000012f, 1.0f}, "", 1.0f, 0.0f, ""},
    {"Scale", "Scale", NodeSocketType::Float, 1, true, false, false, {0.00499999989f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1000.0f, "NONE"},
    {"Radius", "Radius", NodeSocketType::Vector, 3, true, false, false, {1.0f, 0.200000003f, 0.100000001f, 0.0f}, "", 0.0f, 100.0f, "NONE"},
    {"IOR", "IOR", NodeSocketType::Float, 1, true, false, false, {1.39999998f, 0.0f, 0.0f, 0.0f}, "", 1.00999999f, 3.79999995f, "FACTOR"},
    {"Roughness", "Roughness", NodeSocketType::Float, 1, true, false, false, {1.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
    {"Anisotropy", "Anisotropy", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -1.0f, 1.0f, "FACTOR"},
    {"Normal", "Normal", NodeSocketType::Vector, 3, true, false, true, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Weight", "Weight", NodeSocketType::Float, 1, false, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"}};
constexpr NodeSocketData table191[] = {{"BSSRDF", "BSSRDF", NodeSocketType::Shader, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", 1.0f, 0.0f, ""}};
constexpr NodePropertyData table192[] = {{"falloff", "Method", NodePropertyKind::Text, 0.0f, 0, "RANDOM_WALK", {}, table57, 1.0f, 0.0f, true, true}};
constexpr std::string_view table193[] = {"falloff"};
constexpr std::uint8_t table194[] = {0, 1, 2, 1};
constexpr NodeSocketState table195[] = {{0x47, 0x1, 0x40, {}}, {0x7f, 0x1, 0x40, {}}, {0x6f, 0x1, 0x40, {}}};
constexpr NodeSocketData table196[] = {
    {"Tangent", "Tangent", NodeSocketType::Vector, 3, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"}};
constexpr NodeEnumData table197[] = {{"RADIAL", "Radial"}, {"UV_MAP", "UV Map"}};
constexpr NodeEnumData table198[] = {{"X", "X"}, {"Y", "Y"}, {"Z", "Z"}};
constexpr NodePropertyData table199[] = {{"direction_type", "Direction", NodePropertyKind::Text, 0.0f, 0, "RADIAL", {}, table197, 1.0f, 0.0f, true, true},
                                         {"axis", "Axis", NodePropertyKind::Text, 0.0f, 0, "Z", {}, table198, 1.0f, 0.0f, true, true},
                                         {"uv_map", "UV Map", NodePropertyKind::Text, 0.0f, 0, "", {}, {}, 1.0f, 0.0f, true, true}};
constexpr NodeSocketData table200[] = {
    {"Vector", "Vector", NodeSocketType::Vector, 3, true, false, true, {0.0f, 0.0f, 0.0f, 0.0f}, "", -10000.0f, 10000.0f, "NONE"},
    {"Color1", "Color1", NodeSocketType::Color, 4, true, false, false, {0.800000012f, 0.800000012f, 0.800000012f, 1.0f}, "", 1.0f, 0.0f, ""},
    {"Color2", "Color2", NodeSocketType::Color, 4, true, false, false, {0.200000003f, 0.200000003f, 0.200000003f, 1.0f}, "", 1.0f, 0.0f, ""},
    {"Mortar", "Mortar", NodeSocketType::Color, 4, true, false, false, {0.0f, 0.0f, 0.0f, 1.0f}, "", 1.0f, 0.0f, ""},
    {"Scale", "Scale", NodeSocketType::Float, 1, true, false, false, {5.0f, 0.0f, 0.0f, 0.0f}, "", -1000.0f, 1000.0f, "NONE"},
    {"Mortar Size", "Mortar Size", NodeSocketType::Float, 1, true, false, false, {0.0199999996f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 0.125f, "NONE"},
    {"Mortar Smooth", "Mortar Smooth", NodeSocketType::Float, 1, true, false, false, {0.100000001f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "NONE"},
    {"Bias", "Bias", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -1.0f, 1.0f, "NONE"},
    {"Brick Width", "Brick Width", NodeSocketType::Float, 1, true, false, false, {0.5f, 0.0f, 0.0f, 0.0f}, "", 0.00999999978f, 100.0f, "NONE"},
    {"Row Height", "Row Height", NodeSocketType::Float, 1, true, false, false, {0.25f, 0.0f, 0.0f, 0.0f}, "", 0.00999999978f, 100.0f, "NONE"}};
constexpr NodeSocketData table201[] = {
    {"Color", "Color", NodeSocketType::Color, 4, true, false, false, {0.800000012f, 0.800000012f, 0.800000012f, 1.0f}, "", 1.0f, 0.0f, ""},
    {"Fac", "Factor", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"}};
constexpr NodePropertyData table202[] = {{"offset_frequency", "Offset Frequency", NodePropertyKind::Integer, 0.0f, 2, "", {}, {}, 1.0f, 99.0f, true, true},
                                         {"squash_frequency", "Squash Frequency", NodePropertyKind::Integer, 0.0f, 2, "", {}, {}, 1.0f, 99.0f, true, true},
                                         {"offset", "Offset Amount", NodePropertyKind::Number, 0.5f, 0, "", {}, {}, 0.0f, 1.0f, true, true},
                                         {"squash", "Squash Amount", NodePropertyKind::Number, 1.0f, 0, "", {}, {}, 0.0f, 99.0f, true, true}};
constexpr NodeMuting table203[] = {{0x6, 1}, {0x6, 1}};
constexpr NodeSocketData table204[] = {
    {"Vector", "Vector", NodeSocketType::Vector, 3, true, false, true, {0.0f, 0.0f, 0.0f, 0.0f}, "", -10000.0f, 10000.0f, "NONE"},
    {"Color1", "Color1", NodeSocketType::Color, 4, true, false, false, {0.800000012f, 0.800000012f, 0.800000012f, 1.0f}, "", 1.0f, 0.0f, ""},
    {"Color2", "Color2", NodeSocketType::Color, 4, true, false, false, {0.200000003f, 0.200000003f, 0.200000003f, 1.0f}, "", 1.0f, 0.0f, ""},
    {"Scale", "Scale", NodeSocketType::Float, 1, true, false, false, {5.0f, 0.0f, 0.0f, 0.0f}, "", -10000.0f, 10000.0f, "NONE"}};
constexpr NodeSocketData table205[] = {
    {"Generated", "Generated", NodeSocketType::Vector, 3, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Normal", "Normal", NodeSocketType::Vector, 3, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"UV", "UV", NodeSocketType::Vector, 3, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Object", "Object", NodeSocketType::Vector, 3, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Camera", "Camera", NodeSocketType::Vector, 3, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Window", "Window", NodeSocketType::Vector, 3, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Reflection", "Reflection", NodeSocketType::Vector, 3, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"}};
constexpr NodePropertyData table206[] = {{"object", "Object", NodePropertyKind::Text, 0.0f, 0, "", {}, {}, 1.0f, 0.0f, false, false},
                                         {"from_instancer", "From Instancer", NodePropertyKind::Boolean, 0.0f, 0, "", {}, {}, 1.0f, 0.0f, true, true}};
constexpr NodeMuting table207[] = {{0x0, -1}, {0x0, -1}, {0x0, -1}, {0x0, -1}, {0x0, -1}, {0x0, -1}, {0x0, -1}};
constexpr NodeSocketData table208[] = {
    {"Vector", "Vector", NodeSocketType::Vector, 3, true, false, true, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"}};
constexpr NodeEnumData table209[] = {{"EQUIRECTANGULAR", "Equirectangular"}, {"MIRROR_BALL", "Mirror Ball"}};
constexpr NodeEnumData table210[] = {{"Linear", "Linear"}, {"Closest", "Closest"}, {"Cubic", "Cubic"}, {"Smart", "Smart"}};
constexpr NodePropertyData table211[] = {{"image", "Image", NodePropertyKind::Text, 0.0f, 0, "", {}, {}, 1.0f, 0.0f, true, false},
                                         {"projection", "Projection", NodePropertyKind::Text, 0.0f, 0, "EQUIRECTANGULAR", {}, table209, 1.0f, 0.0f, true, true},
                                         {"interpolation", "Interpolation", NodePropertyKind::Text, 0.0f, 0, "Linear", {}, table210, 1.0f, 0.0f, true, true}};
constexpr NodeSocketData table212[] = {
    {"Vector", "Vector", NodeSocketType::Vector, 3, true, false, true, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Scale", "Scale", NodeSocketType::Float, 1, true, false, false, {5.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Frequency", "Frequency", NodeSocketType::Float, 1, true, false, false, {2.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 3.40282347e+38f, "NONE"},
    {"Anisotropy", "Anisotropy", NodeSocketType::Float, 1, true, false, false, {1.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
    {"Orientation 2D",
     "Orientation",
     NodeSocketType::Float,
     1,
     true,
     false,
     false,
     {0.785398185f, 0.0f, 0.0f, 0.0f},
     "",
     -3.40282347e+38f,
     3.40282347e+38f,
     "ANGLE"},
    {"Orientation 3D",
     "Orientation",
     NodeSocketType::Vector,
     3,
     false,
     false,
     false,
     {1.41421354f, 1.41421354f, 0.0f, 0.0f},
     "",
     -3.40282347e+38f,
     3.40282347e+38f,
     "DIRECTION"}};
constexpr NodeSocketData table213[] = {
    {"Value", "Value", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Phase", "Phase", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Intensity", "Intensity", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"}};
constexpr NodeEnumData table214[] = {{"2D", "2D"}, {"3D", "3D"}};
constexpr NodePropertyData table215[] = {{"gabor_type", "Type", NodePropertyKind::Text, 0.0f, 0, "2D", {}, table214, 1.0f, 0.0f, true, true}};
constexpr NodeMuting table216[] = {{0x3e, 1}, {0x3e, 1}, {0x3e, 1}};
constexpr std::string_view table217[] = {"gabor_type"};
constexpr NodeSocketState table218[] = {{0x1f, 0x7, 0x1, {}}, {0x2f, 0x7, 0x1, {}}};
constexpr NodeEnumData table219[] = {{"LINEAR", "Linear"},     {"QUADRATIC", "Quadratic"}, {"EASING", "Easing"},
                                     {"DIAGONAL", "Diagonal"}, {"SPHERICAL", "Spherical"}, {"QUADRATIC_SPHERE", "Quadratic Sphere"},
                                     {"RADIAL", "Radial"}};
constexpr NodePropertyData table220[] = {{"gradient_type", "Gradient Type", NodePropertyKind::Text, 0.0f, 0, "LINEAR", {}, table219, 1.0f, 0.0f, true, true}};
constexpr NodeMuting table221[] = {{0x0, -1}, {0x0, -1}};
constexpr NodeSocketData table222[] = {
    {"Vector", "Vector", NodeSocketType::Vector, 3, true, false, true, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Strength", "Strength", NodeSocketType::Float, 1, true, false, false, {1.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1000000.0f, "NONE"}};
constexpr NodeEnumData table223[] = {{"INTERNAL", "Internal"}, {"EXTERNAL", "External"}};
constexpr NodePropertyData table224[] = {{"ies", "IES Text", NodePropertyKind::Text, 0.0f, 0, "", {}, {}, 1.0f, 0.0f, false, false},
                                         {"filepath", "File Path", NodePropertyKind::Text, 0.0f, 0, "", {}, {}, 1.0f, 0.0f, true, true},
                                         {"mode", "Source", NodePropertyKind::Text, 0.0f, 0, "INTERNAL", {}, table223, 1.0f, 0.0f, true, true}};
constexpr NodeSocketData table225[] = {
    {"Color", "Color", NodeSocketType::Color, 4, true, false, false, {0.800000012f, 0.800000012f, 0.800000012f, 1.0f}, "", 1.0f, 0.0f, ""},
    {"Alpha", "Alpha", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"}};
constexpr NodeEnumData table226[] = {{"FLAT", "Flat"}, {"BOX", "Box"}, {"SPHERE", "Sphere"}, {"TUBE", "Tube"}};
constexpr NodeEnumData table227[] = {{"REPEAT", "Repeat"}, {"EXTEND", "Extend"}, {"CLIP", "Clip"}, {"MIRROR", "Mirror"}};
constexpr NodePropertyData table228[] = {{"image", "Image", NodePropertyKind::Text, 0.0f, 0, "", {}, {}, 1.0f, 0.0f, true, false},
                                         {"projection", "Projection", NodePropertyKind::Text, 0.0f, 0, "FLAT", {}, table226, 1.0f, 0.0f, true, true},
                                         {"interpolation", "Interpolation", NodePropertyKind::Text, 0.0f, 0, "Linear", {}, table210, 1.0f, 0.0f, true, true},
                                         {"projection_blend", "Projection Blend", NodePropertyKind::Number, 0.0f, 0, "", {}, {}, 0.0f, 1.0f, true, true},
                                         {"extension", "Extension", NodePropertyKind::Text, 0.0f, 0, "REPEAT", {}, table227, 1.0f, 0.0f, true, true}};
constexpr NodeSocketData table229[] = {
    {"Vector", "Vector", NodeSocketType::Vector, 3, true, false, true, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Scale", "Scale", NodeSocketType::Float, 1, true, false, false, {5.0f, 0.0f, 0.0f, 0.0f}, "", -1000.0f, 1000.0f, "NONE"},
    {"Distortion", "Distortion", NodeSocketType::Float, 1, true, false, false, {1.0f, 0.0f, 0.0f, 0.0f}, "", -1000.0f, 1000.0f, "NONE"}};
constexpr NodePropertyData table230[] = {{"turbulence_depth", "Depth", NodePropertyKind::Integer, 0.0f, 2, "", {}, {}, 0.0f, 10.0f, true, true}};
constexpr NodeSocketData table231[] = {
    {"Vector", "Vector", NodeSocketType::Vector, 3, true, false, true, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"W", "W", NodeSocketType::Float, 1, false, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -1000.0f, 1000.0f, "NONE"},
    {"Scale", "Scale", NodeSocketType::Float, 1, true, false, false, {5.0f, 0.0f, 0.0f, 0.0f}, "", -1000.0f, 1000.0f, "NONE"},
    {"Detail", "Detail", NodeSocketType::Float, 1, true, false, false, {2.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 15.0f, "NONE"},
    {"Roughness", "Roughness", NodeSocketType::Float, 1, true, false, false, {0.5f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
    {"Lacunarity", "Lacunarity", NodeSocketType::Float, 1, true, false, false, {2.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1000.0f, "NONE"},
    {"Offset", "Offset", NodeSocketType::Float, 1, false, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -1000.0f, 1000.0f, "NONE"},
    {"Gain", "Gain", NodeSocketType::Float, 1, false, false, false, {1.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1000.0f, "NONE"},
    {"Distortion", "Distortion", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -1000.0f, 1000.0f, "NONE"}};
constexpr NodeSocketData table232[] = {
    {"Fac", "Factor", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Color", "Color", NodeSocketType::Color, 4, true, false, false, {0.800000012f, 0.800000012f, 0.800000012f, 1.0f}, "", 1.0f, 0.0f, ""}};
constexpr NodeEnumData table233[] = {{"1D", "1D"}, {"2D", "2D"}, {"3D", "3D"}, {"4D", "4D"}};
constexpr NodeEnumData table234[] = {{"MULTIFRACTAL", "Multifractal"},
                                     {"RIDGED_MULTIFRACTAL", "Ridged Multifractal"},
                                     {"HYBRID_MULTIFRACTAL", "Hybrid Multifractal"},
                                     {"FBM", "fBM"},
                                     {"HETERO_TERRAIN", "Hetero Terrain"}};
constexpr NodePropertyData table235[] = {{"noise_dimensions", "Dimensions", NodePropertyKind::Text, 0.0f, 0, "3D", {}, table233, 1.0f, 0.0f, true, true},
                                         {"noise_type", "Type", NodePropertyKind::Text, 0.0f, 0, "FBM", {}, table234, 1.0f, 0.0f, true, true},
                                         {"normalize", "Normalize", NodePropertyKind::Boolean, 0.0f, 1, "", {}, {}, 1.0f, 0.0f, true, true}};
constexpr std::string_view table236[] = {"noise_dimensions", "noise_type"};
constexpr std::uint8_t table237[] = {0, 1, 1, 0, 2, 3, 4, 4, 3, 5, 3, 4, 4, 3, 5, 6, 7, 7, 6, 8};
constexpr NodeSocketState table238[] = {{0x13e, 0x3, 0x1, {}}, {0x1fe, 0x3, 0x1, {}}, {0x17e, 0x3, 0x1, {}}, {0x13d, 0x3, 0x1, {}}, {0x1fd, 0x3, 0x1, {}},
                                        {0x17d, 0x3, 0x1, {}}, {0x13f, 0x3, 0x1, {}}, {0x1ff, 0x3, 0x1, {}}, {0x17f, 0x3, 0x1, {}}};
constexpr NodeSocketData table239[] = {
    {"Vector", "Vector", NodeSocketType::Vector, 3, false, false, true, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"}};
constexpr NodeEnumData table240[] = {
    {"SINGLE_SCATTERING", "Single Scattering"}, {"MULTIPLE_SCATTERING", "Multiple Scattering"}, {"PREETHAM", "Preetham"}, {"HOSEK_WILKIE", "Hosek / Wilkie"}};
constexpr float table241[] = {0.0f, 0.0f, 1.0f};
constexpr NodePropertyData table242[] = {
    {"sky_type", "Sky Type", NodePropertyKind::Text, 0.0f, 0, "MULTIPLE_SCATTERING", {}, table240, 1.0f, 0.0f, true, true},
    {"sun_disc", "Sun Disc", NodePropertyKind::Boolean, 0.0f, 1, "", {}, {}, 1.0f, 0.0f, true, true},
    {"sun_size", "Sun Size", NodePropertyKind::Number, 0.00951204449f, 0, "", {}, {}, 0.0f, 1.57079637f, true, true},
    {"sun_intensity", "Sun Intensity", NodePropertyKind::Number, 1.0f, 0, "", {}, {}, 0.0f, 1000.0f, true, true},
    {"sun_elevation", "Sun Elevation", NodePropertyKind::Number, 0.261799395f, 0, "", {}, {}, -10000.0f, 10000.0f, true, true},
    {"sun_rotation", "Sun Rotation", NodePropertyKind::Number, 0.0f, 0, "", {}, {}, -10000.0f, 10000.0f, true, true},
    {"altitude", "Altitude", NodePropertyKind::Number, 100.0f, 0, "", {}, {}, 0.0f, 100000.0f, true, true},
    {"air_density", "Air", NodePropertyKind::Number, 1.0f, 0, "", {}, {}, 0.0f, 1000.0f, true, true},
    {"aerosol_density", "Aerosols", NodePropertyKind::Number, 1.0f, 0, "", {}, {}, 0.0f, 1000.0f, true, true},
    {"ozone_density", "Ozone", NodePropertyKind::Number, 1.0f, 0, "", {}, {}, 0.0f, 1000.0f, true, true},
    {"sun_direction", "Sun Direction", NodePropertyKind::Numbers, 0.0f, 0, "", table241, {}, 1.0f, 0.0f, false, true},
    {"turbidity", "Turbidity", NodePropertyKind::Number, 2.20000005f, 0, "", {}, {}, 1.0f, 10.0f, true, true},
    {"ground_albedo", "Ground Albedo", NodePropertyKind::Number, 0.300000012f, 0, "", {}, {}, 0.0f, 1.0f, true, true}};
constexpr std::string_view table243[] = {"sky_type", "sun_disc"};
constexpr std::uint8_t table244[] = {0, 1, 0, 1, 0, 0, 0, 0};
constexpr NodeSocketState table245[] = {{0x1, 0x1, 0x1, {}}, {0x0, 0x1, 0x1, {}}};
constexpr NodeSocketData table246[] = {
    {"Vector", "Vector", NodeSocketType::Vector, 3, true, false, true, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"W", "W", NodeSocketType::Float, 1, false, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -1000.0f, 1000.0f, "NONE"},
    {"Scale", "Scale", NodeSocketType::Float, 1, true, false, false, {5.0f, 0.0f, 0.0f, 0.0f}, "", -1000.0f, 1000.0f, "NONE"},
    {"Detail", "Detail", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 15.0f, "NONE"},
    {"Roughness", "Roughness", NodeSocketType::Float, 1, true, false, false, {0.5f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
    {"Lacunarity", "Lacunarity", NodeSocketType::Float, 1, true, false, false, {2.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1000.0f, "NONE"},
    {"Smoothness", "Smoothness", NodeSocketType::Float, 1, false, false, false, {1.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
    {"Exponent", "Exponent", NodeSocketType::Float, 1, false, false, false, {0.5f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 32.0f, "NONE"},
    {"Randomness", "Randomness", NodeSocketType::Float, 1, true, false, false, {1.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"}};
constexpr NodeSocketData table247[] = {
    {"Distance", "Distance", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Color", "Color", NodeSocketType::Color, 4, true, false, false, {0.800000012f, 0.800000012f, 0.800000012f, 1.0f}, "", 1.0f, 0.0f, ""},
    {"Position", "Position", NodeSocketType::Vector, 3, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"W", "W", NodeSocketType::Float, 1, false, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Radius", "Radius", NodeSocketType::Float, 1, false, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"}};
constexpr NodeEnumData table248[] = {{"EUCLIDEAN", "Euclidean"}, {"MANHATTAN", "Manhattan"}, {"CHEBYCHEV", "Chebychev"}, {"MINKOWSKI", "Minkowski"}};
constexpr NodeEnumData table249[] = {
    {"F1", "F1"}, {"F2", "F2"}, {"SMOOTH_F1", "Smooth F1"}, {"DISTANCE_TO_EDGE", "Distance to Edge"}, {"N_SPHERE_RADIUS", "N-Sphere Radius"}};
constexpr NodePropertyData table250[] = {{"voronoi_dimensions", "Dimensions", NodePropertyKind::Text, 0.0f, 0, "3D", {}, table233, 1.0f, 0.0f, true, true},
                                         {"distance", "Distance Metric", NodePropertyKind::Text, 0.0f, 0, "EUCLIDEAN", {}, table248, 1.0f, 0.0f, true, true},
                                         {"feature", "Feature Output", NodePropertyKind::Text, 0.0f, 0, "F1", {}, table249, 1.0f, 0.0f, true, true},
                                         {"normalize", "Normalize", NodePropertyKind::Boolean, 0.0f, 0, "", {}, {}, 1.0f, 0.0f, true, true}};
constexpr NodeMuting table251[] = {{0x0, -1}, {0x0, -1}, {0x0, -1}, {0x0, -1}, {0x0, -1}};
constexpr std::string_view table252[] = {"voronoi_dimensions", "distance", "feature"};
constexpr std::uint8_t table253[] = {0, 0, 1, 2, 3, 0, 0,  1,  2,  3,  0,  0,  1,  2,  3,  0,  0,  1,  2,  3,  4,  4,  5,  6,  7,  4, 4,
                                     5, 6, 7, 4, 4, 5, 6,  7,  8,  8,  9,  6,  7,  4,  4,  5,  6,  7,  4,  4,  5,  6,  7,  4,  4,  5, 6,
                                     7, 8, 8, 9, 6, 7, 10, 10, 11, 12, 13, 10, 10, 11, 12, 13, 10, 10, 11, 12, 13, 14, 14, 15, 12, 13};
constexpr NodeSocketState table254[] = {{0x13e, 0xb, 0x1, {}}, {0x17e, 0xb, 0x1, {}},  {0x13e, 0x1, 0x1, {}}, {0x106, 0x10, 0x1, {}},
                                        {0x13d, 0x7, 0x1, {}}, {0x17d, 0x7, 0x1, {}},  {0x13d, 0x1, 0x1, {}}, {0x105, 0x10, 0x1, {}},
                                        {0x1bd, 0x7, 0x1, {}}, {0x1fd, 0x7, 0x1, {}},  {0x13f, 0xf, 0x1, {}}, {0x17f, 0xf, 0x1, {}},
                                        {0x13f, 0x1, 0x1, {}}, {0x107, 0x10, 0x1, {}}, {0x1bf, 0xf, 0x1, {}}, {0x1ff, 0xf, 0x1, {}}};
constexpr NodeSocketData table255[] = {
    {"Vector", "Vector", NodeSocketType::Vector, 3, true, false, true, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Scale", "Scale", NodeSocketType::Float, 1, true, false, false, {5.0f, 0.0f, 0.0f, 0.0f}, "", -1000.0f, 1000.0f, "NONE"},
    {"Distortion", "Distortion", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -1000.0f, 1000.0f, "NONE"},
    {"Detail", "Detail", NodeSocketType::Float, 1, true, false, false, {2.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 15.0f, "NONE"},
    {"Detail Scale", "Detail Scale", NodeSocketType::Float, 1, true, false, false, {1.0f, 0.0f, 0.0f, 0.0f}, "", -1000.0f, 1000.0f, "NONE"},
    {"Detail Roughness", "Detail Roughness", NodeSocketType::Float, 1, true, false, false, {0.5f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
    {"Phase Offset", "Phase Offset", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -1000.0f, 1000.0f, "NONE"}};
constexpr NodeEnumData table256[] = {{"BANDS", "Bands"}, {"RINGS", "Rings"}};
constexpr NodeEnumData table257[] = {{"X", "X"}, {"Y", "Y"}, {"Z", "Z"}, {"DIAGONAL", "Diagonal"}};
constexpr NodeEnumData table258[] = {{"X", "X"}, {"Y", "Y"}, {"Z", "Z"}, {"SPHERICAL", "Spherical"}};
constexpr NodeEnumData table259[] = {{"SIN", "Sine"}, {"SAW", "Saw"}, {"TRI", "Triangle"}};
constexpr NodePropertyData table260[] = {{"wave_type", "Wave Type", NodePropertyKind::Text, 0.0f, 0, "BANDS", {}, table256, 1.0f, 0.0f, true, true},
                                         {"bands_direction", "Bands Direction", NodePropertyKind::Text, 0.0f, 0, "X", {}, table257, 1.0f, 0.0f, true, true},
                                         {"rings_direction", "Rings Direction", NodePropertyKind::Text, 0.0f, 0, "X", {}, table258, 1.0f, 0.0f, true, true},
                                         {"wave_profile", "Wave Profile", NodePropertyKind::Text, 0.0f, 0, "SIN", {}, table259, 1.0f, 0.0f, true, true}};
constexpr NodeSocketData table261[] = {
    {"Vector", "Vector", NodeSocketType::Vector, 3, true, false, true, {0.0f, 0.0f, 0.0f, 0.0f}, "", -10000.0f, 10000.0f, "NONE"},
    {"W", "W", NodeSocketType::Float, 1, false, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -10000.0f, 10000.0f, "NONE"}};
constexpr NodeSocketData table262[] = {
    {"Value", "Value", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Color", "Color", NodeSocketType::Color, 4, true, false, false, {0.800000012f, 0.800000012f, 0.800000012f, 1.0f}, "", 1.0f, 0.0f, ""}};
constexpr NodePropertyData table263[] = {{"noise_dimensions", "Dimensions", NodePropertyKind::Text, 0.0f, 0, "3D", {}, table233, 1.0f, 0.0f, true, true}};
constexpr NodeMuting table264[] = {{0x3, 1}, {0x2, 1}};
constexpr std::string_view table265[] = {"noise_dimensions"};
constexpr std::uint8_t table266[] = {0, 1, 1, 2};
constexpr NodeSocketState table267[] = {{0x2, 0x3, 0x1, {}}, {0x1, 0x3, 0x1, {}}, {0x3, 0x3, 0x1, {}}};
constexpr NodeSocketData table268[] = {
    {"UV", "UV", NodeSocketType::Vector, 3, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"}};
constexpr NodePropertyData table269[] = {{"from_instancer", "From Instancer", NodePropertyKind::Boolean, 0.0f, 0, "", {}, {}, 1.0f, 0.0f, true, true},
                                         {"uv_map", "UV Map", NodePropertyKind::Text, 0.0f, 0, "", {}, {}, 1.0f, 0.0f, true, true}};
constexpr NodeSocketData table270[] = {{"Fac", "Factor", NodeSocketType::Float, 1, true, false, false, {0.5f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"}};
constexpr NodeEnumData table271[] = {{"EASE", "Ease"}, {"CARDINAL", "Cardinal"}, {"LINEAR", "Linear"}, {"B_SPLINE", "B-Spline"}, {"CONSTANT", "Constant"}};
constexpr NodeEnumData table272[] = {{"NEAR", "Near"}, {"FAR", "Far"}, {"CW", "Clockwise"}, {"CCW", "Counter-Clockwise"}};
constexpr float table273[] = {0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
constexpr NodePropertyData table274[] = {
    {"color_ramp.interpolation", "Interpolation", NodePropertyKind::Text, 0.0f, 0, "LINEAR", {}, table271, 1.0f, 0.0f, true, true},
    {"color_ramp.color_mode", "Color Mode", NodePropertyKind::Text, 0.0f, 0, "RGB", {}, table84, 1.0f, 0.0f, true, true},
    {"color_ramp.hue_interpolation", "Color Interpolation", NodePropertyKind::Text, 0.0f, 0, "NEAR", {}, table272, 1.0f, 0.0f, true, true},
    {"color_ramp.elements", "Elements", NodePropertyKind::Numbers, 0.0f, 0, "", table273, {}, 1.0f, 0.0f, true, true}};
constexpr NodeSocketData table275[] = {{"Fac", "Factor", NodeSocketType::Float, 1, true, false, false, {1.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "FACTOR"},
                                       {"Vector", "Vector", NodeSocketType::Vector, 3, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -1.0f, 1.0f, "NONE"}};
constexpr float table276[] = {1.0f, -1.0f, -1.0f, 1.0f, 1.0f};
constexpr float table277[] = {-1.0f, -1.0f, 1.0f, 1.0f};
constexpr NodePropertyData table278[] = {{"mapping.curve_count", "Curves", NodePropertyKind::Integer, 0.0f, 3, "", {}, {}, 1.0f, 0.0f, false, true},
                                         {"mapping.extend", "Extend", NodePropertyKind::Text, 0.0f, 0, "EXTRAPOLATED", {}, table97, 1.0f, 0.0f, true, true},
                                         {"mapping.clip", "Clip", NodePropertyKind::Numbers, 0.0f, 0, "", table276, {}, 1.0f, 0.0f, false, true},
                                         {"mapping.curve0", "X", NodePropertyKind::Numbers, 0.0f, 0, "", table277, {}, 1.0f, 0.0f, true, true},
                                         {"mapping.handles0", "X Handles", NodePropertyKind::Numbers, 0.0f, 0, "", table100, {}, 1.0f, 0.0f, false, true},
                                         {"mapping.curve1", "Y", NodePropertyKind::Numbers, 0.0f, 0, "", table277, {}, 1.0f, 0.0f, true, true},
                                         {"mapping.handles1", "Y Handles", NodePropertyKind::Numbers, 0.0f, 0, "", table100, {}, 1.0f, 0.0f, false, true},
                                         {"mapping.curve2", "Z", NodePropertyKind::Numbers, 0.0f, 0, "", table277, {}, 1.0f, 0.0f, true, true},
                                         {"mapping.handles2", "Z Handles", NodePropertyKind::Numbers, 0.0f, 0, "", table100, {}, 1.0f, 0.0f, false, true}};
constexpr NodeSocketData table279[] = {
    {"Vector", "Vector", NodeSocketType::Color, 4, true, false, true, {0.800000012f, 0.800000012f, 0.800000012f, 1.0f}, "", 1.0f, 0.0f, ""},
    {"Midlevel", "Midlevel", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1000.0f, "NONE"},
    {"Scale", "Scale", NodeSocketType::Float, 1, true, false, false, {0.00999999978f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1000.0f, "NONE"}};
constexpr NodeEnumData table280[] = {{"TANGENT", "Tangent Space"}, {"OBJECT", "Object Space"}, {"WORLD", "World Space"}};
constexpr NodePropertyData table281[] = {{"space", "Space", NodePropertyKind::Text, 0.0f, 0, "TANGENT", {}, table280, 1.0f, 0.0f, true, true}};
constexpr NodeSocketData table282[] = {
    {"Vector", "Vector", NodeSocketType::Vector, 3, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -10000.0f, 10000.0f, "NONE"},
    {"Vector_001", "Vector", NodeSocketType::Vector, 3, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -10000.0f, 10000.0f, "NONE"},
    {"Vector_002", "Vector", NodeSocketType::Vector, 3, false, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -10000.0f, 10000.0f, "NONE"},
    {"Scale", "Scale", NodeSocketType::Float, 1, false, false, false, {1.0f, 0.0f, 0.0f, 0.0f}, "", -10000.0f, 10000.0f, "NONE"}};
constexpr NodeSocketData table283[] = {
    {"Vector", "Vector", NodeSocketType::Vector, 3, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Value", "Value", NodeSocketType::Float, 1, false, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"}};
constexpr NodeEnumData table284[] = {{"ADD", "Add"},
                                     {"SUBTRACT", "Subtract"},
                                     {"MULTIPLY", "Multiply"},
                                     {"DIVIDE", "Divide"},
                                     {"MULTIPLY_ADD", "Multiply Add"},
                                     {"CROSS_PRODUCT", "Cross Product"},
                                     {"PROJECT", "Project"},
                                     {"REFLECT", "Reflect"},
                                     {"REFRACT", "Refract"},
                                     {"FACEFORWARD", "Faceforward"},
                                     {"DOT_PRODUCT", "Dot Product"},
                                     {"DISTANCE", "Distance"},
                                     {"LENGTH", "Length"},
                                     {"SCALE", "Scale"},
                                     {"NORMALIZE", "Normalize"},
                                     {"ABSOLUTE", "Absolute"},
                                     {"POWER", "Power"},
                                     {"SIGN", "Sign"},
                                     {"MINIMUM", "Minimum"},
                                     {"MAXIMUM", "Maximum"},
                                     {"ROUND", "Round"},
                                     {"FLOOR", "Floor"},
                                     {"CEIL", "Ceil"},
                                     {"FRACTION", "Fraction"},
                                     {"MODULO", "Modulo"},
                                     {"WRAP", "Wrap"},
                                     {"SNAP", "Snap"},
                                     {"SINE", "Sine"},
                                     {"COSINE", "Cosine"},
                                     {"TANGENT", "Tangent"}};
constexpr NodePropertyData table285[] = {{"operation", "Operation", NodePropertyKind::Text, 0.0f, 0, "ADD", {}, table284, 1.0f, 0.0f, true, true}};
constexpr NodeMuting table286[] = {{0xf, 0}, {0x3, 0}};
constexpr std::uint8_t table287[] = {0, 0, 0, 0, 1, 0, 0, 0, 2, 1, 3, 3, 4, 5, 6, 6, 0, 6, 0, 0, 6, 6, 6, 6, 0, 1, 0, 6, 6, 6};
constexpr NodeSocketState table288[] = {{0x3, 0x1, 0x0, {}}, {0x7, 0x1, 0x0, {}}, {0xb, 0x1, 0x0, {}}, {0x3, 0x2, 0x0, {}},
                                        {0x1, 0x2, 0x0, {}}, {0x9, 0x1, 0x0, {}}, {0x1, 0x1, 0x0, {}}};
constexpr NodeSocketData table289[] = {
    {"Vector", "Vector", NodeSocketType::Vector, 3, true, false, true, {0.0f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 1.0f, "NONE"},
    {"Center", "Center", NodeSocketType::Vector, 3, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Axis", "Axis", NodeSocketType::Vector, 3, true, false, false, {0.0f, 0.0f, 1.0f, 0.0f}, "", -1.0f, 1.0f, "NONE"},
    {"Angle", "Angle", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "ANGLE"},
    {"Rotation", "Rotation", NodeSocketType::Vector, 3, false, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "EULER"}};
constexpr NodeEnumData table290[] = {{"AXIS_ANGLE", "Axis Angle"}, {"X_AXIS", "X Axis"}, {"Y_AXIS", "Y Axis"}, {"Z_AXIS", "Z Axis"}, {"EULER_XYZ", "Euler"}};
constexpr NodePropertyData table291[] = {{"rotation_type", "Type", NodePropertyKind::Text, 0.0f, 0, "AXIS_ANGLE", {}, table290, 1.0f, 0.0f, true, true},
                                         {"invert", "Invert", NodePropertyKind::Boolean, 0.0f, 0, "", {}, {}, 1.0f, 0.0f, true, true}};
constexpr NodeMuting table292[] = {{0x1f, 0}};
constexpr std::string_view table293[] = {"rotation_type"};
constexpr std::uint8_t table294[] = {0, 1, 1, 1, 2};
constexpr NodeSocketState table295[] = {{0xf, 0x1, 0x1, {}}, {0xb, 0x1, 0x1, {}}, {0x13, 0x1, 0x1, {}}};
constexpr NodeSocketData table296[] = {
    {"Vector", "Vector", NodeSocketType::Vector, 3, true, false, false, {0.5f, 0.5f, 0.5f, 0.0f}, "", -10000.0f, 10000.0f, "NONE"}};
constexpr NodeEnumData table297[] = {{"POINT", "Point"}, {"VECTOR", "Vector"}, {"NORMAL", "Normal"}};
constexpr NodeEnumData table298[] = {{"WORLD", "World"}, {"OBJECT", "Object"}, {"CAMERA", "Camera"}};
constexpr NodePropertyData table299[] = {{"vector_type", "Type", NodePropertyKind::Text, 0.0f, 0, "VECTOR", {}, table297, 1.0f, 0.0f, true, true},
                                         {"convert_from", "Convert From", NodePropertyKind::Text, 0.0f, 0, "WORLD", {}, table298, 1.0f, 0.0f, true, true},
                                         {"convert_to", "Convert To", NodePropertyKind::Text, 0.0f, 0, "OBJECT", {}, table298, 1.0f, 0.0f, true, true}};
constexpr NodePropertyData table300[] = {{"layer_name", "Color Attribute", NodePropertyKind::Text, 0.0f, 0, "", {}, {}, 1.0f, 0.0f, true, true}};
constexpr NodeSocketData table301[] = {
    {"Color", "Color", NodeSocketType::Color, 4, true, false, false, {0.800000012f, 0.800000012f, 0.800000012f, 1.0f}, "", 1.0f, 0.0f, ""},
    {"Density", "Density", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Flame", "Flame", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"},
    {"Temperature", "Temperature", NodeSocketType::Float, 1, true, false, false, {0.0f, 0.0f, 0.0f, 0.0f}, "", -3.40282347e+38f, 3.40282347e+38f, "NONE"}};
constexpr NodeSocketData table302[] = {
    {"Wavelength", "Wavelength", NodeSocketType::Float, 1, true, false, false, {500.0f, 0.0f, 0.0f, 0.0f}, "", 380.0f, 780.0f, "WAVELENGTH"}};
constexpr NodeSocketData table303[] = {
    {"Size", "Size", NodeSocketType::Float, 1, true, false, false, {0.00999999978f, 0.0f, 0.0f, 0.0f}, "", 0.0f, 100.0f, "NONE"}};
constexpr NodePropertyData table304[] = {{"use_pixel_size", "Pixel Size", NodePropertyKind::Boolean, 0.0f, 0, "", {}, {}, 1.0f, 0.0f, true, true}};
constexpr NodeDefault table305[] = {{"ShaderNodeBsdfPrincipled", {-200.0f, 100.0f}, 240.0f}, {"ShaderNodeOutputMaterial", {200.0f, 100.0f}, 140.0f}};
constexpr NodeLink table306[] = {{0, 0, 1, 0}};
constexpr float table307[] = {0.800000012f, 0.800000012f, 0.800000012f, 1.0f};
constexpr NodePropertyData table308[] = {{"use_nodes", "", NodePropertyKind::Boolean, 0.0f, 1, "", {}, {}, 1.0f, 0.0f, true, true},
                                         {"pass_index", "", NodePropertyKind::Integer, 0.0f, 0, "", {}, {}, 1.0f, 0.0f, true, true},
                                         {"displacement_method", "", NodePropertyKind::Text, 0.0f, 0, "BUMP", {}, {}, 1.0f, 0.0f, true, true},
                                         {"emission_sampling", "", NodePropertyKind::Text, 0.0f, 0, "AUTO", {}, {}, 1.0f, 0.0f, true, true},
                                         {"diffuse_color", "", NodePropertyKind::Numbers, 0.0f, 0, "", table307, {}, 1.0f, 0.0f, true, true},
                                         {"metallic", "", NodePropertyKind::Number, 0.0f, 0, "", {}, {}, 1.0f, 0.0f, true, true},
                                         {"roughness", "", NodePropertyKind::Number, 0.400000006f, 0, "", {}, {}, 1.0f, 0.0f, true, true}};
constexpr NodeDefault table309[] = {{"ShaderNodeEmission", {-200.0f, 100.0f}, 140.0f}, {"ShaderNodeOutputLight", {200.0f, 100.0f}, 140.0f}};
constexpr float table310[] = {1.0f, 1.0f, 1.0f};
constexpr NodePropertyData table311[] = {{"use_nodes", "", NodePropertyKind::Boolean, 0.0f, 1, "", {}, {}, 1.0f, 0.0f, true, true},
                                         {"energy", "", NodePropertyKind::Number, 10.0f, 0, "", {}, {}, 1.0f, 0.0f, true, true},
                                         {"exposure", "", NodePropertyKind::Number, 0.0f, 0, "", {}, {}, 1.0f, 0.0f, true, true},
                                         {"color", "", NodePropertyKind::Numbers, 0.0f, 0, "", table310, {}, 1.0f, 0.0f, true, true},
                                         {"use_temperature", "", NodePropertyKind::Boolean, 0.0f, 0, "", {}, {}, 1.0f, 0.0f, true, true},
                                         {"normalize", "", NodePropertyKind::Boolean, 0.0f, 1, "", {}, {}, 1.0f, 0.0f, true, true}};
constexpr NodeDefault table312[] = {{"ShaderNodeBackground", {-200.0f, 100.0f}, 140.0f}, {"ShaderNodeOutputWorld", {200.0f, 100.0f}, 140.0f}};
constexpr NodeDefaultValue table313[] = {{0, 0, {0.0500000007f, 0.0500000007f, 0.0500000007f, 1.0f}}};
constexpr NodePropertyData table314[] = {{"name", "", NodePropertyKind::Text, 0.0f, 0, "World", {}, {}, 1.0f, 0.0f, true, true},
                                         {"use_nodes", "", NodePropertyKind::Boolean, 0.0f, 1, "", {}, {}, 1.0f, 0.0f, true, true},
                                         {"film_transparent", "", NodePropertyKind::Boolean, 0.0f, 0, "", {}, {}, 1.0f, 0.0f, true, true},
                                         {"cycles_visibility.camera", "", NodePropertyKind::Boolean, 0.0f, 1, "", {}, {}, 1.0f, 0.0f, true, true},
                                         {"cycles_visibility.diffuse", "", NodePropertyKind::Boolean, 0.0f, 1, "", {}, {}, 1.0f, 0.0f, true, true},
                                         {"cycles_visibility.glossy", "", NodePropertyKind::Boolean, 0.0f, 1, "", {}, {}, 1.0f, 0.0f, true, true},
                                         {"cycles_visibility.transmission", "", NodePropertyKind::Boolean, 0.0f, 1, "", {}, {}, 1.0f, 0.0f, true, true},
                                         {"cycles_visibility.scatter", "", NodePropertyKind::Boolean, 0.0f, 1, "", {}, {}, 1.0f, 0.0f, true, true},
                                         {"sampling_method", "", NodePropertyKind::Text, 0.0f, 0, "AUTOMATIC", {}, {}, 1.0f, 0.0f, true, true}};

constexpr NodeTypeData types[] = {
    {"NodeFrame", "Frame", "Layout", 160.0f, {}, {}, table0, table1, {}, {}, {}, {}},
    {"NodeReroute", "Reroute", "Layout", 140.0f, table2, table3, table4, table1, {}, {}, {}, table5},
    {"ShaderNodeAddShader", "Add Shader", "Shader", 140.0f, table6, table7, {}, table1, {}, {}, {}, table8},
    {"ShaderNodeAmbientOcclusion", "Ambient Occlusion", "Input", 140.0f, table9, table10, table11, table12, {}, {}, {}, table13},
    {"ShaderNodeAttribute", "Attribute", "Input", 140.0f, {}, table14, table16, table1, {}, {}, {}, table17},
    {"ShaderNodeBackground", "Background", "Shader", 140.0f, table18, table19, {}, table20, {}, {}, {}, table21},
    {"ShaderNodeBevel", "Bevel", "Input", 140.0f, table22, table23, table24, table12, {}, {}, {}, table25},
    {"ShaderNodeBlackbody", "Blackbody", "Color", 160.0f, table26, table27, {}, table1, {}, {}, {}, table28},
    {"ShaderNodeBrightContrast", "Brightness/Contrast", "Color", 140.0f, table29, table27, {}, table1, {}, {}, {}, table30},
    {"ShaderNodeBsdfAnisotropic", "Glossy BSDF", "Shader", 160.0f, table31, table32, table34, table12, {}, {}, {}, table21},
    {"ShaderNodeBsdfDiffuse", "Diffuse BSDF", "Shader", 160.0f, table35, table32, {}, table12, {}, {}, {}, table21},
    {"ShaderNodeBsdfGlass", "Glass BSDF", "Shader", 160.0f, table36, table32, table38, table12, {}, {}, {}, table21},
    {"ShaderNodeBsdfHair", "Hair BSDF", "Shader", 160.0f, table39, table32, table41, table12, {}, {}, {}, table21},
    {"ShaderNodeBsdfHairPrincipled", "Principled Hair BSDF", "Shader", 240.0f, table42, table32, table45, table12, table46, table47, table48, table21},
    {"ShaderNodeBsdfMetallic", "Metallic BSDF", "Shader", 240.0f, table49, table32, table51, table12, table52, table53, table54, table21},
    {"ShaderNodeBsdfPrincipled", "Principled BSDF", "Shader", 240.0f, table55, table32, table58, table12, table59, table60, table61, table21},
    {"ShaderNodeBsdfRayPortal", "Ray Portal BSDF", "Shader", 140.0f, table62, table32, {}, table12, {}, {}, {}, table21},
    {"ShaderNodeBsdfRefraction", "Refraction BSDF", "Shader", 160.0f, table63, table32, table65, table12, {}, {}, {}, table21},
    {"ShaderNodeBsdfSheen", "Sheen BSDF", "Shader", 140.0f, table66, table32, table68, table12, {}, {}, {}, table21},
    {"ShaderNodeBsdfToon", "Toon BSDF", "Shader", 160.0f, table69, table32, table71, table12, {}, {}, {}, table21},
    {"ShaderNodeBsdfTranslucent", "Translucent BSDF", "Shader", 140.0f, table72, table32, {}, table12, {}, {}, {}, table21},
    {"ShaderNodeBsdfTransparent", "Transparent BSDF", "Shader", 140.0f, table73, table32, {}, table12, {}, {}, {}, table21},
    {"ShaderNodeBump", "Bump", "Displacement", 140.0f, table74, table23, table75, table1, {}, {}, {}, table76},
    {"ShaderNodeCameraData", "Camera Data", "Input", 140.0f, {}, table77, {}, table1, {}, {}, {}, table78},
    {"ShaderNodeClamp", "Clamp", "Math", 140.0f, table79, table80, table82, table1, {}, {}, {}, table30},
    {"ShaderNodeCombineColor", "Combine Color", "Color", 140.0f, table83, table27, table85, table1, {}, {}, {}, table30},
    {"ShaderNodeCombineXYZ", "Combine XYZ", "Vector", 140.0f, table86, table87, {}, table1, {}, {}, {}, table30},
    {"ShaderNodeDisplacement", "Displacement", "Displacement", 140.0f, table88, table89, table91, table1, {}, {}, {}, table92},
    {"ShaderNodeEmission", "Emission", "Shader", 140.0f, table93, table94, {}, table1, {}, {}, {}, table21},
    {"ShaderNodeFloatCurve", "Float Curve", "Math", 240.0f, table95, table96, table101, table1, {}, {}, {}, table25},
    {"ShaderNodeFresnel", "Fresnel", "Input", 140.0f, table102, table103, {}, table12, {}, {}, {}, table8},
    {"ShaderNodeGamma", "Gamma", "Color", 140.0f, table104, table27, {}, table1, {}, {}, {}, table8},
    {"ShaderNodeHairInfo", "Curves Info", "Input", 140.0f, {}, table105, {}, table12, {}, {}, {}, table106},
    {"ShaderNodeHoldout", "Holdout", "Shader", 140.0f, table107, table108, {}, table12, {}, {}, {}, table21},
    {"ShaderNodeHueSaturation", "Hue/Saturation/Value", "Color", 160.0f, table109, table27, {}, table1, {}, {}, {}, table76},
    {"ShaderNodeInvert", "Invert Color", "Color", 140.0f, table110, table27, {}, table1, {}, {}, {}, table25},
    {"ShaderNodeLayerWeight", "Layer Weight", "Input", 140.0f, table111, table112, {}, table12, {}, {}, {}, table113},
    {"ShaderNodeLightFalloff", "Light Falloff", "Color", 160.0f, table114, table115, {}, table1, {}, {}, {}, table116},
    {"ShaderNodeLightPath", "Light Path", "Input", 140.0f, {}, table117, {}, table1, {}, {}, {}, table118},
    {"ShaderNodeMapRange", "Map Range", "Math", 140.0f, table119, table120, table123, table1, table125, table126, table127, table124},
    {"ShaderNodeMapping", "Mapping", "Vector", 140.0f, table128, table87, table130, table1, table132, table133, table134, table131},
    {"ShaderNodeMath", "Math", "Math", 140.0f, table135, table96, table137, table1, table138, table139, table140, table30},
    {"ShaderNodeMix", "Mix", "Color", 140.0f, table141, table142, table146, table1, table148, table149, table150, table147},
    {"ShaderNodeMixRGB", "Mix (Legacy)", "", 140.0f, table151, table27, table152, {}, {}, {}, {}, table153},
    {"ShaderNodeMixShader", "Mix Shader", "Shader", 140.0f, table154, table7, {}, table1, {}, {}, {}, table153},
    {"ShaderNodeNewGeometry", "Geometry", "Input", 140.0f, {}, table155, {}, table1, {}, {}, {}, table156},
    {"ShaderNodeNormal", "Normal", "Vector", 140.0f, table157, table158, {}, table1, {}, {}, {}, table159},
    {"ShaderNodeNormalMap", "Normal Map", "Displacement", 160.0f, table160, table23, table164, table1, {}, {}, {}, table28},
    {"ShaderNodeObjectInfo", "Object Info", "Input", 140.0f, {}, table165, {}, table166, {}, {}, {}, table106},
    {"ShaderNodeOutputLight", "Light Output", "Output", 140.0f, table167, {}, table169, table166, {}, {}, {}, {}},
    {"ShaderNodeOutputMaterial", "Material Output", "Output", 140.0f, table170, {}, table169, table12, {}, {}, {}, {}},
    {"ShaderNodeOutputWorld", "World Output", "Output", 140.0f, table171, {}, table169, table20, {}, {}, {}, {}},
    {"ShaderNodeParticleInfo", "Particle Info", "Input", 140.0f, {}, table172, {}, table12, {}, {}, {}, table173},
    {"ShaderNodePointInfo", "Point Info", "Input", 140.0f, {}, table174, {}, table12, {}, {}, {}, table78},
    {"ShaderNodeRGB", "Color", "Input", 140.0f, {}, table175, {}, table1, {}, {}, {}, table21},
    {"ShaderNodeRGBCurve", "RGB Curves", "Color", 240.0f, table176, table27, table177, table1, {}, {}, {}, table25},
    {"ShaderNodeRGBToBW", "RGB to BW", "Color", 140.0f, table175, table178, {}, table1, {}, {}, {}, table28},
    {"ShaderNodeRadialTiling", "Radial Tiling", "Vector", 140.0f, table179, table180, table181, table1, {}, {}, {}, table17},
    {"ShaderNodeRaycast", "Raycast", "Input", 140.0f, table182, table183, table184, table12, {}, {}, {}, table185},
    {"ShaderNodeSeparateColor", "Separate Color", "Color", 140.0f, table27, table186, table85, table1, {}, {}, {}, table187},
    {"ShaderNodeSeparateXYZ", "Separate XYZ", "Vector", 140.0f, table188, table189, {}, table1, {}, {}, {}, table187},
    {"ShaderNodeSubsurfaceScattering", "Subsurface Scattering", "Shader", 160.0f, table190, table191, table192, table12, table193, table194, table195, table21},
    {"ShaderNodeTangent", "Tangent", "Input", 160.0f, {}, table196, table199, table1, {}, {}, {}, table21},
    {"ShaderNodeTexBrick", "Brick Texture", "Texture", 180.0f, table200, table201, table202, table1, {}, {}, {}, table203},
    {"ShaderNodeTexChecker", "Checker Texture", "Texture", 140.0f, table204, table201, {}, table1, {}, {}, {}, table203},
    {"ShaderNodeTexCoord", "Texture Coordinate", "Input", 140.0f, {}, table205, table206, table1, {}, {}, {}, table207},
    {"ShaderNodeTexEnvironment", "Environment Texture", "Texture", 240.0f, table208, table27, table211, table1, {}, {}, {}, table21},
    {"ShaderNodeTexGabor", "Gabor Texture", "Texture", 140.0f, table212, table213, table215, table1, table217, table53, table218, table216},
    {"ShaderNodeTexGradient", "Gradient Texture", "Texture", 140.0f, table208, table201, table220, table1, {}, {}, {}, table221},
    {"ShaderNodeTexIES", "IES Texture", "Texture", 140.0f, table222, table103, table224, table1, {}, {}, {}, table25},
    {"ShaderNodeTexImage", "Image Texture", "Texture", 240.0f, table208, table225, table228, table1, {}, {}, {}, table221},
    {"ShaderNodeTexMagic", "Magic Texture", "Texture", 140.0f, table229, table201, table230, table1, {}, {}, {}, table221},
    {"ShaderNodeTexNoise", "Noise Texture", "Texture", 160.0f, table231, table232, table235, table1, table236, table237, table238, table221},
    {"ShaderNodeTexSky", "Sky Texture", "Texture", 160.0f, table239, table27, table242, table1, table243, table244, table245, table21},
    {"ShaderNodeTexVoronoi", "Voronoi Texture", "Texture", 160.0f, table246, table247, table250, table1, table252, table253, table254, table251},
    {"ShaderNodeTexWave", "Wave Texture", "Texture", 160.0f, table255, table201, table260, table1, {}, {}, {}, table221},
    {"ShaderNodeTexWhiteNoise", "White Noise Texture", "Texture", 140.0f, table261, table262, table263, table1, table265, table266, table267, table264},
    {"ShaderNodeUVMap", "UV Map", "Input", 160.0f, {}, table268, table269, table12, {}, {}, {}, table21},
    {"ShaderNodeValToRGB", "Color Ramp", "Color", 240.0f, table270, table225, table274, table1, {}, {}, {}, table159},
    {"ShaderNodeValue", "Value", "Input", 140.0f, {}, table96, {}, table1, {}, {}, {}, table21},
    {"ShaderNodeVectorCurve", "Vector Curves", "Vector", 240.0f, table275, table87, table278, table1, {}, {}, {}, table25},
    {"ShaderNodeVectorDisplacement", "Vector Displacement", "Displacement", 140.0f, table279, table89, table281, table1, {}, {}, {}, table153},
    {"ShaderNodeVectorMath", "Vector Math", "Vector", 140.0f, table282, table283, table285, table1, table138, table287, table288, table286},
    {"ShaderNodeVectorRotate", "Vector Rotate", "Vector", 140.0f, table289, table87, table291, table1, table293, table294, table295, table292},
    {"ShaderNodeVectorTransform", "Vector Transform", "Vector", 140.0f, table296, table87, table299, table1, {}, {}, {}, table28},
    {"ShaderNodeVertexColor", "Color Attribute", "Input", 140.0f, {}, table225, table300, table12, {}, {}, {}, table221},
    {"ShaderNodeVolumeInfo", "Volume Info", "Input", 140.0f, {}, table301, {}, table12, {}, {}, {}, table17},
    {"ShaderNodeWavelength", "Wavelength", "Color", 160.0f, table302, table27, {}, table1, {}, {}, {}, table28},
    {"ShaderNodeWireframe", "Wireframe", "Input", 140.0f, table303, table103, table304, table12, {}, {}, {}, table28}};

constexpr NodeTreeDefault defaults[] = {{NodeTreeKind::Material, "Shader Nodetree", table305, table306, {}, table308},
                                        {NodeTreeKind::Light, "Shader Nodetree", table309, table306, {}, table311},
                                        {NodeTreeKind::World, "World Nodetree", table312, table306, table313, table314}};

}

std::span<const NodeTypeData> nodeTypeData() {
    return types;
}

std::span<const NodeTreeDefault> nodeTreeDefaults() {
    return defaults;
}

}
