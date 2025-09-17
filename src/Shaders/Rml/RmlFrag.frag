#version 450

#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) in vec4 in_color;
layout(location = 1) in vec2 in_tex_coord;

layout(location = 0) out vec4 out_color;

layout(push_constant) uniform PushConstants {
    mat4 transform;
    vec2 translation;
    int texture_id;
} push_constants;

// Your bindless texture array
layout(set = 0, binding = 1) uniform sampler2D textures[];

void main() {
    vec4 color = in_color;
    if (push_constants.texture_id >= 0)
        color *= texture(textures[push_constants.texture_id], in_tex_coord);
    out_color = color;
}