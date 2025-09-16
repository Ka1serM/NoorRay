#version 450

// Input attributes matching the Rml::Vertex struct
layout(location = 0) in vec2 in_pos;
layout(location = 1) in vec4 in_color;
layout(location = 2) in vec2 in_tex_coord;

// Outputs passed to the fragment shader
layout(location = 0) out vec4 out_color;
layout(location = 1) out vec2 out_tex_coord;

// Push constants for per-draw data
layout(push_constant) uniform PushConsts {
    mat4 transform;
    vec2 translate;
    int texture_ids; // Unused in vertex shader, but part of the struct
} push_constants;

void main() {
    // Apply translation and then the full transform matrix
    gl_Position = push_constants.transform * vec4(in_pos + push_constants.translate, 0.0, 1.0);

    // Pass-through color and texture coordinates
    out_color = in_color;
    out_tex_coord = in_tex_coord;
}