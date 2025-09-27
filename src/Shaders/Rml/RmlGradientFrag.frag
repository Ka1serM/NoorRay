#version 450
#extension GL_EXT_nonuniform_qualifier : require

#define PI                3.14159265359
#define MAX_STOPS         16

// Gradient function types
#define GRADIENT_LINEAR           0
#define GRADIENT_RADIAL           1
#define GRADIENT_CONIC            2

layout(location = 0) in vec4 in_color;
layout(location = 1) in vec2 in_tex_coord;

layout(location = 0) out vec4 out_color;

layout(push_constant) uniform PushConstants {
    mat4 transform;
    vec2 translation;
    int texture_id;
} push_constants;

// Defines a single color stop for a gradient
struct ColorStop {
    vec4 color;
    float position;
};

// Gradient UBO
layout(std140, set = 0, binding = 0) uniform GradientUBO {
    int gradient_function; int repeating;
    int num_stops; int pad0; // Padding to align to vec4
    vec2 p;
    vec2 v;
    ColorStop stops[MAX_STOPS];
} gradient;

// Bindless texture array
layout(set = 1, binding = 0) uniform sampler2D textures[];

vec4 mix_stop_colors(float t) {
    vec4 color = gradient.stops[0].color;
    for (int i = 1; i < gradient.num_stops; i++)
        color = mix(color, gradient.stops[i].color, smoothstep(gradient.stops[i-1].position, gradient.stops[i].position, t));
    return color;
}

void main() {
    vec4 color = in_color;
    float t = 0.0;

    vec2 tex_coord = in_tex_coord;
    vec2 delta = tex_coord - gradient.p;

    // Calculate Gradient Value 't'
    if (gradient.gradient_function == GRADIENT_LINEAR) {
        float len2 = dot(gradient.v, gradient.v);
        if (len2 > 1.0e-6)
            t = dot(delta, gradient.v) / len2;
        
    } else if (gradient.gradient_function == GRADIENT_RADIAL) {
        t = length(gradient.v * delta);
        
    } else if (gradient.gradient_function == GRADIENT_CONIC) {
        mat2 R = mat2(gradient.v.x, gradient.v.y, -gradient.v.y, gradient.v.x);
        vec2 rotated_delta = R * delta;
        float angle_rad = atan(rotated_delta.y, rotated_delta.x);
        t = mod(1.0 - angle_rad / (2.0 * PI), 1.0);
    }

    if (bool(gradient.repeating)) {
        float t0 = gradient.stops[0].position;
        float t1 = gradient.stops[gradient.num_stops - 1].position;
        if (t1 > t0)
            t = t0 + mod(t - t0, t1 - t0);
    }

    color *= mix_stop_colors(t);

    if (push_constants.texture_id >= 0)
        color *= texture(textures[push_constants.texture_id], tex_coord);

    out_color = color;
}