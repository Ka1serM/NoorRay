#version 450
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) in vec4 in_color;
layout(location = 1) in vec2 in_tex_coord;

layout(location = 0) out vec4 out_color;

layout(push_constant) uniform PushConstants {
    mat4 transform;
    vec2 translation;
    int texture_id;
    int padding0;
} push_constants;

// Defines a single color stop for a gradient
struct ColorStop {
    vec4 color;
    float position;
};

// Gradient UBO at set = 0
layout(std140, set = 0, binding = 0) uniform GradientUBO {
    int gradient_function; // 0=linear, 1=radial, 2=conic, 3+=repeating
    int num_stops;
    // vec2 padding
    vec2 p;   // start point / center
    vec2 v;   // direction / rotation vector
    ColorStop stops[16];
} gradient;

// Bindless texture array at set = 1
layout(set = 1, binding = 0) uniform sampler2D textures[];

// Interpolates the final color based on the calculated gradient value 't'
vec4 mix_stop_colors(float t) {
    vec4 color = gradient.stops[0].color;
    // Mix between adjacent color stops
    for (int i = 1; i < gradient.num_stops; i++) {
        color = mix(color, gradient.stops[i].color, smoothstep(gradient.stops[i-1].position, gradient.stops[i].position, t));
    }
    return color;
}

void main() {
    vec4 color = in_color;
    float t = 0.0;
    const float PI = 3.14159265359;

    // --- Calculate Gradient Value 't' ---
    if (gradient.gradient_function == 0 || gradient.gradient_function == 3) { // Linear / repeating-linear
                                                                              float len2 = dot(gradient.v, gradient.v);
                                                                              vec2 V = in_tex_coord - gradient.p;
                                                                              t = dot(gradient.v, V) / len2;
    } else if (gradient.gradient_function == 1 || gradient.gradient_function == 4) { // Radial / repeating-radial
                                                                                     vec2 V = in_tex_coord - gradient.p;
                                                                                     // v.x = 1/radius_x^2, v.y = 1/radius_y^2 for elliptical gradient
                                                                                     t = sqrt(dot(V * V, gradient.v));
    } else if (gradient.gradient_function == 2 || gradient.gradient_function == 5) { // Conic / repeating-conic
                                                                                     // v = (cos(angle), sin(angle))
                                                                                     mat2 R = mat2(gradient.v.x, -gradient.v.y, gradient.v.y, gradient.v.x);
                                                                                     vec2 V = R * (in_tex_coord - gradient.p);
                                                                                     t = 0.5 + atan(-V.x, V.y) / (2.0 * PI);
    }

    // --- Handle Repeating Gradients ---
    if (gradient.gradient_function >= 3) {
        float t0 = gradient.stops[0].position;
        float t1 = gradient.stops[gradient.num_stops - 1].position;
        if (t1 > t0)
        t = t0 + mod(t - t0, t1 - t0);
    }

    // --- Final Color Calculation ---
    color *= mix_stop_colors(t);

    if (push_constants.texture_id >= 0) {
        color *= texture(textures[nonuniformEXT(push_constants.texture_id)], in_tex_coord);
    }

    out_color = color;
}