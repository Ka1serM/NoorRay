#version 450
#extension GL_EXT_nonuniform_qualifier : require

// Inputs from vertex shader
layout(location = 0) in vec4 in_color;
layout(location = 1) in vec2 in_tex_coord;

// Output color
layout(location = 0) out vec4 out_color;

layout(push_constant) uniform PushConstants {
    mat4 transform;
    vec2 translate;
    int texture_id;
} push_constants;

// Bindless texture array (at set 0)
layout(set = 0, binding = 0) uniform sampler2D textures[];

struct Stop {
    float position;
    vec3 _padding;
    vec4 color;
};
        
// Gradient UBO (at set 1)
layout(set = 1, binding = 0) uniform GradientUBO {
    int gradient_function; // 0: linear, 1: radial, 2: conic, +3 for repeating
    int num_stops;
    vec2 p, v;
    Stop stops[16];
} gradient;

vec4 mix_stop_colors(float t) {
    vec4 color = gradient.stops[0].color;
    // We can't use a dynamic loop limit, so loop up to the max.
    for (int i = 1; i < 16; i++) {
        if (i >= gradient.num_stops) break;
        color = mix(color, gradient.stops[i].color, smoothstep(gradient.stops[i-1].position, gradient.stops[i].position, t));
    }
    return color;
}

void main() {
    float t = 0.0;

    int func = gradient.gradient_function % 3;
    bool repeating = gradient.gradient_function >= 3;

    if (func == 0) { // linear
                     float dist_sq = dot(gradient.v, gradient.v);
                     if (dist_sq > 0.0) {
                         t = dot(in_tex_coord - gradient.p, gradient.v) / dist_sq;
                     }
    } else if (func == 1) { // radial
                            vec2 V = in_tex_coord - gradient.p;
                            t = length(gradient.v * V);
    } else if (func == 2) { // conic
                            const float PI2 = 6.28318530718;
                            mat2 R = mat2(gradient.v.x, -gradient.v.y, gradient.v.y, gradient.v.x);
                            vec2 V = R * (in_tex_coord - gradient.p);
                            t = 0.5 + atan(-V.x, V.y) / PI2;
    }

    if (repeating && gradient.num_stops > 1) {
        float t0 = gradient.stops[0].position;
        float t1 = gradient.stops[gradient.num_stops - 1].position;
        float range = t1 - t0;
        if (range > 0.0) {
            t = t0 + mod(t - t0, range);
        }
    }

    vec4 gradient_color = mix_stop_colors(t);

    vec4 final_color = in_color * gradient_color;

    // Apply texture if one is provided
    if (push_constants.texture_id >= 0) {
        final_color *= texture(textures[nonuniformEXT(push_constants.texture_id)], in_tex_coord);
    }

    out_color = final_color;
}

