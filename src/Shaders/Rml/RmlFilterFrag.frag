#version 450

// Filter Type Definitions 
// These must match the C++ defines
#define FILTER_TYPE_OPACITY             0
#define FILTER_TYPE_COLOR_MATRIX        1
#define FILTER_TYPE_BLUR                2
#define FILTER_TYPE_DROP_SHADOW_ALPHA   3
#define BLUR_NUM_WEIGHTS                8 // Increased to match C++

// Inputs / Outputs 
layout(location = 0) in vec2 in_uv;
layout(location = 0) out vec4 out_color;

// Push Constants 
// For data that changes on every draw call (e.g., between vertical and horizontal blur passes)
layout(push_constant) uniform PushConstants {
    int filter_type;
    vec2 blur_direction;
    vec2 texel_size;
} pc;

// UBO (Uniform Buffer Object) 
// For data that is constant for a specific filter instance
layout(std140, set = 0, binding = 0) uniform UniformBufferObject {
    float scalar_value;
    mat4 color_matrix;
    float blur_weights[BLUR_NUM_WEIGHTS];
    vec2 drop_shadow_offset;
    vec4 drop_shadow_color;
} ubo;

// Input Texture 
// The image from the previous pass or layer
layout(set = 1, binding = 0) uniform sampler2D in_texture;

void main()
{
    // The initial color from the input texture
    vec4 color = texture(in_texture, in_uv);

    // Filter Logic 
    // The filter to run is now selected using the push constant
    if (pc.filter_type == FILTER_TYPE_COLOR_MATRIX)
    {
        // Apply the color transformation matrix from the UBO
        vec3 transformed_rgb = (ubo.color_matrix * color).rgb;
        color = vec4(transformed_rgb, color.a);
    }
    else if (pc.filter_type == FILTER_TYPE_BLUR)
    {
        // Perform a Gaussian blur using weights from the UBO and direction from the push constant
        vec2 blur_offset = pc.blur_direction * pc.texel_size;
        vec4 blurred_color = color * ubo.blur_weights[0]; // Center tap

        // Sample neighboring texels
        for (int i = 1; i < BLUR_NUM_WEIGHTS; ++i) {
            blurred_color += texture(in_texture, in_uv + blur_offset * float(i)) * ubo.blur_weights[i];
            blurred_color += texture(in_texture, in_uv - blur_offset * float(i)) * ubo.blur_weights[i];
        }

        color = blurred_color;
    }
    else if (pc.filter_type == FILTER_TYPE_OPACITY)
    {
        // Apply opacity using the scalar value from the UBO
        color *= ubo.scalar_value;
    }
    else if (pc.filter_type == FILTER_TYPE_DROP_SHADOW_ALPHA)
    {
        // Create the shadow shape using offset and color from the UBO
        vec2 offset_uv = ubo.drop_shadow_offset * pc.texel_size;
        float source_alpha = texture(in_texture, in_uv - offset_uv).a;
        color = ubo.drop_shadow_color * source_alpha;
    }

    out_color = color;
}