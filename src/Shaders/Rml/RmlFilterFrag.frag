#version 450

#define FILTER_TYPE_OPACITY             0
#define FILTER_TYPE_COLOR_MATRIX        1
#define FILTER_TYPE_BLUR_VERTICAL       2
#define FILTER_TYPE_BLUR_HORIZONTAL     3
#define FILTER_TYPE_DROP_SHADOW_ALPHA   4

// --- Inputs / Outputs ---
layout(location = 0) in vec2 in_uv;
layout(location = 0) out vec4 out_color;

// --- Descriptor Sets ---
// Set 0: The unified UBO containing all filter parameters
layout(std140, set = 0, binding = 0) uniform UniformBufferObject {
    // This layout MUST exactly match your C++ FilterData struct
    int filter_type;
    float scalar_value;
    vec2 drop_shadow_offset;

    vec4 drop_shadow_color;

    // Assumes BLUR_NUM_WEIGHTS from the GL renderer is 4
    float blur_weights[4];

    mat4 color_matrix;
} ubo;

// Set 1: The input texture from the previous pass or layer
layout(set = 1, binding = 0) uniform sampler2D in_texture;

void main()
{
    // The initial color from the input texture
    vec4 color = texture(in_texture, in_uv);

    // --- Filter Logic ---
    // The C++ side sets 'ubo.filter_type' to select which logic to run.
    if (ubo.filter_type == FILTER_TYPE_COLOR_MATRIX)
    {
        // Apply the 4x4 color transformation matrix.
        // The final column of the matrix provides a constant offset,
        // which must be multiplied by the alpha for premultiplied colors.
        vec4 transformed_color = ubo.color_matrix * color;
        transformed_color.rgb += ubo.color_matrix[3].rgb * color.a;
        color = transformed_color;
    }
    else if (ubo.filter_type == FILTER_TYPE_BLUR_VERTICAL || ubo.filter_type == FILTER_TYPE_BLUR_HORIZONTAL)
    {
        // Perform a 7-tap Gaussian blur in one direction.
        // Your C++ code must run this shader twice (once vertical, once horizontal) for a full blur.
        vec2 texel_size = 1.0 / vec2(textureSize(in_texture, 0));
        vec2 blur_dir = (ubo.filter_type == FILTER_TYPE_BLUR_VERTICAL)  ? vec2(0.0, texel_size.y)  : vec2(texel_size.x, 0.0);

        // Center tap
        vec4 blurred_color = texture(in_texture, in_uv) * ubo.blur_weights[0];

        // Sample neighboring texels
        for (int i = 1; i < 4; ++i) {
            blurred_color += texture(in_texture, in_uv + blur_dir * float(i)) * ubo.blur_weights[i];
            blurred_color += texture(in_texture, in_uv - blur_dir * float(i)) * ubo.blur_weights[i];
        }

        color = blurred_color;
    }
    else if (ubo.filter_type == FILTER_TYPE_OPACITY)
    {
        // Apply opacity. Since we use premultiplied alpha, we multiply all components.
        color *= ubo.scalar_value;
    }
    else if (ubo.filter_type == FILTER_TYPE_DROP_SHADOW_ALPHA)
    {
        // Create the shadow shape by sampling the source texture with an offset.
        vec2 texel_size = 1.0 / vec2(textureSize(in_texture, 0));
        vec2 offset_uv = ubo.drop_shadow_offset * texel_size;

        // The shadow's color is the specified drop-shadow color, and its alpha
        // is the alpha of the original shape at the offsetted position.
        float source_alpha = texture(in_texture, in_uv - offset_uv).a;
        color = ubo.drop_shadow_color * source_alpha;
    }

    out_color = color;
}