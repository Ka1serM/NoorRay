#version 450

// Input from the vertex shader
layout(location = 0) in vec2 fragTexCoord;

// Final output color
layout(location = 0) out vec4 out_color;

// The input texture from the previous layer or post-process buffer.
// The set/binding must match your m_passthrough_layout.
layout(set = 0, binding = 0) uniform sampler2D in_texture;

void main()
{
    // Sample the texture and write it to the output.
    out_color = texture(in_texture, fragTexCoord);
}