#version 450

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec4 inColor0; 
layout(location = 2) in vec2 inTexCoord0;

// Output to fragment shader
layout(location = 0) out vec2 fragTexCoord;

void main()
{
    fragTexCoord = inTexCoord0;
    gl_Position = vec4(inPosition, 0.0, 1.0);
}