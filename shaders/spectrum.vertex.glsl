#version 460 core

layout (location = 0) in vec2 aPos;

uniform mat3 transform;

void main()
{
    vec3 transformedPos = transform * vec3(aPos, 1.0);
    gl_Position = vec4(transformedPos.xy, -0.1, 1.0);
    
    // vec2 transformedPos = (aPos.xy - vec2(0.5, 0.5));
    // gl_Position = vec4(transformedPos, -0.1, 1.0);

    // gl_Position = vec4(aPos, -0.1, 1.0);
}
