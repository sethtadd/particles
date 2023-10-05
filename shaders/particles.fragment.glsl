#version 460 core

in vec2 TexCoords;
out vec4 FragColor;

void main()
{
    float distance = length(TexCoords - vec2(0.5, 0.5));
    float glow = smoothstep(0.2, 0.4, distance);
    FragColor = vec4(vec3(glow), 1.0 - glow);
}
