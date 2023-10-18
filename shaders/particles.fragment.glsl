#version 460 core

in vec4 vertexColor;
in vec2 TexCoords;

in vec4 geomColor;
out vec4 FragColor;

void main()
{
    float distance = length(TexCoords - vec2(0.5, 0.5));
    float fade = 1.0 - smoothstep(0.0, 0.5, distance);
    FragColor = vec4(geomColor * fade); // Darken color further away from center and fade to transparent
}
