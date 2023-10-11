#version 460 core

in vec2 TexCoords;
out vec4 FragColor;

void main()
{
    float distance = length(TexCoords - vec2(0.5, 0.5));
    float glow = smoothstep(0.0, 0.5, distance);
    vec3 color = vec3(0.3, 0.8, 0.95);
    color *= pow(1.0 - glow, 1);  // Darken color further away from center
    FragColor = vec4(color, 1.0 - glow);
}
