#version 460 core

in vec4 vertexColor;
in vec2 TexCoords;

in vec4 geomColor;
out vec4 FragColor;

void main()
{
    float distance = length(TexCoords - vec2(0.5, 0.5));
    float fade = 1.0 - smoothstep(0.0, 0.5, distance);
    // vec3 color = vec3(0.3, 0.8, 0.95);
    // vec3 color = vec3(1.0, 1.0, 1.0);
    vec3 color = geomColor.rgb;
    color *= fade;  // Darken color further away from center
    FragColor = vec4(color, fade);
}
