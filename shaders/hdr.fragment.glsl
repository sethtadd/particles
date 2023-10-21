#version 460 core

in vec2 TexCoord;

out vec4 FragColor;

uniform sampler2D hdrBuffer;

void main()
{
    const float gamma = 2.2;
    const float exposure = 1.0;
    vec3 hdrColor = texture(hdrBuffer, TexCoord).rgb;

    // Reinhard tone mapping
    hdrColor /= (hdrColor + vec3(1.0));
    // Exposure tone mapping
    // hdrColor = vec3(1.0) - exp(-hdrColor * exposure);
    // Gamma correction
    hdrColor = pow(hdrColor, vec3(1.0 / gamma));

    FragColor = vec4(hdrColor, 1.0);
}
