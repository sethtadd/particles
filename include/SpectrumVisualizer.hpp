#ifndef SPECTRUMVISUALIZER_HPP
#define SPECTRUMVISUALIZER_HPP

#include <vector>
#include <glad/gl.h>
#include <glm/glm.hpp>

#include "Shader.hpp"

class SpectrumVisualizer
{
private:
    std::vector<float> magnitudes;
    GLuint VAO, VBO;
    Shader shader;
    glm::mat3 transform_{glm::mat3(1.0f)};

public:
    SpectrumVisualizer(const Shader &shaderProgram);
    ~SpectrumVisualizer();

    void setTransform(float xPos, float yPos, float xScale, float yScale);
    void setMagnitudes(const std::vector<float> &newMagnitudes);
    void draw();
};

#endif // SPECTRUMVISUALIZER_HPP
