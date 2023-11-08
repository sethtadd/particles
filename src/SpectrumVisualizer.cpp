#include <vector>
#include <iostream>
#include <glad/gl.h>
#include <glm/glm.hpp>

#include "SpectrumVisualizer.hpp"
#include "Shader.hpp"

/**
 * @brief Positions and scales the spectrum visualizer. By default, the spectrum visualizer's bottom left corner is at (0, 0) with scale (1, 1). To make the visualizer take up the whole space (NDC is [-1, 1]), use setTransform(-1, -1, 2, 2).
 *
 * @param xPos
 * @param yPos
 * @param xScale
 * @param yScale
 */
void SpectrumVisualizer::setTransform(float xPos, float yPos, float xScale, float yScale)
{
    glm::mat3 transform = glm::mat3(1.0f); // Identity matrix

    // Apply the scale
    transform[0][0] = xScale;
    transform[1][1] = yScale;

    // Translate to move the origin to the bottom-left corner
    transform[0][2] = xPos; // Move to the left edge
    transform[1][2] = yPos; // Move to the bottom edge

    // OpenGL is column-major, so we need to transpose the matrix
    transform_ = glm::transpose(transform);
}

SpectrumVisualizer::SpectrumVisualizer(const Shader &shaderProgram)
    : shader(shaderProgram)
{

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
}

SpectrumVisualizer::~SpectrumVisualizer()
{
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
}

void SpectrumVisualizer::setMagnitudes(const std::vector<float> &newMagnitudes)
{
    magnitudes = newMagnitudes;
}

void SpectrumVisualizer::draw()
{
    if (magnitudes.empty())
        return;

    glBindVertexArray(VAO);

    std::vector<float> vertices;

    float barWidth = 1.0f / (1.0f * magnitudes.size());

    float x = 0.0f;
    for (const auto &magnitude : magnitudes)
    {
        float normalizedMagnitude = magnitude > 1.0f ? 1.0f : magnitude;
        // Append quad vertices to vertices vector
        vertices.insert(vertices.end(), {x, 0.0f,
                                         x + barWidth, 0.0f,
                                         x + barWidth, normalizedMagnitude,
                                         x, normalizedMagnitude});

        x += barWidth;
    }

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertices.size(), &vertices[0], GL_DYNAMIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (GLvoid *)0);
    glEnableVertexAttribArray(0);

    shader.use();
    shader.setMatrix3f("transform", transform_);

    glDrawArrays(GL_QUADS, 0, (GLsizei)vertices.size() / 2);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}
