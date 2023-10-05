#include <string>
#include <fstream>
#include <sstream>
#include <cstdio>

#include "glad/gl.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Shader.hpp"

// Constructor
Shader::Shader(const char *vertexPath, const char *geometryPath, const char *fragmentPath)
{
    std::ifstream vertexShaderFile;
    std::ifstream geometryShaderFile;
    std::ifstream fragmentShaderFile;

    std::stringstream vertexShaderStream;
    std::stringstream geometryShaderStream;
    std::stringstream fragmentShaderStream;

    std::string vertexSourceCodeBuffer;
    std::string geometrySourceCodeBuffer;
    std::string fragmentSourceCodeBuffer;

    // Make sure ifstream objects can throw exceptions
    vertexShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    geometryShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    fragmentShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    try
    {
        // Open shader source code files
        vertexShaderFile.open(vertexPath);
        geometryShaderFile.open(geometryPath);
        fragmentShaderFile.open(fragmentPath);

        // Read files' buffer contents into string streams
        vertexShaderStream << vertexShaderFile.rdbuf();
        geometryShaderStream << geometryShaderFile.rdbuf();
        fragmentShaderStream << fragmentShaderFile.rdbuf();

        // Close file handlers
        vertexShaderFile.close();
        geometryShaderFile.close();
        fragmentShaderFile.close();

        // Send string stream data to string buffers in string format
        vertexSourceCodeBuffer = vertexShaderStream.str();
        geometrySourceCodeBuffer = geometryShaderStream.str();
        fragmentSourceCodeBuffer = fragmentShaderStream.str();
    }
    catch (std::ifstream::failure &e)
    {
        fprintf(stderr, "ERROR::SHADER::FILE_NOT_SUCCESSFULLY_READ: %s\n", e.what());
    }

    // Send string buffers' content to char*'s as c_strings
    const char *vertexSourceCode = vertexSourceCodeBuffer.c_str();
    const char *geometrySourceCode = geometrySourceCodeBuffer.c_str();
    const char *fragmentSourceCode = fragmentSourceCodeBuffer.c_str();

    // ---------- Vertex Shader ---------- //
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexSourceCode, NULL);
    glCompileShader(vertexShader);
    // Error checking
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        printf("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n%s\n", infoLog);
    }

    // ---------- Geometry Shader ---------- //
    unsigned int geometryShader = glCreateShader(GL_GEOMETRY_SHADER);
    glShaderSource(geometryShader, 1, &geometrySourceCode, NULL);
    glCompileShader(geometryShader);
    // Error checking
    glGetShaderiv(geometryShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(geometryShader, 512, NULL, infoLog);
        printf("ERROR::SHADER::GEOMETRY::COMPILATION_FAILED\n%s\n", infoLog);
    }

    // ---------- Fragment Shader ---------- //
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentSourceCode, NULL);
    glCompileShader(fragmentShader);
    // Error checking
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        printf("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n%s\n", infoLog);
    }

    // ---------- Shader Program ---------- //
    shaderProgramId = glCreateProgram();
    glAttachShader(shaderProgramId, vertexShader);
    glAttachShader(shaderProgramId, geometryShader);
    glAttachShader(shaderProgramId, fragmentShader);
    glLinkProgram(shaderProgramId);

    // Error checking
    glGetProgramiv(shaderProgramId, GL_LINK_STATUS, &success);
    if (!success)
    {
        glGetProgramInfoLog(shaderProgramId, 512, NULL, infoLog);
        printf("ERROR::SHADER::PROGRAM::LINKING_FAILED\n%s\n", infoLog);
    }

    // Clean up
    glDeleteShader(vertexShader);
    glDeleteShader(geometryShader);
    glDeleteShader(fragmentShader);
}

// Set shader to current
void Shader::use()
{
    glUseProgram(shaderProgramId);
}

// Set shader uniforms
// -------------------
void Shader::setBool(const std::string &name, bool value) const
{
    glUniform1i(glGetUniformLocation(shaderProgramId, name.c_str()), (int)value);
}

void Shader::setInt(const std::string &name, int value) const
{
    glUniform1i(glGetUniformLocation(shaderProgramId, name.c_str()), value);
}

void Shader::setFloat(const std::string &name, float value) const
{
    glUniform1f(glGetUniformLocation(shaderProgramId, name.c_str()), value);
}

void Shader::setFloat3(const std::string &name, float value1, float value2, float value3) const
{
    glUniform3f(glGetUniformLocation(shaderProgramId, name.c_str()), value1, value2, value3);
}
void Shader::setFloat3(const std::string &name, glm::vec3 val) const
{
    glUniform3f(glGetUniformLocation(shaderProgramId, name.c_str()), val.x, val.y, val.z);
}

void Shader::setFloat4(const std::string &name, float value1, float value2, float value3, float value4) const
{
    glUniform4f(glGetUniformLocation(shaderProgramId, name.c_str()), value1, value2, value3, value4);
}

void Shader::setMatrix3f(const std::string &name, glm::mat3 matrix) const
{
    glUniformMatrix3fv(glGetUniformLocation(shaderProgramId, name.c_str()), 1, GL_FALSE, glm::value_ptr(matrix));
}

void Shader::setMatrix4f(const std::string &name, glm::mat4 matrix) const
{
    glUniformMatrix4fv(glGetUniformLocation(shaderProgramId, name.c_str()), 1, GL_FALSE, glm::value_ptr(matrix));
}

void Shader::setDouble(const std::string &name, double value) const
{
    glUniform1d(glGetUniformLocation(shaderProgramId, name.c_str()), value);
}