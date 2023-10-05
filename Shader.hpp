#ifndef SHADER_HPP
#define SHADER_HPP

#include <fstream>
#include <sstream>

#include "glad/gl.h"
#include <glm/glm.hpp>

// Shader handles the loading-in and compilation of shader source code (written in GLSL). This is essentially a wrapper class around the shader object stored in GPU memory
// See relevant source file for function descriptions
class Shader
{
public:
    Shader(const char *vertexPath, const char *geometryPath, const char *fragmentPath); // Constructor: Read in shader code files and build shader program

    void use(); // Tells the OpenGL state machine to use (this)Shader

    // Shader uniform setting functions (sets uniforms on the shader object stored in GPU memory)
    void setBool(const std::string &name, bool value) const; // the addition of const means the function cannot modify the state of the (this) object
    void setInt(const std::string &name, int value) const;
    void setFloat(const std::string &name, float value) const;
    void setFloat3(const std::string &name, float value1, float value2, float value3) const;
    void setFloat3(const std::string &name, glm::vec3 val) const;
    void setFloat4(const std::string &name, float value1, float value2, float value3, float value4) const;
    void setMatrix3f(const std::string &name, glm::mat3 matrix) const;
    void setMatrix4f(const std::string &name, glm::mat4 matrix) const;

    void setDouble(const std::string &name, double value) const;

private:
    unsigned int shaderProgramId; // Shader program ID for referencing object stored on video card

    // Error handling fields
    int success;
    char infoLog[512];
};

#endif // SHADER_HPP
