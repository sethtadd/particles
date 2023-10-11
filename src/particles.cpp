#include <iostream>
#include <vector>

#include <glad/gl.h> // Include first to avoid errors
#include <GLFW/glfw3.h>

#include "Shader.hpp"
#include "CudaKernels.hpp"

const uint WIDTH = 828;
const uint HEIGHT = 512;

void glfwErrorCallback(int error, const char *description);

void handleInput(GLFWwindow *window);

// Callback functions
void mouse_callback(GLFWwindow *window, double xPos, double yPos);               // Mouse movement
void scroll_callback(GLFWwindow *window, double xOffset, double yOffset);        // Zooming in/out
void framebuffer_size_callback(GLFWwindow *window, int newWidth, int newHeight); // Handle window resizing

int main()
{
    glfwSetErrorCallback(glfwErrorCallback);
    if (!glfwInit())
        return -1;

    GLFWwindow *window = glfwCreateWindow(WIDTH, HEIGHT, "Hello World", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    // set callback functions
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // Initialize GLAD2
    int version = gladLoadGL(glfwGetProcAddress);

    // Print version info
    std::cout << "GLAD2 GL version: " << GLAD_VERSION_MAJOR(version) << "." << GLAD_VERSION_MINOR(version) << std::endl;
    std::cout << "OpenGL version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "Renderer: " << glGetString(GL_RENDERER) << std::endl;
    std::cout << "Vendor: " << glGetString(GL_VENDOR) << std::endl;

    glViewport(0, 0, WIDTH, HEIGHT);

    Shader shader("shaders/particles.vertex.glsl", "shaders/particles.geometry.glsl", "shaders/particles.fragment.glsl");

    float vertices[] = {
        0.0f, 0.0f, 0.0f,      // Position (x, y, z)
        1.0f, 1.0f, 1.0f, 1.0f // Color (r, g, b, a)
    };

    // Particle offsets
    float instanceData[] = {
        0.5f, 0.5f, 0.0f,   // Top right
        0.5f, -0.5f, 0.0f,  // Bottom right
        -0.5f, -0.5f, 0.0f, // Bottom left
        -0.5f, 0.5f, 0.0f   // Top left
    };
    int numParticles = sizeof(instanceData) / (3 * sizeof(float));
    std::cout << "Particle count: " << numParticles << std::endl;

    // Instance particles VAO
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    // Singular particle vertex data
    GLuint particleVbo;
    glGenBuffers(1, &particleVbo);
    glBindBuffer(GL_ARRAY_BUFFER, particleVbo);
    glBufferData(
        GL_ARRAY_BUFFER,  // target
        sizeof(vertices), // size
        vertices,         // data
        GL_STATIC_DRAW);  // usage

    // Position attribute
    glVertexAttribPointer(
        0,          // index
        3,          // size
        GL_FLOAT,   // type
        GL_FALSE,   // normalized
        0,          // stride
        (void *)0); // pointer
    glEnableVertexAttribArray(0);

    // Color attribute
    glVertexAttribPointer(
        1,                            // index
        4,                            // size
        GL_FLOAT,                     // type
        GL_FALSE,                     // normalized
        0,                            // stride
        (void *)(3 * sizeof(float))); // pointer
    glEnableVertexAttribArray(1);

    // Instance data
    GLuint instanceVbo;
    glGenBuffers(1, &instanceVbo);
    glBindBuffer(GL_ARRAY_BUFFER, instanceVbo);
    glBufferData(
        GL_ARRAY_BUFFER,      // target
        sizeof(instanceData), // size
        instanceData,         // data
        GL_STATIC_DRAW);      // usage

    // Offset attribute
    glVertexAttribPointer(
        2,                 // index
        3,                 // size
        GL_FLOAT,          // type
        GL_FALSE,          // normalized
        3 * sizeof(float), // stride
        (void *)0);        // pointer
    glEnableVertexAttribArray(2);
    glVertexAttribDivisor(2, 1); // Update attribute every 1 instance

    // Unbind particleVbo and vao
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    while (!glfwWindowShouldClose(window))
    {
        glClearColor(0.53332607f, 0.4382106f, 0.72355703f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        shader.use();

        // Draw particles
        glBindVertexArray(vao);
        glDrawArraysInstanced(
            GL_POINTS,     // mode: type of primitives to render
            0,             // first: starting index
            1,             // count: number of indicies to render
            numParticles); // instance count
        glBindVertexArray(0);

        handleInput(window);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // CUDA test after glfw termination
    // --------------------------------
    std::cout << " ----- CUDA test ----- " << std::endl;

    const int numElements = 15;

    std::vector<float> A(numElements);
    std::vector<float> B(numElements);
    std::vector<float> C(numElements);

    // Initialize host memory with random data
    for (int i = 0; i < numElements; i++)
    {
        A[i] = rand() / (float)RAND_MAX;
        B[i] = rand() / (float)RAND_MAX;
    }
    runVectorAdd(A.data(), B.data(), C.data(), numElements);
    for (int i = 0; i < numElements; ++i)
    {
        std::cout << A[i] << " + " << B[i] << " = " << C[i] << std::endl;
    }

    // Clean up
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &particleVbo);
    glDeleteBuffers(1, &instanceVbo);
    glfwTerminate();
    return 0;
}

void glfwErrorCallback(int error, const char *description)
{
    std::cerr << "GLFW Error: " << error << " - " << description << std::endl;
}

void handleInput(GLFWwindow *window)
{
    // exit program
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}

void mouse_callback(GLFWwindow *window, double xPos, double yPos)
{
    // TODO Implement
    // std::cout << "Mouse position: (" << xPos << ", " << yPos << ")" << std::endl;
}

void scroll_callback(GLFWwindow *window, double xOffset, double yOffset)
{
    std::cout << "Scroll offset: (" << xOffset << ", " << yOffset << ")" << std::endl;
}

void framebuffer_size_callback(GLFWwindow *window, int newWidth, int newHeight)
{
    glViewport(0, 0, newWidth, newHeight);
}
