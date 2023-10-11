#include <iostream>
#include <vector>

#include <glad/gl.h> // Include first to avoid errors
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

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

__global__ void updateParticles(float3 *d_instancePositions, float3 *d_instanceVelocities, int numParticles, float time)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numParticles)
    {
        d_instancePositions[i].x += cos(time + i) * 0.004f;
        d_instancePositions[i].y += sin(time + i) * 0.004f;
    }
}

int main()
{
    glfwSetErrorCallback(glfwErrorCallback);
    if (!glfwInit())
        return -1;

    GLFWwindow *window = glfwCreateWindow(WIDTH, HEIGHT, "CUDA Particles", NULL, NULL);
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

    // Singular particle vertex data
    float particleVertex[] = {
        1.0f, 1.0f, 1.0f, 1.0f // Color (r, g, b, a)
    };

    uint numParticles = 50000;
    std::cout << "Particle count: " << numParticles << std::endl;

    // Particle initial positions
    std::vector<float3> instancePositionData;
    for (int i = 0; i < numParticles; i++)
    {
        float3 position = make_float3(
            (float)rand() / RAND_MAX - 0.5f, // x
            (float)rand() / RAND_MAX - 0.5f, // y
            0.0f);                           // z
        instancePositionData.push_back(position);
    }

    // Particle position offsets
    std::vector<float3> instanceVelocityData;
    for (int i = 0; i < numParticles; i++)
    {
        float3 position = make_float3(
            (float)rand() / RAND_MAX - 0.5f, // x
            (float)rand() / RAND_MAX - 0.5f, // y
            0.0f);                           // z
        instanceVelocityData.push_back(position);
    }

    // Particle VAO (for instanced rendering)
    GLuint particleVao;
    glGenVertexArrays(1, &particleVao);
    glBindVertexArray(particleVao);

    // Singular particle vertex data
    GLuint particleVbo;
    glGenBuffers(1, &particleVbo);
    glBindBuffer(GL_ARRAY_BUFFER, particleVbo);
    glBufferData(
        GL_ARRAY_BUFFER,        // target
        sizeof(particleVertex), // size
        particleVertex,         // data
        GL_STATIC_DRAW);        // usage

    // Color attribute
    glVertexAttribPointer(
        0,          // index
        4,          // size
        GL_FLOAT,   // type
        GL_FALSE,   // normalized
        0,          // stride
        (void *)0); // pointer
    glEnableVertexAttribArray(0);

    // Instance positions
    GLuint instancePositionsVbo;
    glGenBuffers(1, &instancePositionsVbo);
    glBindBuffer(GL_ARRAY_BUFFER, instancePositionsVbo);
    glBufferData(
        GL_ARRAY_BUFFER,               // target
        numParticles * sizeof(float3), // size
        instancePositionData.data(),   // data
        GL_STATIC_DRAW);               // usage

    // Position attribute
    glVertexAttribPointer(
        1,                 // index
        3,                 // size
        GL_FLOAT,          // type
        GL_FALSE,          // normalized
        3 * sizeof(float), // stride
        (void *)0);        // pointer
    glEnableVertexAttribArray(1);
    glVertexAttribDivisor(1, 1); // Update attribute every 1 instance

    // Instance velocities
    GLuint instanceVelocitiesVbo;
    glGenBuffers(1, &instanceVelocitiesVbo);
    glBindBuffer(GL_ARRAY_BUFFER, instanceVelocitiesVbo);
    glBufferData(
        GL_ARRAY_BUFFER,               // target
        numParticles * sizeof(float3), // size
        instanceVelocityData.data(),   // data
        GL_STATIC_DRAW);               // usage

    // Velocity attribute
    glVertexAttribPointer(
        2,                 // index
        3,                 // size
        GL_FLOAT,          // type
        GL_FALSE,          // normalized
        3 * sizeof(float), // stride
        (void *)0);        // pointer
    glEnableVertexAttribArray(2);
    glVertexAttribDivisor(2, 1); // Update attribute every 1 instance

    glBindBuffer(GL_ARRAY_BUFFER, 0); // Unbind VBO
    glBindVertexArray(0);             // Unbind VAO

    // Register instance VBOs with CUDA
    cudaGraphicsResource *cuda_positions_vbo_resource;
    cudaGraphicsResource *cuda_velocities_vbo_resource;
    cudaGraphicsGLRegisterBuffer(&cuda_positions_vbo_resource, instancePositionsVbo, cudaGraphicsMapFlagsWriteDiscard);
    cudaGraphicsGLRegisterBuffer(&cuda_velocities_vbo_resource, instanceVelocitiesVbo, cudaGraphicsMapFlagsWriteDiscard);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    while (!glfwWindowShouldClose(window))
    {
        glClearColor(0.15f, 0.15f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Map instance VBOs to CUDA on device
        float3 *d_instancePositions;
        float3 *d_instanceVelocities;
        cudaGraphicsMapResources(1, &cuda_positions_vbo_resource, 0);
        cudaGraphicsMapResources(1, &cuda_velocities_vbo_resource, 0);
        cudaGraphicsResourceGetMappedPointer((void **)&d_instancePositions, nullptr, cuda_positions_vbo_resource);
        cudaGraphicsResourceGetMappedPointer((void **)&d_instanceVelocities, nullptr, cuda_velocities_vbo_resource);

        // Update particles
        int threadsPerBlock = 256;
        int blocksPerGrid = (numParticles + threadsPerBlock - 1) / threadsPerBlock;
        updateParticles<<<blocksPerGrid, threadsPerBlock>>>(d_instancePositions, d_instanceVelocities, numParticles, glfwGetTime());
        cudaError_t err = cudaDeviceSynchronize();
        err = cudaDeviceSynchronize(); // Blocks execution until kernel is finished
        if (err != cudaSuccess)
        {
            std::cerr << "Failed to synchronize on the CUDA device (error code " << cudaGetErrorString(err) << ")!" << std::endl;
            exit(EXIT_FAILURE);
        }

        // Unmap instance VBO data from CUDA (so OpenGL can use it)
        cudaGraphicsUnmapResources(1, &cuda_positions_vbo_resource, 0);
        cudaGraphicsUnmapResources(1, &cuda_velocities_vbo_resource, 0);

        // Draw particles
        shader.use();
        glBindVertexArray(particleVao);
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

    // Clean up
    cudaGraphicsUnregisterResource(cuda_positions_vbo_resource);
    cudaGraphicsUnregisterResource(cuda_velocities_vbo_resource);
    glDeleteVertexArrays(1, &particleVao);
    glDeleteBuffers(1, &particleVbo);
    glDeleteBuffers(1, &instancePositionsVbo);
    glDeleteBuffers(1, &instanceVelocitiesVbo);
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
