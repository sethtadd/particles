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

__global__ void updateParticles(float3 *d_initialPositionsInstanceData, float3 *d_positionOffsetsInstanceData, int numParticles, float time)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numParticles)
    {
        float rotationSpeed = (1 + (i % 10)) * 0.06f;
        d_positionOffsetsInstanceData[i].x = cos(rotationSpeed * time + i) * 0.45f;
        d_positionOffsetsInstanceData[i].y = sin(rotationSpeed * time + i) * 0.45f;
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
    float vertices[] = {
        1.0f, 1.0f, 1.0f, 1.0f // Color (r, g, b, a)
    };

    uint numParticles = 50000;
    std::cout << "Particle count: " << numParticles << std::endl;

    // Particle initial positions
    std::vector<float3> instanceInitialPositionsData;
    for (int i = 0; i < numParticles; i++)
    {
        float3 position = make_float3(
            (float)rand() / RAND_MAX - 0.5f, // x
            (float)rand() / RAND_MAX - 0.5f, // y
            0.0f);                           // z
        instanceInitialPositionsData.push_back(position);
    }

    // Particle position offsets
    std::vector<float3> instancePositionOffsetsData;
    for (int i = 0; i < numParticles; i++)
    {
        float3 position = make_float3(
            (float)rand() / RAND_MAX - 0.5f, // x
            (float)rand() / RAND_MAX - 0.5f, // y
            0.0f);                           // z
        instancePositionOffsetsData.push_back(position);
    }

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

    // Color attribute
    glVertexAttribPointer(
        0,          // index
        4,          // size
        GL_FLOAT,   // type
        GL_FALSE,   // normalized
        0,          // stride
        (void *)0); // pointer
    glEnableVertexAttribArray(0);

    // Instance initial position data
    GLuint instanceInitialPositionsVbo;
    glGenBuffers(1, &instanceInitialPositionsVbo);
    glBindBuffer(GL_ARRAY_BUFFER, instanceInitialPositionsVbo);
    glBufferData(
        GL_ARRAY_BUFFER,                     // target
        numParticles * sizeof(float3),       // size
        instanceInitialPositionsData.data(), // data
        GL_STATIC_DRAW);                     // usage

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

    // Instance position offset data
    GLuint instancePositionOffsetsVbo;
    glGenBuffers(1, &instancePositionOffsetsVbo);
    glBindBuffer(GL_ARRAY_BUFFER, instancePositionOffsetsVbo);
    glBufferData(
        GL_ARRAY_BUFFER,                    // target
        numParticles * sizeof(float3),      // size
        instancePositionOffsetsData.data(), // data
        GL_STATIC_DRAW);                    // usage

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

    // Register instanceInitialPositionsVbo with CUDA
    cudaGraphicsResource *cuda_initial_positions_vbo_resource;
    cudaGraphicsGLRegisterBuffer(&cuda_initial_positions_vbo_resource, instanceInitialPositionsVbo, cudaGraphicsMapFlagsWriteDiscard);

    // Register instancePositionOffsetsVbo with CUDA
    cudaGraphicsResource *cuda_position_offsets_vbo_resource;
    cudaGraphicsGLRegisterBuffer(&cuda_position_offsets_vbo_resource, instancePositionOffsetsVbo, cudaGraphicsMapFlagsWriteDiscard);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    while (!glfwWindowShouldClose(window))
    {
        glClearColor(0.15f, 0.15f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Map instanceInitialPositionsVbo to CUDA
        float3 *d_initialPositionsInstanceData;
        cudaGraphicsMapResources(1, &cuda_initial_positions_vbo_resource, 0);
        size_t num_bytes;
        cudaGraphicsResourceGetMappedPointer((void **)&d_initialPositionsInstanceData, &num_bytes, cuda_initial_positions_vbo_resource);

        // Map instanceInitialPositionsVbo to CUDA
        float3 *d_positionOffsetsInstanceData;
        cudaGraphicsMapResources(1, &cuda_position_offsets_vbo_resource, 0);
        // size_t num_bytes;
        cudaGraphicsResourceGetMappedPointer((void **)&d_positionOffsetsInstanceData, &num_bytes, cuda_position_offsets_vbo_resource);

        // Update particles
        int threadsPerBlock = 256;
        int blocksPerGrid = (numParticles + threadsPerBlock - 1) / threadsPerBlock;
        updateParticles<<<blocksPerGrid, threadsPerBlock>>>(d_initialPositionsInstanceData, d_positionOffsetsInstanceData, numParticles, glfwGetTime());
        cudaError_t err = cudaDeviceSynchronize();
        err = cudaDeviceSynchronize(); // Blocks execution until kernel is finished
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to synchronize on the CUDA device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Unmap instanceInitialPositionsVbo from CUDA (so OpenGL can use it)
        cudaGraphicsUnmapResources(1, &cuda_initial_positions_vbo_resource, 0);
        cudaGraphicsUnmapResources(1, &cuda_position_offsets_vbo_resource, 0);

        // Draw particles
        shader.use();
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

    // Clean up
    cudaGraphicsUnregisterResource(cuda_initial_positions_vbo_resource);
    cudaGraphicsUnregisterResource(cuda_position_offsets_vbo_resource);
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &particleVbo);
    glDeleteBuffers(1, &instanceInitialPositionsVbo);
    glDeleteBuffers(1, &instancePositionOffsetsVbo);
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
