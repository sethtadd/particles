#include <iostream>

#include <glad/gl.h> // Include first to avoid errors
#include <GLFW/glfw3.h>

#include "ParticleSystem.hpp"
#include "Shader.hpp"
#include "Framebuffer.hpp"
#include "Camera.hpp"

const int WIDTH = 1024;
const int HEIGHT = 1024;

void glfwErrorCallback(int error, const char *description);

void handleInput(GLFWwindow *window);

// Callback functions
void mouse_callback(GLFWwindow *window, double xPos, double yPos);               // Mouse movement
void scroll_callback(GLFWwindow *window, double xOffset, double yOffset);        // Zooming in/out
void framebuffer_size_callback(GLFWwindow *window, int newWidth, int newHeight); // Handle window resizing
void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods);

void renderQuad();
float quadVertices[] = {
    // Positions        // Texture Coords
    -1.0f, 1.0f, 0.0f, 0.0f, 1.0f,  // Top left
    -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, // Bottom left
    1.0f, -1.0f, 0.0f, 1.0f, 0.0f,  // Bottom right

    -1.0f, 1.0f, 0.0f, 0.0f, 1.0f, // Top left
    1.0f, -1.0f, 0.0f, 1.0f, 0.0f, // Bottom right
    1.0f, 1.0f, 0.0f, 1.0f, 1.0f,  // Top right
};
GLuint quadVAO, quadVBO;

Camera camera((float)WIDTH / HEIGHT, glm::vec3(0.0f, 0.0f, 1.1f));

// Controls
float lastMouseX = WIDTH / 2.0f;
float lastMouseY = HEIGHT / 2.0f;

// Timing
double lastTime = glfwGetTime();
double currentTime;
float deltaTime;
float timeScale = 1.0f;

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
    // Capture cursor
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    // Disable vsync
    glfwSwapInterval(0);

    // set callback functions
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetKeyCallback(window, key_callback);

    // Initialize GLAD2
    int version = gladLoadGL(glfwGetProcAddress);

    // Print version info
    std::cout << "GLAD2 GL version: " << GLAD_VERSION_MAJOR(version) << "." << GLAD_VERSION_MINOR(version) << std::endl;
    std::cout << "OpenGL version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "Renderer: " << glGetString(GL_RENDERER) << std::endl;

    glViewport(0, 0, WIDTH, HEIGHT);

    ParticleSystem particleSystem = ParticleSystem();
    particleSystem.init(1000000, 0.003f);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    Shader hdrShader;
    hdrShader.init("shaders/hdr.vertex.glsl", "shaders/hdr.fragment.glsl");
    Framebuffer hdrFramebuffer(WIDTH, HEIGHT);

    for (int frameCount = 0; !glfwWindowShouldClose(window); ++frameCount)
    {
        // Calculate and print FPS
        currentTime = glfwGetTime();
        deltaTime = (float)(currentTime - lastTime);
        lastTime = currentTime;
        if (frameCount % 100 == 0)
        {
            const char *carriageReturn = "\r";
            const char *clearLine = "\033[K";
            std::cout << carriageReturn << "FPS: " << 1.0f / deltaTime << clearLine << std::flush;
        }

        // Render particles to hdrFramebuffer
        hdrFramebuffer.bind();
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        // glClearColor(0.07f, 0.07f, 0.07f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        particleSystem.update(deltaTime * timeScale);
        particleSystem.render(camera);
        hdrFramebuffer.unbind();

        // Render hdrFramebuffer to screen
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        hdrShader.use();
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, hdrFramebuffer.texture_);
        renderQuad();

        glfwPollEvents();
        handleInput(window);
        glfwSwapBuffers(window);
    }

    glDeleteVertexArrays(1, &quadVAO);
    glDeleteBuffers(1, &quadVBO);

    glfwTerminate();
    return 0;
}

void glfwErrorCallback(int error, const char *description)
{
    std::cerr << "GLFW Error: " << error << " - " << description << std::endl;
}

void handleInput(GLFWwindow *window)
{
    // camera movement
    float movementSpeed = 3.0f;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.position += deltaTime * movementSpeed * camera.up;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.position -= deltaTime * movementSpeed * camera.up;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.position -= deltaTime * movementSpeed * camera.right;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.position += deltaTime * movementSpeed * camera.right;
}

void mouse_callback(GLFWwindow *window, double xPos, double yPos)
{
    float xOffset = xPos - lastMouseX;
    float yOffset = lastMouseY - yPos; // Reversed since y-coordinates go from bottom to top

    lastMouseX = xPos;
    lastMouseY = yPos;

    float sensitivity = 0.03f;
    xOffset *= sensitivity;
    yOffset *= sensitivity;

    camera.yaw += xOffset;
    camera.pitch += yOffset;

    if (camera.pitch > 89.0f)
        camera.pitch = 89.0f;
    if (camera.pitch < -89.0f)
        camera.pitch = -89.0f;

    camera.updateCameraVectors();
}

void scroll_callback(GLFWwindow *window, double xOffset, double yOffset)
{
    float sensitivity = 0.4f;
    camera.position += sensitivity * (float)yOffset * camera.front;
}

void framebuffer_size_callback(GLFWwindow *window, int newWidth, int newHeight)
{
    glViewport(0, 0, newWidth, newHeight);
    camera.aspectRatio = (float)newWidth / newHeight;
}

void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    // Close window on escape key press
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
        timeScale /= 1.1f;
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
        timeScale *= 1.1f;
}

void renderQuad()
{
    if (quadVAO == 0)
    {
        glGenVertexArrays(1, &quadVAO);
        glGenBuffers(1, &quadVBO);
        glBindVertexArray(quadVAO);

        glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
    }
    glBindVertexArray(quadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
}
