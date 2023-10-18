#include <iostream>

#include <glad/gl.h> // Include first to avoid errors
#include <GLFW/glfw3.h>

#include "ParticleSystem.cuh"
#include "Shader.hpp"
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

Camera camera((float)WIDTH / HEIGHT, glm::vec3(0.0f, 0.0f, 1.1f));
float lastMouseX = WIDTH / 2.0f;
float lastMouseY = HEIGHT / 2.0f;

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
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED); // Capture cursor

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
    std::cout << "Vendor: " << glGetString(GL_VENDOR) << std::endl;

    glViewport(0, 0, WIDTH, HEIGHT);

    Shader shader("shaders/particles.vertex.glsl", "shaders/particles.geometry.glsl", "shaders/particles.fragment.glsl");

    ParticleSystem particleSystem = ParticleSystem();
    particleSystem.init(10000);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    double lastTime = glfwGetTime();
    double currentTime;
    int lastFrame = 0;
    int currentFrame;

    for (int frameCount = 0; !glfwWindowShouldClose(window); ++frameCount)
    {
        // Calculate and print FPS
        currentTime = glfwGetTime();
        currentFrame = frameCount;
        if (currentTime - lastTime >= 1.0)
        {

            std::cout << "\rFPS: " << currentFrame - lastFrame << std::flush;
            lastTime = currentTime;
            lastFrame = currentFrame;
        }

        glClearColor(0.07f, 0.07f, 0.07f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        particleSystem.update();

        // Draw particles
        shader.use();
        shader.setMatrix4f("view", camera.getViewMatrix());
        shader.setMatrix4f("projection", camera.getProjectionMatrix());

        particleSystem.render();

        glfwPollEvents();
        handleInput(window);
        glfwSwapBuffers(window);
    }

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
    float movementSpeed = 0.05f;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.position += movementSpeed * camera.up;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.position -= movementSpeed * camera.up;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.position -= camera.right * movementSpeed;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.position += camera.right * movementSpeed;
}

void mouse_callback(GLFWwindow *window, double xPos, double yPos)
{
    float xoffset = xPos - lastMouseX;
    float yoffset = lastMouseY - yPos; // Reversed since y-coordinates go from bottom to top

    lastMouseX = xPos;
    lastMouseY = yPos;

    float sensitivity = 0.05f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    camera.yaw += xoffset;
    camera.pitch += yoffset;

    if (camera.pitch > 89.0f)
        camera.pitch = 89.0f;
    if (camera.pitch < -89.0f)
        camera.pitch = -89.0f;

    camera.updateCameraVectors();
}

void scroll_callback(GLFWwindow *window, double xOffset, double yOffset)
{
    camera.position += (float)yOffset * camera.front;
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
}