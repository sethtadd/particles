#include "glad/gl.h" // Include first to avoid errors
#include <GLFW/glfw3.h>
#include <stdio.h>
#include "Shader.hpp"

int main()
{
    if (!glfwInit())
        return -1;

    GLFWwindow *window = glfwCreateWindow(768, 512, "Hello World", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    int version = gladLoadGL(glfwGetProcAddress);
    printf("GL %d.%d\n", GLAD_VERSION_MAJOR(version), GLAD_VERSION_MINOR(version));

    Shader shader("shaders/triangle.vertex.glsl", "shaders/triangle.fragment.glsl");

    GLfloat vertices[] = {
        // positions       // colors
        0.5f, -0.5f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f,  // bottom right
        -0.5f, -0.5f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, // bottom left
        0.0f, 0.5f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f    // top center
    };

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(
        GL_ARRAY_BUFFER,  // target
        sizeof(vertices), // size
        vertices,         // data
        GL_STATIC_DRAW);  // usage

    // Position attribute
    glVertexAttribPointer(
        0,                 // index
        3,                 // size
        GL_FLOAT,          // type
        GL_FALSE,          // normalized
        7 * sizeof(float), // stride
        (void *)0);        // pointer
    glEnableVertexAttribArray(0);

    // Color attribute
    glVertexAttribPointer(
        1,                            // index
        4,                            // size
        GL_FLOAT,                     // type
        GL_FALSE,                     // normalized
        7 * sizeof(float),            // stride
        (void *)(3 * sizeof(float))); // pointer
    glEnableVertexAttribArray(1);

    // Unbind vbo and vao
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    while (!glfwWindowShouldClose(window))
    {
        glClearColor(0.53332607f, 0.4382106f, 0.72355703f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        shader.use();

        // Draw triangle
        glBindVertexArray(vao);
        glDrawArrays(
            GL_TRIANGLES,     // mode: type of primitives to render
            0,                // first: starting index
            3);               // count: number of indicies to render
        glBindVertexArray(0); // Unbind vao

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Clean up
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glfwTerminate();
    return 0;
}
