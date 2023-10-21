#ifndef FRAMEBUFFER_HPP
#define FRAMEBUFFER_HPP

#include <glad/gl.h>

class Framebuffer
{
public:
    GLuint fbo_;
    GLuint texture_;

    Framebuffer(int width, int height);
    ~Framebuffer();

    void bind();

    static void unbind();
};

#endif // FRAMEBUFFER_HPP
