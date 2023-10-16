#ifndef CAMERA_H
#define CAMERA_H

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class Camera
{
public:
    // Constructor parameters
    float aspectRatio;
    glm::vec3 position;
    glm::vec3 worldUp;
    float yaw;
    float pitch;

    // Other parameters
    glm::vec3 up;
    glm::vec3 front;
    glm::vec3 right;
    float fov = glm::radians(70.0f); // Field of View in radians
    float nearClip = 0.1f;           // Near clipping plane
    float farClip = 100.0f;          // Far clipping plane

    Camera(float aspectRatio, glm::vec3 position, glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f), float yaw = -90.0f, float pitch = 0.0f);
    glm::mat4 getViewMatrix();
    glm::mat4 getProjectionMatrix();
    void updateCameraVectors();
};

#endif // CAMERA_H
