#include "Camera.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

Camera::Camera(float aspectRatio, glm::vec3 position, glm::vec3 up, float yaw, float pitch)
    : aspectRatio(aspectRatio), position(position), worldUp(up), yaw(yaw), pitch(pitch), front(glm::vec3(0.0f, 0.0f, -1.0f))
{
    updateCameraVectors();
}

glm::mat4 Camera::getViewMatrix()
{
    return glm::lookAt(position, position + front, up);
}

glm::mat4 Camera::getProjectionMatrix()
{
    return glm::perspective(fov, aspectRatio, nearClip, farClip);
}

void Camera::updateCameraVectors()
{
    front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    front.y = sin(glm::radians(pitch));
    front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    front = glm::normalize(front);
    right = glm::normalize(glm::cross(front, worldUp));
    up = glm::normalize(glm::cross(right, front));
}

void Camera::orbit(float radius, float speed, float time)
{
    // Orbit around the origin
    float amount = speed * time;
    position.z = cosf(amount);
    position.x = sinf(amount);
    position.y = sinf(amount / 2.0f) / 2.0f;

    float scaledRadius = radius * (1.0f + sinf(amount / 2.0f) / 2.0f);
    position = glm::normalize(position) * scaledRadius;

    // Look at the origin
    front = glm::normalize(-position);
    yaw = glm::degrees(atan2(front.z, front.x));
    pitch = glm::degrees(asin(front.y));
    right = glm::normalize(glm::cross(front, worldUp));
    up = glm::normalize(glm::cross(right, front));
}