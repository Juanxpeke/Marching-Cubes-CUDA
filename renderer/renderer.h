#pragma once

#include <iostream>
#include <string>
#include <fstream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

class Renderer
{
public:
  GLFWwindow* window;

  Renderer();
  void init();
  void compileShaders(const char* vsFile, const char* fsFile);
  void useShaders();
  void initBuffers();
  void render();
private:
  GLuint mVAO;
  GLuint mVBO;
  GLuint mShaderProgram;
  GLuint mNumberOfPrimitives;
};