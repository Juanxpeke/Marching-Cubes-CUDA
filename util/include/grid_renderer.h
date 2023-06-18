#pragma once

#include <iostream>
#include <string>
#include <fstream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "file_manager.h"

class GridRenderer
{
public:
  GLuint shaderProgram;

  float renderDistance;

  GridRenderer(int cells, float size, float renderDistance);
  void render();
  ~GridRenderer();

private:
  int cells;
  float size;

  GLuint VAO;
  GLuint VBO;
  int numVertices;

  void compileShaders();
  void initBuffers();
};