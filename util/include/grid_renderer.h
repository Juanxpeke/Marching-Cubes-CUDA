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

  GridRenderer(int cells, float gridSize);
  void render();
  ~GridRenderer();

private:
  int cells;
  float gridSize;

  GLuint VAO;
  GLuint VBO;
  int numVertices;

  void compileShaders();
  void initBuffers();
};