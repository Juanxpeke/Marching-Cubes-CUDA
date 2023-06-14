#include "renderer.h"

std::string getFileContent(const char* filename)
{
  std::ifstream file(filename);

  if (!file) {
    std::cout << "Error reading the file " << filename;
    exit(0);
  }

  if (file.is_open())
  {
    std::string content((std::istreambuf_iterator<char>(file)),
                        (std::istreambuf_iterator<char>()));

    file.close();

    return content;
  } else {
    std::cout << "Failed to open the file " << filename << std::endl;
    exit(1);
  }
}

Renderer::Renderer()
{
}

void Renderer::init()
{
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

  // Creating a glfw window    
  window = glfwCreateWindow(1280, 720, "Test OpenGL", NULL, NULL);

  if (window == NULL)
  {
    std::cout << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    exit(1);
  } else {
    std::cout << "GLFW window created" << std::endl;
  }

  glfwMakeContextCurrent(window);

  // Loading all OpenGL function pointers with glad
  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
  {
    std::cout << "Failed to initialize GLAD" << std::endl;
    exit(1);
  } else {
    std::cout << "GLAD initialized successfully" << std::endl;
  }
}

void Renderer::compileShaders(const char* vsFile, const char* fsFile)
{
  // Shaders
  std::string vertexShaderCode = getFileContent(vsFile);
	std::string fragmentShaderCode = getFileContent(fsFile);

  const char* vertexShaderSource = vertexShaderCode.c_str();
	const char* fragmentShaderSource = fragmentShaderCode.c_str();

	// Create Vertex Shader Object and get its reference
	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
	// Attach Vertex Shader source to the Vertex Shader Object
	glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
	// Compile the Vertex Shader into machine code
	glCompileShader(vertexShader);

	// Create Fragment Shader Object and get its reference
	GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	// Attach Fragment Shader source to the Vertex Shader Object
	glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
	// Compile the Fragment Shader into machine code
	glCompileShader(fragmentShader);

	// Create Shader Program Object and get its reference
	mShaderProgram = glCreateProgram();
	// Attach the Vertex and Fragment Shaders to the Shader Program
	glAttachShader(mShaderProgram, vertexShader);
	glAttachShader(mShaderProgram, fragmentShader);
	// Wrap-up / link all the shaders together into the Shader Program
	glLinkProgram(mShaderProgram);

	// Delete the now useless Vertex and Fragment Shader Objects
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);
}

void Renderer::useShaders()
{
  // Use the shader program
  glUseProgram(mShaderProgram);
}

void Renderer::initBuffers()
{
	// Generate the VAO and VBO with only 1 object each
	glGenVertexArrays(1, &mVAO);
	glGenBuffers(1, &mVBO);

	// Make the VAO the current Vertex Array Object by binding it
	glBindVertexArray(mVAO);

	// Bind the VBO specifying it's a GL_ARRAY_BUFFER
	glBindBuffer(GL_ARRAY_BUFFER, mVBO);
	// Introduce the vertices into the VBO
	glBufferData(GL_ARRAY_BUFFER, mNumberOfPrimitives * 3 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

	// Configure the Vertex Attribute so that OpenGL knows how to read the VBO
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	// Enable the Vertex Attribute so that OpenGL knows to use it
	glEnableVertexAttribArray(0);

	// Bind both the VBO and VAO to 0 so that we don't accidentally modify the VAO and VBO
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Renderer::render()
{
  glClear(GL_COLOR_BUFFER_BIT);
  glDrawArrays(GL_POINTS, 0, mNumberOfPrimitives);
}



