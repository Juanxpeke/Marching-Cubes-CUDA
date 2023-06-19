#pragma once

#include <iostream>
#include <fstream>

inline std::string getFileContent(const char* filename)
{
  std::ifstream file(filename);

  if (file && file.is_open())
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