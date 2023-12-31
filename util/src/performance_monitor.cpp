#include "performance_monitor.h"

PerformanceMonitor::PerformanceMonitor(float currentTime, const std::string &exportFolderName):
  totalIterations(0),
  lastIterationTime(currentTime),
  fpsResetIterations(10),
  fpsPeriodIterations(0),
  fpsTimer(0.0f),
  framesPerSecond(0.0f),
  exportFolderName(exportFolderName),
  exportDataIterations(1000)
{}

void PerformanceMonitor::update(float currentTime)
{
  fpsPeriodIterations += 1;
  totalIterations += 1;

  dt = currentTime - lastIterationTime;

  fpsTimer += dt;
  lastIterationTime = currentTime;

  if (fpsPeriodIterations >= fpsResetIterations)
  {
    framesPerSecond = fpsPeriodIterations / fpsTimer;
    fpsPeriodIterations = 0;
    fpsTimer = 0.0;

    if (!dataExported) {
      store["FPS"].push_back(std::make_pair(framesPerSecond, totalIterations));
    }
  }

  if (totalIterations >= exportDataIterations && !dataExported) {
    exportData("data-" + exportFolderName + "/");
    dataExported = true;
  } 
}

void PerformanceMonitor::startProcessTimer(int process)
{
  if (process >= 4 || dataExported) return;
  processStartTimes[process] = std::chrono::high_resolution_clock::now();
}

void PerformanceMonitor::endProcessTimer(int process)
{
  if (process >= 4 || dataExported) return;
      
  processEndTimes[process] = std::chrono::high_resolution_clock::now();
  double elapsedTime = std::chrono::duration<double>(processEndTimes[process] - processStartTimes[process]).count();
  processElapsedTimes[process] += elapsedTime;
  processIterations[process]++;

  store[processKeys[process]].push_back(std::make_pair(processElapsedTimes[process], processIterations[process]));
}

void PerformanceMonitor::exportData(const std::string &folderPath)
{
  std::cout << "Exporting data..." << std::endl;

  std::filesystem::create_directories(folderPath);

  // For each vector of the store
  for (auto it = store.begin(); it != store.end(); ++it)
  {
    std::string key = it->first;
    std::vector<std::pair<double, double>> values = it->second;

    // Write into a representative csv file all the pairs from the vector
    std::string filePath = folderPath + key + ".csv";
    std::ofstream file(filePath);
    for (auto vIt = values.begin(); vIt != values.end(); ++vIt)
    {
      double p1 = vIt->first;
      double p2 = vIt->second;
      if (file.is_open()) {
        file << p1 << "," << p2 << std::endl;
      } else {
        std::cout << "Failed to open the file: " << filePath << std::endl;
      }
    }
    file.close();
  }
}