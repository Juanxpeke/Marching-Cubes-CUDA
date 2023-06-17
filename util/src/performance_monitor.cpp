/**
 * @file performance_monitor.cpp
 * @brief Simple class to monitor the frames per second of an application
 * 
 * @author Daniel Calderón
 * @license MIT
*/

#include "performance_monitor.h"

void PerformanceMonitor::update(float currentTime)
{
  _framesCounter += 1;

  dt = currentTime - _currentTime;

  _timer += dt;
  _currentTime = currentTime;

  if (_timer > _period)
  {
    _framesPerSecond = _framesCounter / _timer;
    _milisecondsPerFrame = 1000.0 * _timer / _framesCounter;
    _framesCounter = 0;
    _timer = 0.0;
  }
}

std::ostream& operator<<(std::ostream& os, const PerformanceMonitor& perfMonitor)
{
  os << std::fixed << std::setprecision(2)
    << "[" << perfMonitor.getFPS() << " fps - "
    << perfMonitor.getMS() << " ms]";
  return os;
}