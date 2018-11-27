#include "system_util.h"

#include <array>
#include <cstdio>
#include <iostream>
#include <memory>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

int get_global_precision(const std::vector<int>& layers, int training_time, int learning_rate, int momentum) {
 // Convert parameters
  int N = training_time * 20 + 100;
  double L = ((double) learning_rate) / 10000.0;
  double M = ((double) momentum) / 10000.0;
  std::string H = "\"";
  if (layers.size() == 0) {
    H += "\"";
  } else {
    for (unsigned i = 0; i < layers.size(); ++i) {
      H += std::to_string(layers[i]);
      if (i != layers.size() - 1)
        H += ", ";
      else H += "\"";
    }
  }
  // Runs the neural network using Weka and parses the result.
  std::array<char, 128> buffer;
  // The output of the process.
  std::string result;
  std::stringstream ss;
  ss << "java weka.classifiers.functions.MultilayerPerceptron -t ~/code/ForestFireDetector/forestfiresfinal1000.arff";
  ss << " -L " << L;
  ss << " -N " << N;
  ss << " -M " << M;
  ss << " -H " << H;
  std::string command = ss.str();
  std::cout << command << std::endl;
  std::shared_ptr<FILE> pipe(popen(command.c_str(), "r"), pclose);
  if (!pipe) throw std::runtime_error("popen() failed!");
  while (!feof(pipe.get())) {
    if (fgets(buffer.data(), 128, pipe.get()) != nullptr)
      result += buffer.data();
  }

  // Parse using regex to extract the global precision.
  std::smatch sm;
  std::regex reg("validation ===\n\nCorrectly Classified Instances[ ]*[0-9]*[ ]*[0-9]*\\.?[0-9]*");
  bool r = std::regex_search(result, sm, reg);
  if (!r) throw std::runtime_error("Global precision not found");
  result = sm.str();
  // The last value of result is now the global precision.
  // Parse float to integer.
  bool period_found = false;
  int number_of_characters = 0;
  for (auto it = result.rbegin(); it != result.rend(); ++it) {
    if (*it == ' ') {
      break;
    } else if (*it == '.') {
      period_found = true;
    }
    number_of_characters++;
  }
  // Now get the substring.
  result = result.substr(result.length() - number_of_characters, 10000);
  if (period_found) {
    // Get the value before the string.
    int ans = 0;
    std::string s = "";
    for (unsigned i = 0; i < result.length(); ++i) {
      if (result[i] == '.') {
        ans += std::stoi(s) * 1000;
        s = "";
        continue;
      }
      s += result[i];
    }
    int tmp = std::stoi(s);
    while (tmp <= 99) {
      tmp *= 10;
    }
    while (tmp > 999) {
      tmp /= 10;
    }
    return ans + tmp;
  } else {
    return std::stoi(result) * 1000;
  }
}
