#ifndef GENETIC_SYSTEM_UTIL_H
#define GENETIC_SYSTEM_UTIL_H

#include <cstdio>
#include <iostream>
#include <memory>
#include <regex>
#include <stdexcept>
#include <string>
#include <array>

int get_global_precision() {
  // Runs the neural network using Weka and parses the result.
  std::array<char, 128> buffer;
  // The output of the process.
  std::string result;
  std::shared_ptr<FILE> pipe(popen("java weka.classifiers.functions.MultilayerPerceptron -t $WEKAINSTALL/data/iris.arff", "r"), pclose);
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
    string s = "";
    for (unsigned i = 0; i < result.length(); ++i) {
      if (result[i] == '.') {
        ans += stoi(s) * 1000;
        s = "";
        continue;
      }
      s += result[i];
    }
    int tmp = stoi(s);
    while (tmp >= 999) {
      tmp /= 10;
    }
    return ans + tmp;
  } else {
    return stoi(result) * 1000;
  }
}

#endif
