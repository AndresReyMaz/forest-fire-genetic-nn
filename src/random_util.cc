#include "random_util.h"

#include "assert.h"
#include <random>
#include <unordered_set>
#include <vector>

int get_random(int lo, int hi) {
  std::default_random_engine eng {std::random_device{}()};
  std::uniform_int_distribution<> dist {lo, hi};
  return dist(eng);
}

std::vector<int> get_randoms(int n, int lo, int hi) {
  // Gets n random values between lo and hi, inclusive.
  std::default_random_engine eng {std::random_device{}()};
  std::uniform_int_distribution<> dist {lo, hi};
  std::vector<int> v;
  while (n--) {
    v.push_back(dist(eng));
  }
  return v;
}

std::vector<int> get_randoms_unique(int n, int lo, int hi) {
  assert(n <= hi - lo + 1);
  std::default_random_engine eng {std::random_device{}()};
  std::uniform_int_distribution<> dist {lo, hi};
  std::vector<int> v;
  std::unordered_set<int> s;
  while (s.size() != unsigned(n)) {
    int x = dist(eng);
    if (s.find(x) == s.end()) {
      s.insert(x);
      v.push_back(x);
    }
  }
  return v;
}
