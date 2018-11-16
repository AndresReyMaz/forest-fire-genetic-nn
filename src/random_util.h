#ifndef GENETIC_RANDOM_UTIL_H
#define GENETIC_RANDOM_UTIL_H

#include <vector>

// Generates a random number between lo and hi inclusive.
int get_random(int lo, int hi);

// Generates a list of random numbers between lo and hi, with possible repetitions.
std::vector<int> get_randoms(int n, int lo, int hi);

// Generates a list of random numbers between lo and hi inclusive, without repetitions.
std::vector<int> get_randoms_unique(int n, int lo, int hi);

#endif
