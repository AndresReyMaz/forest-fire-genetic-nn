#ifndef GENETIC_REPRODUCTION_UTIL_H
#define GENETIC_REPRODUCTION_UTIL_H

#include <vector>

#include "individual.h"

// Generate children for the population by randomly selecting parents.
void generate_children(std::vector<Individual>& population, unsigned desired_population_size);

#endif
