#ifndef GENETIC_REPRODUCTION_UTIL_H
#define GENETIC_REPRODUCTION_UTIL_H

#include <vector>

#include "individual.h"

/* Generates the necessary number of children to reach the desired population size. */
void generate_children(std::vector<Individual>& population, int desired_population_size);

#endif
