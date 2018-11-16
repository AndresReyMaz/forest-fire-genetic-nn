#include "mutation_util.h"

#include <vector>

#include "individual.h"
#include "random_util.h"

void mutate(std::vector<Individual>& population) {
  int number_to_mutate = population.size() * 0.10;
  std::vector<int> indices = get_randoms_unique(number_to_mutate, 0, population.size() - 1);
  for (auto index : indices) {
    population[index].mutate();
  }
}

