#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <vector>

#include "individual.h"
#include "mutation_util.h"
#include "reproduction_util.h"
#include "system_util.h"

const unsigned POPULATION_SIZE_WITH_CHILDREN = 27;
const unsigned NUMBER_OF_GENERATIONS = 15;

void bubble_sort(std::vector<Individual>& population) {
  unsigned i, j;
  for (i = 0; i < population.size() - 1; i++) {
    for (j = 0; j < population.size() - i - 1; j++) {
      if (population[j + 1].get_precision() > population[j].get_precision()) {
        Individual tmp = population[j];
        tmp.set_precision(population[j].get_precision());
        population[j] = population[j+1];
        population[j+1] = tmp;
      }
    }
  }
}

void cull(std::vector<Individual>& population) {
  // Kills all but the top individuals of the population.
  bubble_sort(population);
  population.erase(population.begin() + POPULATION_SIZE, population.end());
}

void genetic_algorithm(int population_size) {
  // Randomly generate a population.
  std::vector<Individual> population = Individual::generate_random_population(population_size);
  for (unsigned i = 1; i <= NUMBER_OF_GENERATIONS; ++i) {
    std::cout << "Generation #" << i << ":\n";

    // Generate the children for the population.
    while (population.size() < POPULATION_SIZE_WITH_CHILDREN)
      generate_children(population, POPULATION_SIZE_WITH_CHILDREN);

    // Generate random mutations.
    mutate(population);
    // Evaluate population.
    for (unsigned j = 0; j < population.size(); ++j) {
      population[j].evaluate();
    }

    // Cull population.
    cull(population);

    std::cout << "Top individuals from this generation:\n";
    // Print out values.
    for (unsigned j = 0; j < population.size(); ++j) {
      std::cout << "  " << population[j].get_precision() << std::endl;
      std::cout << population[j].get_hidden_layers() << " ";
      std::vector<int> layers = population[j].get_hidden_layer_values();
      for (int layer : layers)
        std::cout << layer << " ";
      std::cout << population[j].get_training_time() << " "
                << population[j].get_learning_rate() << " "
                << population[j].get_momentum() << std::endl;
    }
    std::cout << std::endl;
  }
}

int main() {
  assert(Individual::MOMENTUM_END == N_BITS - 1);
  genetic_algorithm(POPULATION_SIZE);
  return 0;
}
