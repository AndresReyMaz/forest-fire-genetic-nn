#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <vector>
using namespace std;

#include "individual.h"
#include "mutation_util.h"
#include "reproduction_util.h"
#include "system_util.h"
#include "anyoption.h"

void cull(std::vector<Individual>& population) {
  // Kills all but the top individuals of the population.
  std::sort(population.begin(), population.end());
  population.erase(population.begin() + 20, population.end());
}

void genetic_algorithm(int population_size) {
  // Randomly generate a population.
  std::vector<Individual> population = Individual::generate_random_population(population_size);
  for (unsigned i = 1; ; ++i) {
    std::cout << "Generation #" << i << ":\n";

    // Generate the children for the population.
    generate_children(population);

    // Generate random mutations.
    mutate(population);

    // Evaluate population.
    for (Individual individual : population)
      individual.evaluate();

    // Cull population.
    cull(population);

    std::cout << "Top individuals from this generation:\n";
    // Print out values.
    for (Individual individual : population) {
      std::cout << "  " << individual.get_precision() << std::endl;
      std::cout << individual.get_hidden_layers << " ";
      std::vector<int> layers = individual.get_hidden_layer_values();
      for (int layer : layers)
        std::cout << layer << " ";
      std::cout << individual.get_training_time() << " "
                << individual.get_learning_rate() << " "
                << individual.get_momentum() << std::endl;
    }
    std::cout << std::endl;
  }
}

int main() {
  AnyOption *opt = new AnyOption();
  delete opt;
  genetic_algorithm(20);
}
