#include <random>
#include <utility>
#include <vector>

#include "individual.h"
#include "reproduction_util.h"

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

int get_random(int lo, int hi) {
  // Get random value between lo and hi, inclusive.
  std::default_random_engine eng {std::random_device{}()};
  std::uniform_int_distribution<> dist {lo, hi};
  return dist(eng);
}

Individual training_time_swap(const Individual& parent1, const Individual& parent2) {
  Individual child = Individual(parent1);
  child.set_training_time(parent2.get_training_time());
  return child;
}

Individual learning_rate_swap(const Individual& parent1, const Individual& parent2) {
  Individual child = Individual(parent1);
  child.set_learning_rate(parent2.get_learning_rate());
  return child;
}

Individual momentum_swap(const Individual& parent1, const Individual& parent2) {
  Individual child = Individual(parent1);
  child.set_momentum(parent2.get_momentum());
  return child;
}

Individual neurons_swap(const Individual& parent1, const Individual& parent2) {
  Individual child = Individual(parent1);
  child.set_neurons(parent2.get_neurons());
  return child;
}

std::pair<Individual, Individual> random_neuron_selection(const Individual& parent1, const Individual& parent2, int number_of_layers) {
  int total_layers = parent1.get_hidden_layers() + parent2.get_hidden_layers();
  // Select |number_of_layers| random layers.
  std::vector<int> random_layers = get_randoms(number_of_layers, 0, total_layers - 1);
  // Create copy of parent1.
  Individual child1 = Individual(parent1);
  // Get the number of neurons of each of the randomly chosen layers.
  std::vector<int> number_of_neurons_for_layer;
  for (auto layer : random_layers) {
    if (layer < parent1.get_hidden_layers()) {
      // Use parent 1
      number_of_neurons_for_layer.push_back(parent1.get_neurons_for_layer(layer));
    } else {
      // Use parent 2
      number_of_neurons_for_layer.push_back(parent2.get_neurons_for_layer(layer - parent1.get_hidden_layers()));
    }
  }
  // Assign the neurons to the first child.
  child1.set_neurons(number_of_neurons_for_layer);
  // Create copy of parent2.
  Individual child2 = Individual(parent2);
  // Assign the neurons to the second child.
  child2.set_neurons(number_of_neurons_for_layer);
  return {child1, child2};
}


Individual layer_crossover(const Individual& parent1, const Individual& parent2) {
  assert(parent1.get_hidden_layers() > 0 and parent2.get_hidden_layers() > 0);
  // Select a random layer from each parent.
  int layer_index_parent1 = get_random(1, parent1.get_hidden_layers());
  int layer_for_parent1 = parent1.get_neurons_for_layer(layer_index_parent1);
  int layer_index_parent2 = get_random(1, parent2.get_hidden_layers());
  int layer_for_parent2 = parent2.get_neurons_for_layer(layer_index_parent2);
  std::bitset<Individual::NUMBER_NEURONS_LENGTH> tmp_vector1(layer_for_parent1);
  std::bitset<Individual::NUMBER_NEURONS_LENGTH> tmp_vector2(layer_for_parent2);

  // Make copy of parent1.
  Individual child = Individual(parent1);
  
  // Select a random crossover point.
  int point = get_random(1, Individual::NUMBER_NEURONS_LENGTH);
  // Assign values [point, NUMBER_NEURONS_LENGTH] of second parent to first parent.
  for (int i = point; i <= Individual::NUMBER_NEURONS_LENGTH; ++i) {
    tmp_vector1[i] = tmp_vector2[i];
  }
  // Assign new hidden layer to child.
  child.set_neurons_for_layer(layer_index_parent1, tmp_vector1.to_ulong());
  return child;
}

Individual crossover(int crossover_operator, const Individual& parent1, const Individual& parent2) {
  // Does the crossover for the chosen individual.
  switch(crossover_operator) {
  case 1: return training_time_swap(parent1, parent2);
  case 2: return learning_rate_swap(parent1, parent2);
  case 3: return momentum_swap(parent1, parent2);
  case 4: return neurons_swap(parent1, parent2);
  default: return neurons_swap(parent1, parent2);
  }
}


void generate_children(std::vector<Individual>& population, int desired_population_size) {
  desired_population_size++;
  std::pair<int, int> range = population[0].get_momentum_crossover_point();
  Individual a = population[0];
  Individual b = Individual(a);
  b.copy_bits_from(population[1], range);
}
