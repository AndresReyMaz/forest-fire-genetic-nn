#include "reproduction_util.h"

#include <algorithm>
#include <bitset>
#include <unordered_set>
#include <utility>
#include <vector>

#include "individual.h"
#include "random_util.h"

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
  std::vector<int> random_layers = get_randoms(number_of_layers, 1, total_layers);
  // Create copy of parent1.
  Individual child1 = Individual(parent1);
  // Get the number of neurons of each of the randomly chosen layers.
  std::vector<int> number_of_neurons_for_layer;
  for (auto layer : random_layers) {
    if (layer <= parent1.get_hidden_layers()) {
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
  case 5: return layer_crossover(parent1, parent2);
  case 6: return training_time_swap(parent2, parent1);
  case 7: return learning_rate_swap(parent2, parent1);
  case 8: return momentum_swap(parent2, parent1);
  case 9: return neurons_swap(parent2, parent1);
  case 10: return layer_crossover(parent2, parent1);
  default: return neurons_swap(parent1, parent2);
  }
}


void generate_children(std::vector<Individual>& population) {
  // Randomly generate parent indices.
  std::vector<int> parent_indices = get_randoms_unique(20, 0, population.size() - 1);

  // Fast hashset for individuals.
  std::unordered_set<std::bitset<57>> hash_set;
  for (auto individual : population) {
    hash_set.insert(individual.get_vector());
  }

  // Index(i) produces children with Index(i+1).
  for (unsigned i = 0; i < parent_indices.size(); i += 2) {
    const Individual& parent1 = population[parent_indices[i]];
    const Individual& parent2 = population[parent_indices[i+1]];
    for (int oper = 1; oper <= 10; oper++) {
      if ((oper == 5 or oper == 10) and (parent1.get_hidden_layers() == 0 or parent2.get_hidden_layers() == 0))
        continue;
      Individual child = crossover(oper, parent1, parent2);
      if (!child.is_valid_individual())
        continue;
      // Add to population if it is not a clone.
      if (hash_set.find(child.get_vector()) == hash_set.end()) {
        hash_set.insert(child.get_vector());
        population.push_back(child);
      }
    }


    // Crossover point operators.
    std::pair<int, int> range = population[parent_indices[i]].get_momentum_crossover_point();
    Individual child = Individual(parent1);
    child.copy_bits_from(parent2, range);
    if (child.is_valid_individual() and hash_set.find(child.get_vector()) == hash_set.end()) {
      hash_set.insert(child.get_vector());
      population.push_back(child);
    }

    range = population[parent_indices[i]].get_learning_rate_crossover_point();
    child = Individual(parent1);
    child.copy_bits_from(parent2, range);
    if (child.is_valid_individual() and hash_set.find(child.get_vector()) == hash_set.end()) {
      hash_set.insert(child.get_vector());
      population.push_back(child);
    }

    range = population[parent_indices[i]].get_training_time_crossover_point();
    child = Individual(parent1);
    child.copy_bits_from(parent2, range);
    if (child.is_valid_individual() and hash_set.find(child.get_vector()) == hash_set.end()) {
      hash_set.insert(child.get_vector());
      population.push_back(child);
    }

    // Random neuron selection.
    // If either of the parents has no hidden layers, then skip.
    if (parent1.get_hidden_layers() == 0 or parent2.get_hidden_layers() == 0)
      continue;
    // Larger of the two.
    std::pair<Individual, Individual> children = random_neuron_selection(parent1, parent2,
                                                                         std::max(parent1.get_hidden_layers(), parent2.get_hidden_layers()));
    if (children.first.is_valid_individual() and hash_set.find(children.first.get_vector()) == hash_set.end()) {
      hash_set.insert(children.first.get_vector());
      population.push_back(children.first);
    }
    if (children.second.is_valid_individual() and hash_set.find(children.second.get_vector()) == hash_set.end()) {
      hash_set.insert(children.second.get_vector());
      population.push_back(children.second);
    }
  }
}
