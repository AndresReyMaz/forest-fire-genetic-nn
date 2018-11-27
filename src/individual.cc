#include "individual.h"

#include <assert.h>
#include <bitset>
#include <random>
#include <unordered_set>
#include <vector>

#include "random_util.h"
#include "system_util.h"

const int Individual::LEARNING_RATE_END;
const int Individual::TRAINING_TIME_END;
const int Individual::MOMENTUM_END; // circumvents strange linker error

void Individual::evaluate() {
  if (precision == 0)
    precision = get_global_precision(get_hidden_layer_values(),
                                     get_training_time(),
                                     get_learning_rate(),
                                     get_momentum());
}

int Individual::to_value(int start, int end) const {
  // Returns the numeric value represented by a subsequence of bits.
  assert(start >= 0 and start <= end and end < N_BITS);
  int ret = 0;
  for (int i = start; i <= end; ++i) {
    ret *= 2;
    ret += bit_vector[i];
  }
  return ret;
}

void Individual::set_value(int start, int end, int value) {
  for (int i = end; i >= start; --i) {
    bit_vector[i] = value % 2;
    value /= 2;
  }
}

void Individual::set_training_time(int training_time) {
  Individual::set_value(TRAINING_TIME_START, TRAINING_TIME_END, training_time - MIN_TRAINING_TIME);
}

void Individual::set_learning_rate(int learning_rate) {
  Individual::set_value(LEARNING_RATE_START, LEARNING_RATE_END, learning_rate - MIN_LEARNING_RATE);
}

void Individual::set_momentum(int momentum) {
  Individual::set_value(MOMENTUM_START, MOMENTUM_END, momentum - MIN_MOMENTUM);
}

void Individual::set_neurons_for_layer(int layer, int value) {
  Individual::set_value(NUMBER_NEURONS_START + NUMBER_NEURONS_LENGTH * layer, NUMBER_NEURONS_START + NUMBER_NEURONS_LENGTH * (layer + 1), value - MIN_NUMBER_NEURONS);
}

void Individual::set_number_of_layers(int number_of_layers) {
  assert(number_of_layers >= MIN_HIDDEN_LAYERS and number_of_layers <= MAX_HIDDEN_LAYERS);
  Individual::set_value(HIDDEN_LAYERS_START, HIDDEN_LAYERS_END, number_of_layers - MIN_HIDDEN_LAYERS);
}

void Individual::set_neurons(const std::vector<int>& layers) {
  set_number_of_layers(layers.size());
  for (unsigned i = 0; i < layers.size(); ++i) {
    set_neurons_for_layer(i, layers[i]);
  }
}

void Individual::mutate() {
  assert(is_valid_individual());
  while (true) {
    int random_bit = get_random(0, MOMENTUM_END);
    bit_vector[random_bit] = !bit_vector[random_bit];
    if (is_valid_individual()) {
      return;
    } else {
      // Not valid individual; undo mutation and try again.
      bit_vector[random_bit] = !bit_vector[random_bit];
    }
  }
}

Individual Individual::generate_random_individual() {
  Individual individual = Individual();
  int number_of_hidden_layers = get_random(Individual::MIN_HIDDEN_LAYERS, Individual::MAX_HIDDEN_LAYERS);
  individual.set_number_of_layers(number_of_hidden_layers);
  for (int i = 1; i <= number_of_hidden_layers; ++i) {
    individual.set_neurons_for_layer(i, get_random(Individual::MIN_NUMBER_NEURONS, Individual::MAX_NUMBER_NEURONS));
  }
  individual.set_training_time(get_random(Individual::MIN_TRAINING_TIME, Individual::MAX_TRAINING_TIME));
  individual.set_learning_rate(get_random(Individual::MIN_LEARNING_RATE, Individual::MAX_LEARNING_RATE));
  individual.set_momentum(get_random(Individual::MIN_MOMENTUM, Individual::MAX_MOMENTUM));
  assert(individual.is_valid_individual());
  return individual;
}

std::vector<Individual> Individual::generate_random_population(unsigned size) {
  std::vector<Individual> population;
  std::unordered_set<std::bitset<Individual::MOMENTUM_END + 1>> hash_set;
  while (population.size() != size) {
    Individual new_individual = generate_random_individual();
    if (hash_set.find(new_individual.get_vector()) == hash_set.end()) {
      population.push_back(new_individual);
    }
  }
  return population;
}
