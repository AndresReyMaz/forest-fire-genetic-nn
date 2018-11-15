#include <assert.h>
#include <bitset>

#include "individual.h"

const int Individual::MOMENTUM_END; // circumvents strange linker error

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
