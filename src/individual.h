#ifndef GENETIC_INDIVIDUAL_H
#define GENETIC_INDIVIDUAL_H

#include <assert.h>
#include <bitset>
#include <random>
#include <utility>
#include <vector>

#include "random_util.h"

const int N_BITS = 57;

class Individual {
 public:
  static const int MIN_HIDDEN_LAYERS = 0;
  static const int MAX_HIDDEN_LAYERS = 3;
  static const int MIN_NUMBER_NEURONS = 3;
  static const int MAX_NUMBER_NEURONS = 100;
  static const int MIN_TRAINING_TIME = 0;
  static const int MAX_TRAINING_TIME = 195;
  static const int MIN_LEARNING_RATE = 1000;
  static const int MAX_LEARNING_RATE = 6000;
  static const int MIN_MOMENTUM = 1000;
  static const int MAX_MOMENTUM = 7000;

  std::bitset<N_BITS> get_vector() const {
    return bit_vector;
  }

  int get_hidden_layers() const {
    return to_value(HIDDEN_LAYERS_START, HIDDEN_LAYERS_END);
  }

  int get_neurons_for_layer(int layer) const {
    //assert(layer <= get_hidden_layers());
    return to_value(NUMBER_NEURONS_START + NUMBER_NEURONS_LENGTH * (layer - 1),
		    NUMBER_NEURONS_START + NUMBER_NEURONS_LENGTH * layer - 1) + MIN_NUMBER_NEURONS;
  }

  int get_training_time() const {
    return to_value(TRAINING_TIME_START, TRAINING_TIME_END) + MIN_TRAINING_TIME;
  }

  int get_learning_rate() const {
    return to_value(LEARNING_RATE_START, LEARNING_RATE_END) + MIN_LEARNING_RATE;
  }

  int get_momentum() const {
    return to_value(MOMENTUM_START, MOMENTUM_END) + MIN_MOMENTUM;
  }

  int get_neurons() const {
    // Note: this gets all the neuron information as a non-normalized number.
    return to_value(HIDDEN_LAYERS_START, NUMBER_NEURONS_END);
  }

  std::pair<int, int> get_training_time_crossover_point() {
    return {get_random(TRAINING_TIME_START + 1, TRAINING_TIME_END), TRAINING_TIME_END};
  }

  std::pair<int, int> get_learning_rate_crossover_point() {
    return {get_random(LEARNING_RATE_START + 1, LEARNING_RATE_END), LEARNING_RATE_END};
  }

  std::pair<int, int> get_momentum_crossover_point() {
    return {get_random(MOMENTUM_START + 1, MOMENTUM_END), MOMENTUM_END};
  }

  std::vector<int> get_hidden_layer_values() {
    int layers = get_hidden_layers();
    std::vector<int> values;
    for (int i = 1 ; i <= layers; ++i) {
      values.push_back(get_neurons_for_layer(i));
    }
    return values;
  }

  int get_precision() { return precision; }

  void set_training_time(int training_time);
  void set_learning_rate(int learning_rate);
  void set_momentum(int momentum);
  void set_number_of_layers(int number_of_layers);
  void set_neurons(int value) {
    set_value(HIDDEN_LAYERS_START, NUMBER_NEURONS_END, value);
  }
  void set_neurons_for_layer(int layer, int value);

  // Sets the 0-indexed layer to value.
  void set_hidden_layer(int layer, int value);

  // Sets the number of hidden layers to the length of the list and the values to those selected.
  void set_neurons(const std::vector<int>& neurons);

  void copy_bits_from(const Individual& other, const std::pair<int, int>& range) {
    // Copies the bits in the range from other to this.
    for (int i = range.first; i <= range.second; ++i) {
      bit_vector[i] = other.get_vector()[i];
    }
  }

  void copy_bits_from(const Individual& other, const std::pair<int, int>& this_range, const std::pair<int, int>& other_range) {
    // Copies the bits in other defined by other_range to this, in the range defined by this_range.
    assert(this_range.second - this_range.first == other_range.second - other_range.first);
    for (int i = this_range.first, j = other_range.first; i < this_range.second; ++i, ++j) {
      bit_vector[i] = other.get_vector()[j];
    }
  }

  bool is_valid_individual() {
    // Checks if the individual has valid values for all parameters.
    for (int i = 1; i <= 3; ++i) {
      int number_of_neurons = get_neurons_for_layer(i);
      if (number_of_neurons < MIN_NUMBER_NEURONS or number_of_neurons > MAX_NUMBER_NEURONS) {
	return false;
      }
    }
    int training_time = get_training_time();
    if (training_time < MIN_TRAINING_TIME or training_time > MAX_TRAINING_TIME) {
      return false;
    }
    int learning_rate = get_learning_rate();
    if (learning_rate < MIN_LEARNING_RATE or learning_rate > MAX_LEARNING_RATE) {
      return false;
    }
    int momentum = get_momentum();
    if (momentum < MIN_MOMENTUM or momentum > MAX_MOMENTUM) {
      return false;
    }
    return true;
  }

  // Mutates the individual by one bit to a valid representation.
  void mutate();

  // Randomly generates an individual.
  static Individual generate_random_individual();

  // Randomly generates a population of a given size.
  static std::vector<Individual> generate_random_population(unsigned size);

  // Runs the neural network parameters agains the Weka algorithm.
  void evaluate();

  Individual() : precision(0) { };

  Individual(const Individual &other) {
    bit_vector = std::bitset<N_BITS>(other.get_vector().to_ullong());
    precision = 0;
  }

  bool operator < (const Individual& other) const {
    return (precision > other.precision);
  }

  static const int HIDDEN_LAYERS_START = 0;
  static const int HIDDEN_LAYERS_END = HIDDEN_LAYERS_START + 1;
  static const int NUMBER_NEURONS_START = HIDDEN_LAYERS_END + 1;
  static const int NUMBER_NEURONS_LENGTH = 7;
  static const int NUMBER_NEURONS_END = NUMBER_NEURONS_START + NUMBER_NEURONS_LENGTH * MAX_HIDDEN_LAYERS - 1;
  static const int TRAINING_TIME_START = NUMBER_NEURONS_END + 1;
  static const int TRAINING_TIME_LENGTH = 8;
  static const int TRAINING_TIME_END = TRAINING_TIME_START + TRAINING_TIME_LENGTH - 1;
  static const int LEARNING_RATE_START = TRAINING_TIME_END + 1;
  static const int LEARNING_RATE_LENGTH = 13;
  static const int LEARNING_RATE_END = LEARNING_RATE_START + LEARNING_RATE_LENGTH - 1;
  static const int MOMENTUM_START = LEARNING_RATE_END + 1;
  static const int MOMENTUM_LENGTH = 13;
  static const int MOMENTUM_END = MOMENTUM_START + MOMENTUM_LENGTH - 1;

 private:
  std::bitset<N_BITS> bit_vector;

  // The global precision of the neural network defined by the individual.
  int precision;

  int to_value(int start, int end) const;

  void set_value(int start, int end, int value);

};

#endif
