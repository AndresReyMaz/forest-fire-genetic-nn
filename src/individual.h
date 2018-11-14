#ifndef GENETIC_INDIVIDUAL_H
#define GENETIC_INDIVIDUAL_H

#include <assert.h>
#include <bitset>

const int N_BITS = 57;

class Individual {
 public:
  static const int MIN_HIDDEN_LAYERS = 0;
  static const int MAX_HIDDEN_LAYERS = 3;
  static const int MIN_NUMBER_NEURONS = 3;
  static const int MAX_NUMBER_NEURONS = 100;
  static const int MIN_TRAINING_TIME = 100;
  static const int MAX_TRAINING_TIME = 4000;
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
    assert(layer <= get_hidden_layers());
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
  
  void set_training_time(int training_time);
  void set_learning_rate(int learning_rate);
  void set_momentum(int momentum);
  void set_neurons(int value) {
    set_value(HIDDEN_LAYERS_START, NUMBER_NEURONS_END, value);
  }

  Individual() { };
  
  Individual(const Individual &other) {
    bit_vector = std::bitset<N_BITS>(other.get_vector().to_ullong());
  }
  
 private:
  std::bitset<N_BITS> bit_vector;
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

  int to_value(int start, int end) const;

  void set_value(int start, int end, int value);
};

#endif
