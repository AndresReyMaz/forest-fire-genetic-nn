#include <assert.h>
#include <bitset>

#include "individual.h"

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
  assert(training_time >= MIN_TRAINING_TIME and training_time <= MAX_TRAINING_TIME);
  Individual::set_value(TRAINING_TIME_START, TRAINING_TIME_END, training_time - MIN_TRAINING_TIME);
}

void Individual::set_learning_rate(int learning_rate) {
  assert(learning_rate >= MIN_LEARNING_RATE and learning_rate <= MAX_LEARNING_RATE);
  Individual::set_value(LEARNING_RATE_START, LEARNING_RATE_END, learning_rate - MIN_LEARNING_RATE);
}

void Individual::set_momentum(int momentum) {
  assert(momentum >= MIN_MOMENTUM and momentum <= MAX_MOMENTUM);
  Individual::set_value(MOMENTUM_START, MOMENTUM_END, momentum - MIN_MOMENTUM);
}
