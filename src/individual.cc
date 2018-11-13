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
