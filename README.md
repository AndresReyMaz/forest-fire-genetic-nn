# forest-fire-genetic-nn
#### A genetic algorithm for training a neural network that detects presence of forest fires in images.

The genetic algorithm presented here generates values for training a neural network using Weka.

To compile, run the following command in the `src` directory (tested under Ubuntu 16.04):

```
g++ -std=c++14 -W -Wall -Werror -Wextra -o program program.cc individual.cc reproduction_util.cc anyoption.cc random_util.cc mutation_util.cc system_util.cc -ggdb -I .
```

This generates a binary at `src/program` which can then be executed.
