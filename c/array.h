#ifndef ARRAY_H
#define ARRAY_H

#include <stdlib.h>

#define DTYPE float

struct Array {
  size_t rows;
  size_t cols;
  DTYPE **arr;
};

void arr_print(struct Array);

struct Array *arr_zeros(size_t, size_t);

void arr_free(struct Array **);

#endif /* ARRAY_H */
