#ifndef ARRAY_H
#define ARRAY_H

#include <stdio.h>
#include <stdlib.h>

#define DTYPE float

struct Arr {
	size_t rows;
	size_t cols;
	DTYPE **arr;
};

struct Arr *arr_zeros(size_t, size_t);
//struct Arr *arr_zeros_like(size_t, size_t);
//struct Arr *arr_ones(size_t, size_t);
//struct Arr *arr_ones_like(size_t, size_t);
void arr_print(struct Arr);
void arr_free(struct Arr **);

#endif // ARRAY_H 
