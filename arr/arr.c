#include "array.h"

void arr_print(struct Arr arr) {
  printf("[\n");
  for (size_t r = 0; r < arr.rows; r++) {
    printf("[");
    for (size_t c = 0; c < arr.cols; c++) {
      printf("%.2f ", arr.arr[r][c]);
    }
    printf("]\n");
  }
  printf("]\n");
}

struct Arr *arr_zeros(size_t rows, size_t cols) {
  struct Arr *arr = (struct Arr *)malloc(sizeof(struct Arr));
  if (arr == NULL) {
    return NULL;
  }

  arr->rows = rows;
  arr->cols = cols;

  arr->arr = (DTYPE **)calloc(rows, sizeof(DTYPE *));
  if (arr->arr == NULL) {
    free(arr);
    return NULL;
  }

  for (size_t i = 0; i < rows; i++) {
    arr->arr[i] = (DTYPE *)calloc(cols, sizeof(DTYPE));
    if (arr->arr[i] == NULL) {
      for (size_t x = 0; x < i; x++) {
        free(arr->arr[x]);
      }

      free(arr);
      return NULL;
    }
  }

  return arr;
}

void arr_free(struct Arr **arr) {
  for (size_t r = 0; r < (*arr)->rows; r++) {
    free((*arr)->arr[r]);
  }
  free((*arr)->arr);

  free((*arr));
}
