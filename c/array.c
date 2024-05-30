#include "array.h"

#include <stdio.h>

void arr_print(struct Array arr) {
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

struct Array *arr_zeros(size_t rows, size_t cols) {
  struct Array *arr = (struct Array *)malloc(sizeof(struct Array));
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

void arr_free(struct Array **arr) {
  for (size_t r = 0; r < (*arr)->rows; r++) {
    free((*arr)->arr[r]);
  }
  free((*arr)->arr);

  free((*arr));
}

int main(void) {
  struct Array *arr = arr_zeros(2, 2);
  if (arr == NULL) {
    printf("Memory error!\n");
    return 0;
  }

  arr_print(*arr);

  arr_free(&arr);

  return 0;
}
