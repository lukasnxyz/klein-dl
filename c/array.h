#ifndef ARRAY_H
#define ARRAY_H

#include <iostream>

#define DTYPE float

class Array {
  private:
    size_t rows_;
    size_t cols_;
    // conditional based on scalar, vector, or matrix
    //std::vector<> arr;

  public:
    Array(size_t rows, size_t cols, DTYPE num) : rows_{rows}, cols_{cols} { 
      for (size_t i = 0; i < rows
    }

    Array(Array &) = default;

    ~Array() = default;

    size_t getRows(void) const { return rows_; }
    size_t getCols(void) const { return cols_; }

    void print(void); // also make << print function
};

#endif // ARRAY_H 
