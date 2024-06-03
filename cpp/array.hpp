#ifndef ARRAY_H
#define ARRAY_H

#include <iostream>
#include <vector>

#define DTYPE float

class Array {
  private:
    size_t rows_;
    size_t cols_;
    std::vector<std::vector<DTYPE>> arr;

  public:
    Array(size_t rows, size_t cols, DTYPE num) : rows_{rows}, cols_{cols} { 
      for (size_t r = 0; r < rows; r++) {
        std::vector<DTYPE> arr_tmp;
        for (size_t c = 0; c < cols; c++) {
          arr_tmp.push_back(num);
        }
        arr.push_back(arr_tmp);
      }
    }

    Array(Array &) = default;

    ~Array() = default;

    friend std::ostream &operator<<(std::ostream &os, Array array) {
      std::string arr_str = "[\n";
      for (auto &r : array.arr) {
        arr_str.append("[");
        for (auto &c : r) {
          //arr_str.append(c);
          arr_str.append("0"); // tmp
          arr_str.append(",");
        }
        arr_str.append("]\n");
      }
      arr_str.append("]");
      return os << "Array: " << arr_str;
    }

    size_t getRows(void) const { return rows_; }
    size_t getCols(void) const { return cols_; }
};

#endif // ARRAY_H 
