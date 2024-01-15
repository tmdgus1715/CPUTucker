#ifndef HELPER_HPP_
#define HELPER_HPP_

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <type_traits>  // std::remove_pointer

namespace supertensor {
namespace cputucker {

// for Colorful print on terminal
#define RED "\x1B[31m"  // red
#define GRN "\x1B[32m"  // green
#define YEL "\x1B[33m"  // yellow
#define BLU "\x1B[34m"  // blue
#define MAG "\x1B[35m"  // magenta
#define CYN "\x1B[36m"  // cyan
#define WHT "\x1B[37m"  // white
#define RESET "\x1B[0m"

#define MYDEBUG(Fmt, ...) \
  { printf(BLU "\t[%s] " GRN Fmt RESET, __FUNCTION__, ##__VA_ARGS__); }
#define MYDEBUG_1(Fmt, ...) \
  { printf(GRN Fmt RESET, ##__VA_ARGS__); }
#define MYPRINT(Fmt, ...) \
  { printf(CYN Fmt RESET, ##__VA_ARGS__); }

inline void PrintLine() {
  std::cout << "-----------------------------" << std::endl;
}

inline std::string make_error_log(std::string msg, char const *file,
                                  char const *function, std::size_t line) {
  return std::string{"\n\n" RED} + file + "(" + std::to_string(line) + "): [" +
         function + "] \n\t" + msg + "\n\n" RESET;
}

#define ERROR_LOG(...) make_error_log(__VA_ARGS__, __FILE__, __func__, __LINE__)

#define GTUCKER_REMOVE_POINTER_TYPE_ALIAS(Type) \
  typename std::remove_pointer<Type>::type

template <typename T>
T *allocate(size_t num) {
  T *ptr = static_cast<T *>(malloc(sizeof(T) * num));
  if (ptr == NULL) {
    throw std::runtime_error(
        std::string("Memory Allocation ERROR \n\t [ptr == NULL]"));
  }
  return ptr;
}

template <typename T>
void deallocate(T *ptr) {
  free(ptr);
}

template <typename T>
T frand(T x, T y) {
  return ((y - x) * (static_cast<T>(rand()) / RAND_MAX)) + x;
}  // return the random value in (x, y) interval

template <typename T>
T abs(T x) {
  return x > 0 ? x : -x;
}

}  // namespace cputucker
}  // namespace supertensor
#endif  // HELPER_HPP_