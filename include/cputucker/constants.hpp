#ifndef CONSTANTS_HPP_
#define CONSTANTS_HPP_


#ifdef AVX
#include <immintrin.h>
#endif


namespace supertensor {
namespace cputucker {
namespace constants {
constexpr int kMaxOrder{8};
constexpr int kMaxIteration{2};
constexpr double kLambda{0.0001f};
}  // namespace constants

}  // namespace cputucker
}  // namespace supertensor
#endif /* CONSTANTS_HPP_ */