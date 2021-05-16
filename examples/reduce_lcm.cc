#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <mpl/mpl.hpp>

// calculate least common multiple of two arguments
template<typename T>
class lcm {
  // helper: calculate greatest common divisor
  T gcd(T a, T b) {
    T zero = T(), t;
    if (a < zero)
      a = -a;
    if (b < zero)
      b = -b;
    while (b > zero) {
      t = a % b;
      a = b;
      b = t;
    }
    return a;
  }

public:
  T operator()(T a, T b) {
    T zero = T();
    T t((a / gcd(a, b)) * b);
    if (t < zero)
      return -t;
    return t;
  }
};

int main() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  // generate data
  std::mt19937_64 g(std::time(nullptr) * comm_world.rank());  // random seed
  std::uniform_int_distribution uniform(1, 12);
  const int n = 8;
  // populate vector with random data
  std::vector<int> v(n);
  std::generate(v.begin(), v.end(), [&g, &uniform]() { return uniform(g); });
  // calculate the least common multiple and send result to rank 0
  mpl::contiguous_layout<int> layout(n);
  if (comm_world.rank() == 0) {
    std::vector<int> result(n);
    // calculate least common multiple
    comm_world.reduce(lcm<int>(), 0, v.data(), result.data(), layout);
    // to check the result display data from all ranks
    std::cout << "Arguments:\n";
    for (int r = 0; r < comm_world.size(); ++r) {
      if (r > 0)
        comm_world.recv(v.data(), layout, r);
      for (auto i : v)
        std::cout << i << '\t';
      std::cout << '\n';
    }
    // display results of global reduction
    std::cout << "\nResults:\n";
    for (auto i : result)
      std::cout << i << '\t';
    std::cout << '\n';
  } else {
    // calculate least common multiple
    comm_world.reduce(lcm<int>(), 0, v.data(), layout);
    // send data to rank 0 for display
    comm_world.send(v.data(), layout, 0);
  }
  return EXIT_SUCCESS;
}
