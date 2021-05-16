#include <cstdlib>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <random>
#include <mpl/mpl.hpp>

// data type to store data an position of global minimum
using pair_t = std::pair<double, int>;

int main() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  // generate data
  std::mt19937_64 g(std::time(nullptr) * comm_world.rank());  // random seed
  std::uniform_real_distribution<> uniform;
  const int n = 8;
  // populate vector with random data
  std::vector<pair_t> v(n);
  std::generate(v.begin(), v.end(), [&comm_world, &g, &uniform]() {
    return std::make_pair(uniform(g), comm_world.rank());
  });
  // calculate minimum and its location and send result to rank root
  int root = 0;
  mpl::contiguous_layout<pair_t> layout(n);
  if (comm_world.rank() == root) {
    std::vector<pair_t> result(n);
    // calculate minimum
    comm_world.reduce(mpl::min<pair_t>(), root, v.data(), result.data(), layout);
    // display data from all ranks
    std::cout << "arguments:\n";
    for (int r = 0; r < comm_world.size(); ++r) {
      if (r > 0)
        comm_world.recv(v.data(), layout, r);
      for (auto i : v)
        std::cout << std::fixed << std::setprecision(5) << i.first << ' ' << i.second << '\t';
      std::cout << '\n';
    }
    // display results of global reduction
    std::cout << "\nresults:\n";
    for (pair_t i : result)
      std::cout << std::fixed << std::setprecision(5) << i.first << ' ' << i.second << '\t';
    std::cout << '\n';
  } else {
    // calculate minimum and its location and send result to rank 0
    comm_world.reduce(mpl::min<pair_t>(), root, v.data(), layout);
    // send data to rank 0 for display
    comm_world.send(v.data(), layout, root);
  }
  return EXIT_SUCCESS;
}
