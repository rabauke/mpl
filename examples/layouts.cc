#include <cstdlib>
#include <iostream>
#include <list>
#include <vector>
#include <mpl/mpl.hpp>

int main() {
  const mpl::communicator &comm_world=mpl::environment::comm_world();
  std::list<int> L={7, 8, 9};
  mpl::indexed_block_layout<double> l1(100, L.begin(), L.end());
  mpl::indexed_block_layout<double> l2(100, {8, 8, 8, 8});
  std::vector<int> V={70, 80, 90};
  mpl::indexed_layout<double> l3(V.begin(), V.end(), L.begin(), L.end());
  return EXIT_SUCCESS;
}
