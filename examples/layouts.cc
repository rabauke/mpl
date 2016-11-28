#include <cstdlib>
#include <iostream>
#include <list>
#include <vector>
#include <mpl/mpl.hpp>

int main() {
  const mpl::communicator &comm_world=mpl::environment::comm_world();
  mpl::indexed_block_layout<double> l1(3, {0, 4, 9, 15});
  mpl::indexed_layout<double> l2({{7, 70}, {8, 80}, {9, 90}});
  return EXIT_SUCCESS;
}
