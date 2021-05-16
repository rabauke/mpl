#include <cstdlib>
#include <iostream>
#include <vector>
#include <mpl/mpl.hpp>

int main() {
  const mpl::communicator &comm_world(mpl::environment::comm_world());
  int C_rank(comm_world.rank()), C_size(comm_world.size());
  // fill vector with C_rank+1 elements, each having the value C_rank+1
  std::vector<int> x(C_rank + 1, C_rank + 1);
  mpl::contiguous_layout<int> l(C_rank + 1);
  // root rank will send and receive in gather operation
  if (C_rank == 0) {
    // messages of varying size will be received
    // need to specify appropriate memory layouts to define how many elements
    // will be received and where to store them
    mpl::layouts<int> ls;
    for (int i = 0; i < C_size; ++i)
      // define layout for message to be received from rank i
      ls.push_back(mpl::indexed_layout<int>({{
          i + 1,           // number of int elements
          (i * i + i) / 2  // position of the first element in receive buffer
      }}));
    std::vector<int> y((C_size * C_size + C_size) / 2);  // receive buffer
    comm_world.gatherv(0, x.data(), l, y.data(), ls);    // receive data
    // print data
    for (auto f : y)
      std::cout << f << '\n';
  } else
    // non-root ranks just send
    comm_world.gatherv(0, x.data(), l);
  return EXIT_SUCCESS;
}
