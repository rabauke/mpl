#include <cstdlib>
#include <iostream>
#include <mpl/mpl.hpp>

int main() {
  const mpl::communicator & comm_world(mpl::environment::comm_world());
  int C_rank(comm_world.rank()), C_size(comm_world.size());
  std::vector<int> x;
  for (int i=0; i<C_rank+1; ++i)
    x.push_back(C_rank+1);
  mpl::contiguous_layout<int> l(C_rank+1);
  if (C_rank==0) {
    mpl::layouts<int> ls;
    for (int i=0; i<C_size; ++i)
      ls.push_back(mpl::contiguous_layout<int>(i+1));
    std::vector<int> y((C_size*C_size+C_size)/2, -8);
    comm_world.gatherv(0, x.data(), l, y.data(), ls);
    for (auto f: y)
      std::cout << f << '\n';
  } else {
    comm_world.gatherv(0, x.data(), l);
  }
  return EXIT_SUCCESS;
}
