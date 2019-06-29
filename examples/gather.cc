#include <cstdlib>
#include <iostream>
#include <vector>
#include <mpl/mpl.hpp>

int main() {
  const mpl::communicator &comm_world(mpl::environment::comm_world());
  int C_rank(comm_world.rank()), C_size(comm_world.size());
  // gather a single int from all ranks to rank root=0
  {
    int root(0);
    int x(C_rank + 1);
    std::vector<int> y(C_rank == root ? C_size : 0);
    comm_world.gather(root, x, y.data());
    if (C_rank == root) {
      for (int i = 0; i < C_size; ++i)
        std::cout << y[i] << ' ';
      std::cout << "\n";
    }
  }
  // gather a single int from all ranks to rank root=0
  // root and non-root rank use different function overloads of gather
  {
    int root(0);
    int x(-(C_rank + 1));
    if (C_rank == root) {
      std::vector<int> y(C_size);
      comm_world.gather(root, x, y.data());
      for (int i = 0; i < C_size; ++i)
        std::cout << y[i] << ' ';
      std::cout << "\n";
    } else
      comm_world.gather(root, x);
  }
  // gather several ints from all ranks to rank root=0
  {
    int root(0), n = 3;
    std::vector<int> x(n, C_rank + 1);
    std::vector<int> y(C_rank == root ? n * C_size : 0);
    mpl::contiguous_layout<int> l(n);
    comm_world.gather(root, x.data(), l, y.data(), l);
    if (C_rank == root) {
      for (int i = 0; i < C_size * n; ++i)
        std::cout << y[i] << ' ';
      std::cout << "\n";
    }
  }
  return EXIT_SUCCESS;
}
