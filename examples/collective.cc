#include <cstdlib>
#include <iostream>
#include <numeric>
#include <mpl/mpl.hpp>

int main() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  int root = 0;
  // synchronize processes via barrier
  comm_world.barrier();
  std::cout << mpl::environment::processor_name() << " has passed barrier\n";
  comm_world.barrier();
  double x = 0;
  if (comm_world.rank() == root)
    x = 10;
  // broadcast x to all from root rank
  comm_world.bcast(root, x);
  std::cout << "x = " << x << '\n';
  // collect data from all ranks via gather to root rank
  x = comm_world.rank() + 1;
  if (comm_world.rank() == root) {
    std::vector<double> v(comm_world.size());  // receive buffer
    comm_world.gather(root, x, v.data());
    std::cout << "v = ";
    for (auto x : v)
      std::cout << x << ' ';
    std::cout << '\n';
  } else
    comm_world.gather(root, x);
  // send data to all ranks via scatter from root rank
  double y = 0;
  if (comm_world.rank() == root) {
    std::vector<double> v(comm_world.size());  // send buffer
    std::iota(v.begin(), v.end(), 1);          // populate send buffer
    comm_world.scatter(root, v.data(), y);
  } else
    comm_world.scatter(root, y);
  std::cout << "y = " << y << '\n';
  // reduce/sum all values of x on all nodes and send global result to root
  if (comm_world.rank() == root) {
    comm_world.reduce(mpl::plus<double>(), root, x, y);
    std::cout << "sum after reduce " << y << '\n';
  } else
    comm_world.reduce(mpl::plus<double>(), root, x);
  // reduce/multiply all values of x on all nodes and send global result to all
  comm_world.allreduce(mpl::multiplies<double>(), x, y);
  std::cout << "sum after allreduce " << y << '\n';
  return EXIT_SUCCESS;
}
