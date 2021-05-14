#include <cstdlib>
#include <iostream>
#include <mpl/mpl.hpp>

int main() {
  // check communicator properties of comm_world
  const mpl::communicator &comm_world(mpl::environment::comm_world());
  std::cout << "comm_world  rank: " << comm_world.rank() << "\tsize: " << comm_world.size()
            << std::endl;
  comm_world.barrier();
  // split comm_world into 3 disjoint communicators
  // and carry out some collective communication
  mpl::communicator comm_3(mpl::communicator::split, comm_world, comm_world.rank() % 3);
  int key;
  if (comm_3.rank() == 0)
    key = comm_world.rank() % 3;
  comm_3.bcast(0, key);
  std::cout << "comm_3     rank: " << comm_3.rank() << "\tsize: " << comm_3.size()
            << "\tkey: " << key << std::endl;
  comm_world.barrier();
  // split comm_world into a communicator which contains all processes
  // except rank 0 of comm_world and carry out some collective communication
  mpl::communicator comm_without_0(mpl::communicator::split, comm_world,
                                   comm_world.rank() == 0 ? mpl::undefined : 1);
  if (comm_world.rank() != 0) {
    double data = 1;
    comm_without_0.allreduce(mpl::plus<double>(), data);
    std::cout << "sum: " << data << std::endl;
  }
  comm_world.barrier();
  std::cout << "comm_world  rank: " << comm_world.rank()
            << "\tcomm valid: " << (comm_without_0.is_valid() ? "yes" : "no") << std::endl;
  return EXIT_SUCCESS;
}
