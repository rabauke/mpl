#include <cstdlib>
#include <iostream>
// include MPL header file
#include <mpl/mpl.hpp>

int main() {
  // get a reference to communicator "world"
  const mpl::communicator &comm_world(mpl::environment::comm_world());
  // each process prints a message containg processor name, rank in
  // communicator world and size of communicator world
  // output may depend on MPI implementation
  std::cout << "Hello world! I am running on \"" << mpl::environment::processor_name()
            << "\". My rank is " << comm_world.rank() << " out of " << comm_world.size()
            << " processes.\n";
  return EXIT_SUCCESS;
}
