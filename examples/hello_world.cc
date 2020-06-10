#include <cstdlib>
#include <iostream>
// include MPL header file
#include <mpl/mpl.hpp>

int main() {
  // get a reference to communicator "world"
  const mpl::communicator &comm_world(mpl::environment::comm_world());
  // each process prints a message containing the processor name, the rank
  // in communicator world and the size of communicator world
  // output may depend on MPI implementation
  std::cout << "Hello world! I am running on \"" << mpl::environment::processor_name()
            << "\". My rank is " << comm_world.rank() << " out of " << comm_world.size()
            << " processes.\n";
  // if there are two or more processes send a message from process 0 to process 1
  if (comm_world.size() >= 2) {
    if (comm_world.rank() == 0) {
      std::string message{"Hello world!"};
      comm_world.send(message, 1);  // send message to rank 1
    } else if (comm_world.rank() == 1) {
      std::string message;
      comm_world.recv(message, 0);  // receive message from rank 0
      std::cout << "got: \"" << message << "\"\n";
    }
  }
  return EXIT_SUCCESS;
}
