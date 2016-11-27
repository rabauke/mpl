#include <cstdlib>
#include <iostream>
#include <mpl/mpl.hpp>

int main() {
  try {
    const mpl::communicator &comm_world=mpl::environment::comm_world();
    // run the program with two ore more processes
    if (comm_world.size()<2)
      return EXIT_FAILURE;
    // process 0 sends
    if (comm_world.rank()==0) {
      double x=1.23456;
      comm_world.send(x, 1);  // send x to rank 1 via standard send
      ++x;
      {
	// create a buffer for buffered send,
	// memory will be freed on leaving the scope
	int size={ comm_world.bsend_size<decltype(x)>() };
	mpl::bsend_buffer<> buff(size);
	// send x to rank 1 via buffered send
	comm_world.bsend(x, 1);
      }
      ++x;
      comm_world.ssend(x, 1);  // send x to rank 1 via synchronous send
      ++x;
      comm_world.rsend(x, 1);  // send x to rank 1 via ready send
    }
    // process 1 recieves
    if (comm_world.rank()==1) {
      double x;
      comm_world.recv(x, 0);  // receive x from rank 0
      std::cout << "x = " << x << '\n';
      comm_world.recv(x, 0);  // receive x from rank 0
      std::cout << "x = " << x << '\n';
      comm_world.recv(x, 0);  // receive x from rank 0
      std::cout << "x = " << x << '\n';
      comm_world.recv(x, 0);  // receive x from rank 0
      std::cout << "x = " << x << '\n';
    }
  }
  catch (std::exception &err) {
    std::cerr << err.what() << std::endl;
  }
  return EXIT_SUCCESS;
}
