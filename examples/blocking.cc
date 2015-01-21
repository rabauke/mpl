#include <cstdlib>
#include <iostream>
#include <mpl/mpl.hpp>

int main() {
  const mpl::communicator &comm_world=mpl::environment::comm_world();
  if (comm_world.size()<2) 
    return EXIT_FAILURE;
  if (comm_world.rank()==0) {
    double x=1.23456;
    comm_world.send(x, 1);
    ++x;
    {
      int size={ comm_world.bsend_size<decltype(x)>() };
      mpl::bsend_buffer<> buff(size);
      comm_world.bsend(x, 1);
    }
    ++x;
    comm_world.ssend(x, 1);
    ++x;
    comm_world.rsend(x, 1);
  }
  if (comm_world.rank()==1) {
    double x;
    comm_world.recv(x, 0);
    std::cout << "x = " << x << '\n';
    comm_world.recv(x, 0);
    std::cout << "x = " << x << '\n';
    comm_world.recv(x, 0);
    std::cout << "x = " << x << '\n';
    comm_world.recv(x, 0);
    std::cout << "x = " << x << '\n';
  }
  return EXIT_SUCCESS;
}
