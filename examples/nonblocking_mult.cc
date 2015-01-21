#include <cstdlib>
#include <iostream>
#include <mpl/mpl.hpp>

int main() {
  const mpl::communicator &comm_world=mpl::environment::comm_world();
  std::vector<double> v(comm_world.size());
  double x=1.23456+comm_world.rank();
  mpl::irequest r_send(comm_world.isend(x, 0));
  if (comm_world.rank()==0) {
    mpl::irequest_pool r_pool;
    for (int i=0; i<comm_world.size(); ++i) 
      r_pool.push(comm_world.irecv(v[i], i));
    r_pool.waitall();
    for (int i=0; i<comm_world.size(); ++i) 
      std::cout << i << '\t' << v[i] << '\n';
  }
  r_send.wait();
  return EXIT_SUCCESS;
}
