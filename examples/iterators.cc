#include <cstdlib>
#include <list>
#include <vector>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <mpl/mpl.hpp>


int main() {
  const mpl::communicator &comm_world=mpl::environment::comm_world();
  // run the program with two or more processes
  if (comm_world.size()<2)
    comm_world.abort(EXIT_FAILURE);
  // send / receive a single vector
  {
    const int N=10;
    std::vector<double> l(N);
    if (comm_world.rank()==0) {
      std::iota(l.begin(), l.end(), 1);
      comm_world.send(l.begin(), l.end(), 1);
    }
    if (comm_world.rank()==1) {
      comm_world.recv(l.begin(), l.end(), 0);
      std::for_each(l.begin(), l.end(), [](auto x) {
	std::cout << x << '\n';
	});
    }
  }
  // send / receive a single list
  {
    const int N=10;
    std::list<double> l(N);
    if (comm_world.rank()==0) {
      std::iota(l.begin(), l.end(), 1);
      comm_world.send(l.begin(), l.end(), 1);
    }
    if (comm_world.rank()==1) {
      comm_world.recv(l.begin(), l.end(), 0);
      std::for_each(l.begin(), l.end(), [](auto x) {
	std::cout << x << '\n';
	});
    }
  }
  return EXIT_SUCCESS;
}
