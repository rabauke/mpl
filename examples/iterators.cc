#include <cstdlib>
#include <vector>
#include <list>
#include <set>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <mpl/mpl.hpp>


int main() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  // run the program with two or more processes
  if (comm_world.size() < 2)
    comm_world.abort(EXIT_FAILURE);
  // send / receive a single vector
  {
    const int N = 10;
    std::vector<double> l(N);
    if (comm_world.rank() == 0) {
      std::iota(begin(l), end(l), 1);
      comm_world.send(begin(l), end(l), 1);
    }
    if (comm_world.rank() == 1) {
      comm_world.recv(begin(l), end(l), 0);
      std::for_each(begin(l), end(l), [](double x) { std::cout << x << '\n'; });
    }
  }
  // send / receive a single list
  {
    const int N = 10;
    std::list<double> l(N);
    if (comm_world.rank() == 0) {
      std::iota(begin(l), end(l), 1);
      comm_world.send(begin(l), end(l), 1);
    }
    if (comm_world.rank() == 1) {
      comm_world.recv(begin(l), end(l), 0);
      std::for_each(begin(l), end(l), [](double x) { std::cout << x << '\n'; });
    }
  }
  // send a set / receive an array
  {
    const int N = 10;
    if (comm_world.rank() == 0) {
      std::set<double> s;
      for (int i = 1; i <= N; ++i)
        s.insert(i);
      comm_world.send(s.begin(), s.end(), 1);
    }
    if (comm_world.rank() == 1) {
      std::array<double, N> l;
      comm_world.recv(begin(l), end(l), 0);
      std::for_each(begin(l), end(l), [](double x) { std::cout << x << '\n'; });
    }
  }
  return EXIT_SUCCESS;
}
