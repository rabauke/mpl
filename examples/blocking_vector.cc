#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <mpl/mpl.hpp>


template<typename I>
void print_range(const char *const str, I i_1, I i_2) {
  std::cout << str;
  while (i_1 != i_2) {
    std::cout << (*i_1);
    ++i_1;
    std::cout << ((i_1 != i_2) ? ' ' : '\n');
  }
}


int main() {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  // run the program with two or more processes
  if (comm_world.size() < 2)
    return EXIT_FAILURE;
  const int n{12};
  std::vector<int> v(n);             // vector of n elements lying contiguously in memory
  mpl::contiguous_layout<int> l(n);  // corresponding memory layout
  // process 0 sends
  if (comm_world.rank() == 0) {
    // see MPI Standard for the semantics of standard send, buffered send,
    // synchronous send and ready sendG
    std::iota(v.begin(), v.end(), 0);  // fill vector with some data
    auto add_one{[](int x) { return x + 1; }};
    comm_world.send(v.data(), l, 1);  // send vector to rank 1 via standard send
    std::transform(v.begin(), v.end(), v.begin(), add_one);  // update data
    {
      // create a buffer for buffered send,
      // memory will be freed on leaving the scope
      const int size{comm_world.bsend_size(l)};
      mpl::bsend_buffer buff{size};
      comm_world.bsend(v.data(), l, 1);  // send x to rank 1 via buffered send
    }
    std::transform(v.begin(), v.end(), v.begin(), add_one);  // update data
    comm_world.ssend(v.data(), l, 1);  // send x to rank 1 via synchronous send
    std::transform(v.begin(), v.end(), v.begin(), add_one);  // update data
    comm_world.rsend(v.data(), l, 1);                        // send x to rank 1 via ready send
  }
  // process 1 receives
  if (comm_world.rank() == 1) {
    comm_world.recv(v.data(), l, 0);  // receive vector from rank 0
    print_range("v = ", v.begin(), v.end());
    comm_world.recv(v.data(), l, 0);  // receive vector from rank 0
    print_range("v = ", v.begin(), v.end());
    comm_world.recv(v.data(), l, 0);  // receive vector from rank 0
    print_range("v = ", v.begin(), v.end());
    comm_world.recv(v.data(), l, 0);  // receive vector from rank 0
    print_range("v = ", v.begin(), v.end());
  }
  return EXIT_SUCCESS;
}
