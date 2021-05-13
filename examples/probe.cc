#include <cstdlib>
#include <iostream>
#include <vector>
#include <numeric>
#include <mpl/mpl.hpp>

template<typename I>
void print_range(const char *const str, I i1, I i2) {
  std::cout << str;
  while (i1 != i2) {
    std::cout << (*i1);
    ++i1;
    std::cout << ((i1 != i2) ? ' ' : '\n');
  }
}

int main() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  // run the program with two or more processes
  if (comm_world.size() < 2)
    return EXIT_FAILURE;
  if (comm_world.rank() == 0) {
    // send a message of n elements to rank 1
    enum class tag { send = 29 };
    const int n = 12;
    std::vector<int> v(n);
    mpl::contiguous_layout<int> l(n);
    std::iota(v.begin(), v.end(), 0);
    comm_world.send(v.data(), l, 1, tag::send);
  }
  if (comm_world.rank() == 1) {
    // receive a message of an a priory unknown number of elements from rank 0
    // first probe for a message from some arbitrary rank with any tag
    mpl::status s(comm_world.probe(mpl::any_source, mpl::tag::any()));
    // decode the number of elements, the source and the tag
    int n(s.get_count<int>()), source(s.source());
    mpl::tag tag(s.tag());
    std::cerr << "source : " << s.source() << '\n'
              << "tag    : " << s.tag() << '\n'
              << "error  : " << s.error() << '\n'
              << "count  : " << n << '\n';
    // reserve sufficient amount of memory to receive the message
    std::vector<int> v(n);
    mpl::contiguous_layout<int> l(n);
    // finally receive the message
    comm_world.recv(v.data(), l, source, tag);
    print_range("v = ", v.begin(), v.end());
  }
  return EXIT_SUCCESS;
}
