#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <mpl/mpl.hpp>

template<typename I>
void print_range(const char * const str, I i1, I i2) {
  std::cout << str;
  while (i1!=i2) {
    std::cout << (*i1);
    ++i1;
    std::cout << ((i1!=i2) ? ' ' : '\n');
  }
}

int main() {
  const mpl::communicator &comm_world=mpl::environment::comm_world();
  if (comm_world.size()<2) 
    return EXIT_FAILURE;
  const int n=12;
  std::vector<int> v(n);
  mpl::contiguous_layout<int> l(n);
  if (comm_world.rank()==0) {
    std::iota(v.begin(), v.end(), 0);
    auto add_one=[](int x){ return x+1; };
    comm_world.send(v.data(), l, 1);
    std::transform(v.begin(), v.end(), v.begin(), add_one);
    {
      int size={ comm_world.bsend_size(l) };
      mpl::bsend_buffer<> buff(size);
      comm_world.bsend(v.data(), l, 1);
    }
    std::transform(v.begin(), v.end(), v.begin(), add_one);
    comm_world.ssend(v.data(), l, 1);
    std::transform(v.begin(), v.end(), v.begin(), add_one);
    comm_world.rsend(v.data(), l, 1);
  }
  if (comm_world.rank()==1) {
    comm_world.recv(v.data(), l, 0);
    print_range("v = ", v.begin(), v.end());
    comm_world.recv(v.data(), l, 0);
    print_range("v = ", v.begin(), v.end());
    comm_world.recv(v.data(), l, 0);
    print_range("v = ", v.begin(), v.end());
    comm_world.recv(v.data(), l, 0);
    print_range("v = ", v.begin(), v.end());
  }
  return EXIT_SUCCESS;
}
