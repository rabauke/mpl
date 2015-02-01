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
  const int tag=29;
  if (comm_world.rank()==0) {
    const int n=12;
    std::vector<int> v(n);
    mpl::contiguous_layout<int> l(n);
    std::iota(v.begin(), v.end(), 0);
    comm_world.send(v.data(), l, 1, tag);
  }
  if (comm_world.rank()==1) {
    mpl::status s(comm_world.probe(mpl::environment::any_source(), mpl::environment::any_tag()));
    int n(s.get_count<int>()), source(s.source()), tag(s.tag());
    std::cerr << "souce : " << s.source() << '\n'
	      << "tag   : " << s.tag() << '\n'
	      << "error : " << s.error() << '\n'
	      << "count : " << n << '\n';
    std::vector<int> v(n);
    mpl::contiguous_layout<int> l(n);
    comm_world.recv(v.data(), l, source, tag);
    print_range("v = ", v.begin(), v.end());
  }
  return EXIT_SUCCESS;
}
