#include <cstdlib>
#include <iostream>
#include <list>
#include <vector>
#include <numeric>
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
  // run the program with two or more processes
  if (comm_world.size()<2)
    return EXIT_FAILURE;
  // test layout for a piece of contiguous memory
  if (comm_world.rank()==0) {
    std::vector<int> v(20);
    std::iota(v.begin(), v.end(), 1);  // fill vector with some data
    mpl::contiguous_layout<int> l(10);  // contiguous_layout with 10 elements
    comm_world.send(v.data(), l, 1);  // send data to rank 1
  }
  if (comm_world.rank()==1) {
    std::vector<int> v(20, 0);
    mpl::contiguous_layout<int> l(10);  // contiguous_layout with 10 elements
    comm_world.recv(v.data(), l, 0);  // receive data from rank 0
    print_range("v = ", v.begin(), v.end());
  }
  // test layout for pieces of contiguous memory (equally spaced blocks of constant size)
  // layouts on sending and receiving side may differ but must be compatible
  if (comm_world.rank()==0) {
    std::vector<int> v(20);
    std::iota(v.begin(), v.end(), 1);  // fill vector with some data
    mpl::contiguous_layout<int> l(3*4);  // contiguous_layout with 10 elements
    comm_world.send(v.data(), l, 1);  // send data to rank 1
  }
  if (comm_world.rank()==1) {
    std::vector<int> v(20, 0);
    mpl::strided_vector_layout<int> l(3,  // number of blocks
				      4,  // block length
				      6); // block spacing
    comm_world.recv(v.data(), l, 0);  // receive data from rank 0
    print_range("v = ", v.begin(), v.end());
  }
  // test layout for a sequence of blocks of memory of varying block length
  // layouts on sending and receiving side may differ but must be compatible
  if (comm_world.rank()==0) {
    std::vector<int> v(20);
    std::iota(v.begin(), v.end(), 1);  // fill vector with some data
    mpl::contiguous_layout<int> l(3+4+2);  // contiguous_layout with 9 elements
    comm_world.send(v.data(), l, 1);  // send data to rank 1
  }
  if (comm_world.rank()==1) {
    std::vector<int> v(20, 0);
    mpl::indexed_layout<int> l({
	{3, 1},  // 1st block of length 3 with displacement 1
        {4, 8},  // 2nd block of length 4 with displacement 8
        {2, 16}  // 3rd block of length 2 with displacement 16
      });
    comm_world.recv(v.data(), l, 0);  // receive data from rank 0
    print_range("v = ", v.begin(), v.end());
  }
  // test layout for a sequence of blocks of memory of constant block length
  // layouts on sending and receiving side may differ but must be compatible
  if (comm_world.rank()==0) {
    std::vector<int> v(20);
    std::iota(v.begin(), v.end(), 1);  // fill vector with some data
    mpl::contiguous_layout<int> l(3*3);  // contiguous_layout with 9 elements
    comm_world.send(v.data(), l, 1);  // send data to rank 1
  }
  if (comm_world.rank()==1) {
    std::vector<int> v(20, 0);
    mpl::indexed_block_layout<int> l(3, // block length
				     {1, 8, 12}  // block displacements
				     );
    comm_world.recv(v.data(), l, 0);  // receive data from rank 0
    print_range("v = ", v.begin(), v.end());
  }
  return EXIT_SUCCESS;
}
