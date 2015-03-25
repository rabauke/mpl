#include <cstdlib>
#include <iostream>
#include <mpl/mpl.hpp>

int main() {
  const mpl::communicator & comm_world(mpl::environment::comm_world());
  {
    mpl::cart_communicator::sizes sizes( {{0, false}} );
    mpl::cart_communicator comm_c(comm_world, 
				  mpl::dims_create(comm_world.size(), sizes));
    mpl::distributed_grid<1, int> G(comm_c, { {31, 2} });
    for (auto i=G.obegin(0), i_end=G.oend(0); i<i_end; ++i)
      G(i)=comm_c.rank();
    G.update_overlap();
    for (auto i=G.obegin(0), i_end=G.oend(0); i<i_end; ++i)
      std::cout << G(i);
    std::cout << std::endl;
  }
  {
    mpl::cart_communicator::sizes sizes( {{0, true}, {0, false}} );
    mpl::cart_communicator comm_c(comm_world, 
				  mpl::dims_create(comm_world.size(), sizes));
    mpl::distributed_grid<2, int> G(comm_c, 
				    {{11, 2}, {13, 1}});
    for (auto j=G.obegin(1), j_end=G.oend(1); j<j_end; ++j)
      for (auto i=G.obegin(0), i_end=G.oend(0); i<i_end; ++i)
	G(i, j)=comm_c.rank();
    G.update_overlap();
    for (int i=0; i<comm_c.size(); ++i) {
      if (i==comm_c.rank()) {
	std::cout << std::endl;
	for (auto j=G.obegin(1), j_end=G.oend(1); j<j_end; ++j) {
	  for (auto i=G.obegin(0), i_end=G.oend(0); i<i_end; ++i) 
	    std::cout << G(i, j);
	  std::cout << std::endl;
	}
      }
      comm_c.barrier();
    }
  }
  return EXIT_SUCCESS;
}
