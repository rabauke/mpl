#include <cstdlib>
#include <iostream>
#include <mpl/mpl.hpp>

template<std::size_t dim, typename T, typename A>
void update_overlap(const mpl::cart_communicator &cart_communicator,
                    mpl::distributed_grid<dim, T, A> &grid, mpl::tag_t tag = mpl::tag_t()) {
  mpl::shift_ranks ranks;
  for (std::size_t i{0}; i < dim; ++i) {
    // send to left
    ranks = cart_communicator.shift(i, -1);
    cart_communicator.sendrecv(grid.data(), grid.left_border_layout(i), ranks.dest, tag,
                               grid.data(), grid.right_mirror_layout(i), ranks.source, tag);
    // send to right
    ranks = cart_communicator.shift(i, +1);
    cart_communicator.sendrecv(grid.data(), grid.right_border_layout(i), ranks.dest, tag,
                               grid.data(), grid.left_mirror_layout(i), ranks.source, tag);
  }
}


int main() {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  {
    // build a one-dimensional Cartesian communicator
    // Cartesian is non-cyclic
    mpl::cart_communicator::sizes sizes{{{0, mpl::cart_communicator::nonperiodic}}};
    mpl::cart_communicator comm_c{comm_world, mpl::dims_create(comm_world.size(), sizes)};
    // create a distributed grid of 31 total grid points and 2 shadow grid points
    // to mirror data between adjacent processes
    mpl::distributed_grid<1, int> grid{comm_c, {{31, 2}}};
    // fill local grid including shadow grid points
    for (auto i{grid.obegin(0)}, i_end{grid.oend(0)}; i < i_end; ++i)
      grid(i) = comm_c.rank();
    // get shadow data from adjacent processes
    update_overlap(comm_c, grid);
    // print local grid including shadow grid points
    for (int k{0}; k < comm_c.size(); ++k) {
      if (k == comm_c.rank()) {
        for (auto i{grid.obegin(0)}, i_end{grid.oend(0)}; i < i_end; ++i)
          std::cout << grid(i);
        std::cout << std::endl;
      }
      comm_c.barrier();  // barrier may avoid overlapping output
    }
  }
  {
    // build a two-dimensional Cartesian communicator
    // Cartesian is cyclic along 1st dimension, non-cyclic along 2nd dimension
    mpl::cart_communicator::sizes sizes{
        {{0, mpl::cart_communicator::periodic}, {0, mpl::cart_communicator::nonperiodic}}};
    mpl::cart_communicator comm_c{comm_world, mpl::dims_create(comm_world.size(), sizes)};
    // create a distributed grid of 11x13 total grid points and 2 respectively 1
    // shadow grid points to mirror data between adjacent processes
    mpl::distributed_grid<2, int> grid{comm_c, {{11, 2}, {13, 1}}};
    // fill local grid including shadow grid points
    for (auto j{grid.obegin(1)}, j_end{grid.oend(1)}; j < j_end; ++j)
      for (auto i{grid.obegin(0)}, i_end{grid.oend(0)}; i < i_end; ++i)
        grid(i, j) = comm_c.rank();
    // get shadow data from adjacent processes
    update_overlap(comm_c, grid);
    // print local grid including shadow grid points
    for (int k{0}; k < comm_c.size(); ++k) {
      if (k == comm_c.rank()) {
        std::cout << std::endl;
        for (auto j{grid.obegin(1)}, j_end{grid.oend(1)}; j < j_end; ++j) {
          for (auto i{grid.obegin(0)}, i_end{grid.oend(0)}; i < i_end; ++i)
            std::cout << grid(i, j);
          std::cout << std::endl;
        }
      }
      comm_c.barrier();  // barrier may avoid overlapping output
    }
  }
  return EXIT_SUCCESS;
}
