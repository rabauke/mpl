#include <cstdlib>
#include <iostream>
#include <mpl/mpl.hpp>

template<std::size_t dim, typename T, typename A>
void scatter(const mpl::cartesian_communicator &communicator, int root,
             const mpl::local_grid<dim, T, A> &local_grid,
             mpl::distributed_grid<dim, T, A> &distributed_grid) {
  communicator.scatterv(root, local_grid.data(), local_grid.sub_layouts(),
                        distributed_grid.data(), distributed_grid.interior_layout());
}

template<std::size_t dim, typename T, typename A>
void scatter(const mpl::cartesian_communicator &communicator, int root,
             mpl::distributed_grid<dim, T, A> &distributed_grid) {
  communicator.scatterv(root, distributed_grid.data(), distributed_grid.interior_layout());
}


template<std::size_t dim, typename T, typename A>
void gather(const mpl::cartesian_communicator &communicator, int root,
            const mpl::distributed_grid<dim, T, A> &distributed_grid,
            mpl::local_grid<dim, T, A> &local_grid) {
  communicator.gatherv(root, distributed_grid.data(), distributed_grid.interior_layout(),
                       local_grid.data(), local_grid.sub_layouts());
}

template<std::size_t dim, typename T, typename A>
void gather(const mpl::cartesian_communicator &communicator, int root,
            const mpl::distributed_grid<dim, T, A> &distributed_grid) {
  communicator.gatherv(root, distributed_grid.data(), distributed_grid.interior_layout());
}


int main() {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  mpl::cartesian_communicator::dimensions size{mpl::cartesian_communicator::periodic,
                                               mpl::cartesian_communicator::non_periodic};
  const int nx{21}, ny{13};
  mpl::cartesian_communicator comm_c{comm_world, mpl::dims_create(comm_world.size(), size)};
  mpl::distributed_grid<2, int> grid{comm_c, {{nx, 1}, {ny, 1}}};
  for (auto j{grid.obegin(1)}, j_end{grid.oend(1)}; j < j_end; ++j)
    for (auto i{grid.obegin(0)}, i_end{grid.oend(0)}; i < i_end; ++i)
      grid(i, j) = comm_c.rank();
  if (comm_world.rank() == 0) {
    mpl::local_grid<2, int> local_grid(comm_c, {nx, ny});
    for (auto j{local_grid.begin(1)}, j_end{local_grid.end(1)}; j < j_end; ++j)
      for (auto i{local_grid.begin(0)}, i_end{local_grid.end(0)}; i < i_end; ++i)
        local_grid(i, j) = 0;
    scatter(comm_c, 0, local_grid, grid);
  } else
    scatter(comm_c, 0, grid);
  for (int i{0}; i < comm_c.size(); ++i) {
    if (i == comm_c.rank()) {
      std::cout << std::endl;
      for (auto j{grid.obegin(1)}, j_end{grid.oend(1)}; j < j_end; ++j) {
        for (auto i{grid.obegin(0)}, i_end{grid.oend(0)}; i < i_end; ++i)
          std::cout << grid(i, j);
        std::cout << std::endl;
      }
    }
    comm_c.barrier();
  }
  for (auto j{grid.obegin(1)}, j_end{grid.oend(1)}; j < j_end; ++j)
    for (auto i{grid.obegin(0)}, i_end{grid.oend(0)}; i < i_end; ++i)
      grid(i, j) = comm_c.rank();
  if (comm_world.rank() == 0) {
    mpl::local_grid<2, int> local_grid{comm_c, {nx, ny}};
    gather(comm_c, 0, grid, local_grid);
    std::cout << std::endl;
    for (auto j{local_grid.begin(1)}, j_end{local_grid.end(1)}; j < j_end; ++j) {
      for (auto i{grid.begin(0)}, i_end{local_grid.end(0)}; i < i_end; ++i)
        std::cout << local_grid(i, j);
      std::cout << std::endl;
    }
  } else
    gather(comm_c, 0, grid);
  return EXIT_SUCCESS;
}
