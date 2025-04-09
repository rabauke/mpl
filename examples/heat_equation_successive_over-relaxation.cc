#include <cstdlib>
#include <iostream>
#include <cmath>
#include <tuple>
#include <random>
#include <mpl/mpl.hpp>

using double_2 = std::tuple<double, double>;


template<std::size_t dim, typename T, typename A>
void update_overlap(const mpl::cartesian_communicator &communicator,
                    mpl::distributed_grid<dim, T, A> &distributed_grid,
                    mpl::tag_t tag = mpl::tag_t()) {
  mpl::shift_ranks ranks;
  mpl::irequest_pool r;
  for (std::size_t i{0}; i < dim; ++i) {
    // send to left
    ranks = communicator.shift(i, -1);
    r.push(communicator.isend(distributed_grid.data(), distributed_grid.left_border_layout(i),
                              ranks.destination, tag));
    r.push(communicator.irecv(distributed_grid.data(), distributed_grid.right_mirror_layout(i),
                              ranks.source, tag));
    // send to right
    ranks = communicator.shift(i, +1);
    r.push(communicator.isend(distributed_grid.data(), distributed_grid.right_border_layout(i),
                              ranks.destination, tag));
    r.push(communicator.irecv(distributed_grid.data(), distributed_grid.left_mirror_layout(i),
                              ranks.source, tag));
  }
  r.waitall();
}


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
  // world communicator
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  // construct a two-dimensional Cartesian communicator with no periodic boundary conditions
  mpl::cartesian_communicator::dimensions size{mpl::cartesian_communicator::non_periodic,
                                               mpl::cartesian_communicator::non_periodic};
  mpl::cartesian_communicator comm_c(comm_world, mpl::dims_create(comm_world.size(), size));
  // total number of inner grid points
  const int n_x{768}, n_y{512};
  // grid points with extremal indices (-1, Nx or Ny) hold fixed boundary data
  // grid lengths and grid spacings
  const double l_x{1.5}, l_y{1};
  const double dx{l_x / (n_x + 1)}, dy{l_y / (n_y + 1)};
  // distributed grid that holds each processor's subgrid plus one row and
  // one column of neighboring data
  mpl::distributed_grid<2, double> u_d{comm_c, {{n_x, 1}, {n_y, 1}}};
  // rank 0 initializes with some random data
  if (comm_c.rank() == 0) {
    std::default_random_engine engine;
    std::uniform_real_distribution<double> uniform{0, 1};
    // local grid to store the whole set of inner grid points
    mpl::local_grid<2, double> u{comm_c, {n_x, n_y}};
    for (auto j{u.begin(1)}, j_end{u.end(1)}; j < j_end; ++j)
      for (auto i{u.begin(0)}, i_end{u.end(0)}; i < i_end; ++i)
        u(i, j) = uniform(engine);
    // scatter data to each processors subgrid
    scatter(comm_c, 0, u, u_d);
  } else
    scatter(comm_c, 0, u_d);
  // initialize boundary data, loop with obegin and oend over all
  // data including the overlap
  for (auto j : {u_d.obegin(1), u_d.oend(1) - 1})
    for (auto i{u_d.obegin(0)}, i_end{u_d.oend(0)}; i < i_end; ++i) {
      if (u_d.gindex(0, i) < 0 or u_d.gindex(1, j) < 0)
        u_d(i, j) = 1;  // left boundary condition
      if (u_d.gindex(0, i) >= n_x or u_d.gindex(1, j) >= n_y)
        u_d(i, j) = 0;  // right boundary condition
    }
  for (auto i : {u_d.obegin(0), u_d.oend(0) - 1})
    for (auto j{u_d.obegin(1)}, j_end{u_d.oend(1)}; j < j_end; ++j) {
      if (u_d.gindex(0, i) < 0 or u_d.gindex(1, j) < 0)
        u_d(i, j) = 1;  // lower boundary condition
      if (u_d.gindex(0, i) >= n_x or u_d.gindex(1, j) >= n_y)
        u_d(i, j) = 0;  // upper boundary condition
    }
  double w{1.875};  // the over-relaxation parameter
  double dx_2{dx * dx}, dy_2{dy * dy};
  // loop until converged
  bool converged{false};
  while (not converged) {
    // exchange overlap data
    update_overlap(comm_c, u_d);
    // apply one successive over-relaxation step
    double delta_u{0}, sum_u{0};
    for (auto j{u_d.begin(1)}, j_end{u_d.end(1)}; j < j_end; ++j)
      for (auto i{u_d.begin(0)}, i_end{u_d.end(0)}; i < i_end; ++i) {
        double du = -w * u_d(i, j) + w *
                                         (dy_2 * (u_d(i - 1, j) + u_d(i + 1, j)) +
                                          dx_2 * (u_d(i, j - 1) + u_d(i, j + 1))) /
                                         (2 * (dx_2 + dy_2));
        u_d(i, j) += du;
        delta_u += std::abs(du);
        sum_u += std::abs(u_d(i, j));
      }
    // determine global sum of Delta_u and sum_u and distribute to all processors
    double_2 delta_sum_u{delta_u, sum_u};  // pack into pair
    // use a lambda function for global reduction
    comm_c.allreduce(
        [](double_2 a, double_2 b) {
          // reduction adds component-by-component
          return double_2{std::get<0>(a) + std::get<0>(b), std::get<1>(a) + std::get<1>(b)};
        },
        delta_sum_u);
    std::tie(delta_u, sum_u) = delta_sum_u;  // unpack from pair
    converged = delta_u / sum_u < 1e-6;      // check for convergence
  }
  if (comm_c.rank() == 0) {
    // local grid to store the whole set of inner grid points
    mpl::local_grid<2, double> u{comm_c, {n_x, n_y}};
    // gather data and print result
    gather(comm_c, 0, u_d, u);
    for (auto j{u.begin(1)}, j_end{u.end(1)}; j < j_end; ++j) {
      for (auto i{u.begin(0)}, i_end{u.end(0)}; i < i_end; ++i)
        std::cout << u(i, j) << '\t';
      std::cout << '\n';
    }
  } else
    gather(comm_c, 0, u_d);
  return EXIT_SUCCESS;
}
