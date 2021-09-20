#include <cstdlib>
#include <iostream>
#include <cmath>
#include <tuple>
#include <random>
#include <mpl/mpl.hpp>

using double_2 = std::tuple<double, double>;


template<std::size_t dim, typename T, typename A>
mpl::irequest_pool update_overlap(const mpl::cart_communicator &communicator,
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
  return r;
}


template<std::size_t dim, typename T, typename A>
void scatter(const mpl::cart_communicator &communicator, int root,
             const mpl::local_grid<dim, T, A> &local_grid,
             mpl::distributed_grid<dim, T, A> &distributed_grid) {
  communicator.scatterv(root, local_grid.data(), local_grid.sub_layouts(),
                        distributed_grid.data(), distributed_grid.interior_layout());
}

template<std::size_t dim, typename T, typename A>
void scatter(const mpl::cart_communicator &communicator, int root,
             mpl::distributed_grid<dim, T, A> &distributed_grid) {
  communicator.scatterv(root, distributed_grid.data(), distributed_grid.interior_layout());
}


template<std::size_t dim, typename T, typename A>
void gather(const mpl::cart_communicator &communicator, int root,
            const mpl::distributed_grid<dim, T, A> &distributed_grid,
            mpl::local_grid<dim, T, A> &local_grid) {
  communicator.gatherv(root, distributed_grid.data(), distributed_grid.interior_layout(),
                       local_grid.data(), local_grid.sub_layouts());
}

template<std::size_t dim, typename T, typename A>
void gather(const mpl::cart_communicator &communicator, int root,
            const mpl::distributed_grid<dim, T, A> &distributed_grid) {
  communicator.gatherv(root, distributed_grid.data(), distributed_grid.interior_layout());
}


int main() {
  // world communicator
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  // construct a two-dimensional Cartesian communicator with no periodic boundary conditions
  mpl::cart_communicator::sizes sizes{
      {{0, mpl::cart_communicator::nonperiodic}, {0, mpl::cart_communicator::nonperiodic}}};
  mpl::cart_communicator comm_c{comm_world, mpl::dims_create(comm_world.size(), sizes)};
  // total number of inner grid points
  const int n_x{768}, n_y{512};
  // grid points with extremal indices (-1, Nx or Ny) hold fixed boundary data
  // grid lengths and grid spacings
  double l_x{1.5}, l_y{1};
  double dx{l_x / (n_x + 1)}, dy{l_y / (n_y + 1)};
  // distributed grids that hold each processor's subgrid plus one row and
  // one column  of neighboring data
  mpl::distributed_grid<2, double> u_d_1(comm_c, {{n_x, 1}, {n_y, 1}});
  mpl::distributed_grid<2, double> u_d_2(comm_c, {{n_x, 1}, {n_y, 1}});
  // rank 0 initializes with some random data
  if (comm_c.rank() == 0) {
    std::default_random_engine engine;
    std::uniform_real_distribution<double> uniform{0, 1};
    // local grid to store the whole set of inner grid points
    mpl::local_grid<2, double> u(comm_c, {n_x, n_y});
    for (auto j{u.begin(1)}, j_end{u.end(1)}; j < j_end; ++j)
      for (auto i{u.begin(0)}, i_end{u.end(0)}; i < i_end; ++i)
        u(i, j) = uniform(engine);
    // scatter data to each processors subgrid
    scatter(comm_c, 0, u, u_d_1);
  } else
    scatter(comm_c, 0, u_d_1);
  // initialize boundary data, loop with obegin and oend over all
  // data including the overlap, initialize if local border is global border
  for (auto j : {u_d_1.obegin(1), u_d_1.oend(1) - 1})
    for (auto i{u_d_1.obegin(0)}, i_end{u_d_1.oend(0)}; i < i_end; ++i) {
      if (u_d_1.gindex(0, i) < 0 or u_d_1.gindex(1, j) < 0)
        u_d_1(i, j) = u_d_2(i, j) = 1;  // left boundary condition
      if (u_d_1.gindex(0, i) >= n_x or u_d_1.gindex(1, j) >= n_y)
        u_d_1(i, j) = u_d_2(i, j) = 0;  // right boundary condition
    }
  for (auto i : {u_d_1.obegin(0), u_d_1.oend(0) - 1})
    for (auto j{u_d_1.obegin(1)}, j_end{u_d_1.oend(1)}; j < j_end; ++j) {
      if (u_d_1.gindex(0, i) < 0 or u_d_1.gindex(1, j) < 0)
        u_d_1(i, j) = u_d_2(i, j) = 1;  // lower boundary condition
      if (u_d_1.gindex(0, i) >= n_x or u_d_1.gindex(1, j) >= n_y)
        u_d_1(i, j) = u_d_2(i, j) = 0;  // upper boundary condition
    }
  const double dx_2{dx * dx}, dy_2{dy * dy};
  // loop until converged
  bool converged{false};
  int iterations{0};
  while (not converged) {
    iterations++;
    // exchange asynchronously overlapping boundary data
    mpl::irequest_pool r(update_overlap(comm_c, u_d_1));
    // apply one Jacobi iteration step for interior region
    double delta_u{0}, sum_u{0};
    for (auto j{u_d_1.begin(1) + 1}, j_end{u_d_1.end(1) - 1}; j < j_end; ++j)
      for (auto i{u_d_1.begin(0) + 1}, i_end{u_d_1.end(0) - 1}; i < i_end; ++i) {
        u_d_2(i, j) = (dy_2 * (u_d_1(i - 1, j) + u_d_1(i + 1, j)) +
                       dx_2 * (u_d_1(i, j - 1) + u_d_1(i, j + 1))) /
                      (2 * (dx_2 + dy_2));
        delta_u += std::abs(u_d_2(i, j) - u_d_1(i, j));
        sum_u += std::abs(u_d_2(i, j));
      }
    r.waitall();
    // apply one Jacobi iteration step for edge region, which requires overlapping boundary data
    for (auto j : {u_d_1.begin(1), u_d_1.end(1) - 1})
      for (auto i{u_d_1.begin(0)}, i_end{u_d_1.end(0)}; i < i_end; ++i) {
        u_d_2(i, j) = (dy_2 * (u_d_1(i - 1, j) + u_d_1(i + 1, j)) +
                       dx_2 * (u_d_1(i, j - 1) + u_d_1(i, j + 1))) /
                      (2 * (dx_2 + dy_2));
        delta_u += std::abs(u_d_2(i, j) - u_d_1(i, j));
        sum_u += std::abs(u_d_2(i, j));
      }
    for (auto j{u_d_1.begin(1)}, j_end{u_d_1.end(1)}; j < j_end; ++j)
      for (auto i : {u_d_1.begin(0), u_d_1.end(0) - 1}) {
        u_d_2(i, j) = (dy_2 * (u_d_1(i - 1, j) + u_d_1(i + 1, j)) +
                       dx_2 * (u_d_1(i, j - 1) + u_d_1(i, j + 1))) /
                      (2 * (dx_2 + dy_2));
        delta_u += std::abs(u_d_2(i, j) - u_d_1(i, j));
        sum_u += std::abs(u_d_2(i, j));
      }
    // determine global sum of delta_u and sum_u and distribute to all processors
    double_2 delta_sum_u{delta_u, sum_u};  // pack into pair
    comm_c.allreduce(
        [](double_2 a, double_2 b) {
          // reduction adds component-by-component
          return double_2{std::get<0>(a) + std::get<0>(b), std::get<1>(a) + std::get<1>(b)};
        },
        delta_sum_u);
    std::tie(delta_u, sum_u) = delta_sum_u;  // unpack from pair
    converged = delta_u / sum_u < 1e-3;      // check for convergence
    u_d_2.swap(u_d_1);
  }
  // gather data and print result
  if (comm_c.rank() == 0) {
    std::cerr << iterations << " iterations\n";
    // local grid to store the whole set of inner grid points
    mpl::local_grid<2, double> u{comm_c, {n_x, n_y}};
    gather(comm_c, 0, u_d_1, u);
    for (auto j{u.begin(1)}, j_end{u.end(1)}; j < j_end; ++j) {
      for (auto i{u.begin(0)}, i_end{u.end(0)}; i < i_end; ++i)
        std::cout << u(i, j) << '\t';
      std::cout << '\n';
    }
  } else
    gather(comm_c, 0, u_d_1);
  return EXIT_SUCCESS;
}
