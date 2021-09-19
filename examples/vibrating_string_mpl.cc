// solve the time-dependent one-dimensional wave equation
// via a finite difference discretization and explicit time stepping

#include <cstdlib>
#include <iostream>
#include <cmath>
#include <vector>
#include <mpl/mpl.hpp>

const int n{1001};        // total number of grid points
const double l{1};        // lengths of domain
const double c{1};        // speed of sound
const double dt{0.001};   // temporal step width
const double t_end{2.4};  // simulation time

enum class copy : int { left, right };

// update grid points
void string(const std::vector<double> &u, const std::vector<double> &u_old,
            std::vector<double> &u_new, double eps) {
  using size_type = std::vector<double>::size_type;
  const size_type N{u.size()};
  u_new[0] = u[0];
  for (size_type i{1}; i < N - 1; ++i)
    u_new[i] = eps * (u[i - 1] + u[i + 1]) + 2.0 * (1.0 - eps) * u[i] - u_old[i];
  u_new[N - 1] = u[N - 1];
}

// initial elongation of string
inline double u_0(double x) {
  if (x <= 0 or x >= l)
    return 0;
  return std::exp(-200.0 * (x - 0.5 * l) * (x - 0.5 * l));
}

// initial velocity of string
inline double u_0_dt(double x) {
  return 0.0;
}

int main() {
  const double dx{l / (n - 1)};  // grid spacing
  const double eps{dt * dt * c * c / (dx * dx)};
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  const int c_size{comm_world.size()};
  const int c_rank{comm_world.rank()};
  std::vector<int> n_l, n_0_l;
  for (int i{0}; i < c_size; ++i) {
    // number of local grid points of process i
    n_l.push_back((i + 1) * (n - 2) / c_size - i * (n - 2) / c_size + 2);
    // position of local grid of process i within the global grid
    n_0_l.push_back(i * (n - 2) / c_size);
  }
  // grid data for times (t-dt), t and t+dt
  std::vector<double> u_old_l(n_l[c_rank]);
  std::vector<double> u_l(n_l[c_rank]);
  std::vector<double> u_new_l(n_l[c_rank]);
  // 1st propagation step uses current elongation and velocity
  // calculate all grid points including overlapping border data
  for (int i{0}; i < n_l[c_rank]; ++i) {
    double x = (i + n_0_l[c_rank]) * dx;
    u_old_l[i] = u_0(x);
    u_l[i] = 0.5 * eps * (u_0(x - dx) + u_0(x + dx)) + (1.0 - eps) * u_0(x) + dt * u_0_dt(x);
  }
  // propagate
  double t{2 * dt};
  while (t <= t_end) {
    // make one time step to get elongation
    string(u_l, u_old_l, u_new_l, eps);
    // update border data
    mpl::irequest_pool r;
    r.push(comm_world.isend(u_new_l[n_l[c_rank] - 2],
                            c_rank + 1 < c_size ? c_rank + 1 : mpl::proc_null, copy::right));
    r.push(comm_world.isend(u_new_l[1], c_rank - 1 >= 0 ? c_rank - 1 : mpl::proc_null,
                            copy::left));
    r.push(comm_world.irecv(u_new_l[0], c_rank - 1 >= 0 ? c_rank - 1 : mpl::proc_null,
                            copy::right));
    r.push(comm_world.irecv(u_new_l[n_l[c_rank] - 1],
                            c_rank + 1 < c_size ? c_rank + 1 : mpl::proc_null, copy::left));
    r.waitall();
    std::swap(u_l, u_old_l);
    std::swap(u_new_l, u_l);
    t += dt;
  }
  // finally, gather all the data at rank 0 and print result
  mpl::layouts<double> layouts;
  for (int i{0}; i < c_size; ++i)
    layouts.push_back(mpl::indexed_layout<double>({{n_l[i] - 2, n_0_l[i] + 1}}));
  mpl::contiguous_layout<double> layout(n_l[c_rank] - 2);
  if (c_rank == 0) {
    std::vector<double> u(n, 0);
    comm_world.gatherv(0, u_l.data() + 1, layout, u.data(), layouts);
    for (int i{0}; i < n; ++i)
      std::cout << dx * i << '\t' << u[i] << '\n';
  } else
    comm_world.gatherv(0, u_l.data() + 1, layout);
  return EXIT_SUCCESS;
}
