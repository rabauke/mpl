#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <mpl/mpl.hpp>

int main() {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  // run the program with two or more processes
  if (comm_world.size() < 2)
    return EXIT_FAILURE;
  // test layout for a subarray
  // layouts on sending and receiving side may differ but must be compatible
  const int n_0{20}, n_1{8};  // size of two-dimensional array
  const int s_0{11}, s_1{3};  // size of two-dimensional subarray
  // process 0 sends
  if (comm_world.rank() == 0) {
    // C order matrix with two-dimensional C arrays
    double a[n_1][n_0];
    for (int i_1{0}; i_1 < n_1; ++i_1)
      for (int i_0{0}; i_0 < n_0; ++i_0)
        a[i_1][i_0] = i_0 + 0.01 * i_1;
    mpl::subarray_layout<double> subarray{{
        {n_1, s_1, 2},  // 2nd dimension: size of array, size of subarray, start of subarray
        {n_0, s_0, 4}   // 1st dimension: size of array, size of subarray, start of subarray
    }};
    comm_world.send(&a[0][0], subarray, 1);
  }
  // process 1 receives
  if (comm_world.rank() == 1) {
    double a[s_1][s_0];
    mpl::contiguous_layout<double> array{s_0 * s_1};
    comm_world.recv(&a[0][0], array, 0);
    for (int i_1{0}; i_1 < s_1; ++i_1) {
      for (int i_0{0}; i_0 < s_0; ++i_0)
        std::cout << std::fixed << std::setprecision(2) << a[i_1][i_0] << "  ";
      std::cout << '\n';
    }
  }
  return EXIT_SUCCESS;
}
