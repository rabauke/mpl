#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <mpl/mpl.hpp>

int main() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  // run the program with two or more processes
  if (comm_world.size() < 2)
    return EXIT_FAILURE;
  // test layout for a subarray
  // layouts on sending and receiving side may differ but must be compatible
  const int n0 = 20, n1 = 8;  // size of two-dimensional array
  const int s0 = 11, s1 = 3;  // size of two-dimensional subarray
  // process 0 sends
  if (comm_world.rank() == 0) {
    // C order matrix with two-dimensional C arrays
    double A[n1][n0];
    for (int i1 = 0; i1 < n1; ++i1)
      for (int i0 = 0; i0 < n0; ++i0)
        A[i1][i0] = i0 + 0.01 * i1;
    mpl::subarray_layout<double> subarray({
        {n1, s1, 2},  // 2nd dimension: size of array, size of subarray, start of subarray
        {n0, s0, 4}   // 1st dimension: size of array, size of subarray, start of subarray
    });
    comm_world.send(&A[0][0], subarray, 1);
  }
  // process 1 receives
  if (comm_world.rank() == 1) {
    double A[s1][s0];
    mpl::contiguous_layout<double> array(s0 * s1);
    comm_world.recv(&A[0][0], array, 0);
    for (int i1 = 0; i1 < s1; ++i1) {
      for (int i0 = 0; i0 < s0; ++i0)
        std::cout << std::fixed << std::setprecision(2) << A[i1][i0] << "  ";
      std::cout << '\n';
    }
  }
  return EXIT_SUCCESS;
}
