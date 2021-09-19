#include <cstdlib>
#include <iostream>
#include <numeric>
#include <mpl/mpl.hpp>


int main() {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  // run the program with two or more processes
  if (comm_world.size() < 2)
    comm_world.abort(EXIT_FAILURE);
  // send / receive a single array
  {
    const int n{10};
    double arr[n];
    if (comm_world.rank() == 0) {
      std::iota(arr, arr + n, 1);
      comm_world.send(arr, 1);
    }
    if (comm_world.rank() == 1) {
      comm_world.recv(arr, 0);
      for (int j{0}; j < n; ++j)
        std::cout << "arr[" << j << "] = " << arr[j] << '\n';
    }
  }
  // send / receive a single two-dimensional array
  {
    const int n_0{2}, n_1{3};
    double arr[n_0][n_1];
    if (comm_world.rank() == 0) {
      for (int j_1{0}; j_1 < n_1; ++j_1)
        for (int j_0{0}; j_0 < n_0; ++j_0)
          arr[j_0][j_1] = (j_0 + 1) + 100 * (j_1 + 1);
      comm_world.send(arr, 1);
    }
    if (comm_world.rank() == 1) {
      comm_world.recv(arr, 0);
      for (int j_1{0}; j_1 < n_1; ++j_1) {
        for (int j_0{0}; j_0 < n_0; ++j_0)
          std::cout << "arr[" << j_0 << ", " << j_1 << "] = " << arr[j_0][j_1] << '\n';
      }
    }
  }
  return EXIT_SUCCESS;
}
