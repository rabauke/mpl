#include <cstdlib>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <mpl/mpl.hpp>


int main() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  // run the program with two or more processes
  if (comm_world.size() < 2)
    comm_world.abort(EXIT_FAILURE);
  // send / receive a single array
  {
    const int N = 10;
    double arr[N];
    if (comm_world.rank() == 0) {
      std::iota(arr, arr + N, 1);
      comm_world.send(arr, 1);
    }
    if (comm_world.rank() == 1) {
      comm_world.recv(arr, 0);
      for (int j = 0; j < N; ++j)
        std::cout << "arr[" << j << "] = " << arr[j] << '\n';
    }
  }
  // send / receive a single two-dimensional array
  {
    const int N0 = 2, N1 = 3;
    double arr[N0][N1];
    if (comm_world.rank() == 0) {
      for (int j1 = 0; j1 < N1; ++j1)
        for (int j0 = 0; j0 < N0; ++j0)
          arr[j0][j1] = (j0 + 1) + 100 * (j1 + 1);
      comm_world.send(arr, 1);
    }
    if (comm_world.rank() == 1) {
      comm_world.recv(arr, 0);
      for (int j1 = 0; j1 < N1; ++j1) {
        for (int j0 = 0; j0 < N0; ++j0)
          std::cout << "arr[" << j0 << ", " << j1 << "] = " << arr[j0][j1] << '\n';
      }
    }
  }
  return EXIT_SUCCESS;
}
