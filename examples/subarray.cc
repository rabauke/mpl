#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <mpl/mpl.hpp>

int main() {
  const mpl::communicator &comm_world=mpl::environment::comm_world();
  const int n0=20, n1=8; 
  const int s0=11, s1=3; 
  // C order matrix with two-dimensional C arrays 
  if (comm_world.rank()==0) {
    double A[n1][n0];
    for (int i1=0; i1<n1; ++i1)
      for (int i0=0; i0<n0; ++i0)
  	A[i1][i0]=i0+0.01*i1;
    mpl::vector<int, 2> array_of_sizes(n1, n0),
      array_of_subsizes(s1, s0),
      array_of_starts(2, 4);
    mpl::subarray_layout<double, 2> subarray(array_of_sizes, 
					     array_of_subsizes, 
					     array_of_starts, 
					     mpl::subarray_layout<double, 2>::C_order);
    comm_world.send(&A[0][0], subarray, 1, 0);
  }
  if (comm_world.rank()==1) {
    double A[s1][s0];
    mpl::contiguous_layout<double> array(s0*s1);
    comm_world.recv(&A[0][0], array, 0, 0);
    for (int i1=0; i1<s1; ++i1) {
      for (int i0=0; i0<s0; ++i0)
	std::cout << std::fixed << std::setprecision(2) << A[i1][i0] << "  ";
      std::cout << '\n';
    }
  }
  return EXIT_SUCCESS;
}
