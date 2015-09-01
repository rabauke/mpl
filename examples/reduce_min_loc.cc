#include <cstdlib>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <vector>
#include <mpl/mpl.hpp>

typedef std::pair<double, int> pair_t;

int main() {
  const mpl::communicator &comm_world=mpl::environment::comm_world();
  // generate data
  std::srand(std::time(0)*comm_world.rank());  // random seed
  const int n=8;
  std::vector<pair_t> v(n);
  for (pair_t &i : v)
    i=std::make_pair(static_cast<double>(std::rand())/RAND_MAX, comm_world.rank());
  // calculate minium and its location and send result to rank 0
  mpl::contiguous_layout<pair_t> layout(n);
  if (comm_world.rank()==0) {
    std::vector<pair_t> result(n);
    // calculate minimum
    comm_world.reduce(mpl::min_loc<double>(), 0, v.data(), result.data(), layout);
    // display data from all ranks
    std::cout << "Arguments:\n";
    for (int r=0; r<comm_world.size(); ++r) {
      if (r>0)
	comm_world.recv(v.data(), layout, r);
      for (pair_t i : v) 
	std::cout << std::fixed << std::setprecision(5) << i.first << ' ' << i.second << '\t';
      std::cout << '\n';
    }
    // display results of global reduction
    std::cout << "\nResults:\n";
    for (pair_t i : result) 
      std::cout << std::fixed << std::setprecision(5) << i.first << ' ' << i.second << '\t';
    std::cout << '\n';
  } else {
    // calculate minium and its location and send result to rank 0
    comm_world.reduce(mpl::min_loc<double>(), 0, v.data(), layout);
    // send data to rank 0 for display
    comm_world.send(v.data(), layout, 0);
  }
  return EXIT_SUCCESS;
}
