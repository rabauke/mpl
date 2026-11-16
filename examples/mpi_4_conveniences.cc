#include <array>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <mpl/mpl.hpp>


int main(int argc, char *argv[]) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  const mpl::info hints{mpl::info::from_environment(argc, argv)};
  auto [fst, snd]{comm_world.idup_with_info(hints)};
  snd.wait();

  std::vector<int, mpl::mpi_allocator<int>> values(1);
  values[0] = fst.rank();

  const int scalar_in{values[0] + 1};
  int scalar_inout{1};
  mpl::reduce_local(mpl::plus<int>{}, scalar_in, scalar_inout);

  const std::array<int, 2> buffer_in{values[0], values[0] + 1};
  std::array<int, 2> buffer_inout{1, 1};
  const mpl::contiguous_layout<int> layout{buffer_in.size()};
  mpl::reduce_local(mpl::plus<int>{}, buffer_in.data(), buffer_inout.data(), layout);

  if (fst.rank() == 0) {
    std::cout << "MPI library: " << mpl::environment::library_version() << '\n';
    std::cout << "Local reduction: " << scalar_inout << ", (" << buffer_inout[0] << ", "
              << buffer_inout[1] << ")\n";
  }
  return EXIT_SUCCESS;
}
