#include <cstdlib>
#include <iostream>
// include MPL header file
#include <mpl/mpl.hpp>

int main() {
  // get a reference to communicator "world"
  const mpl::communicator &comm_world{mpl::environment::comm_world()};

  mpl::file file;
  file.open(comm_world, "test", mpl::file::openmode::create | mpl::file::openmode::read_write);
  file.preallocate(1024);
  auto size{file.size()};
  std::cout << size << '\n';
  if (comm_world.rank() == 0) {
    for (int i{0}; i < 1024; ++i)
      file.write(i);
  }
  file.close();
  return EXIT_SUCCESS;
}
