#include <cstdlib>
#include <iostream>
// include MPL header file
#include <mpl/mpl.hpp>

int main() {
  // get a reference to communicator "world"
  const mpl::communicator &comm_world{mpl::environment::comm_world()};

  mpl::file file;
  try {
    file.open(comm_world, "test.bin",
              mpl::file::access_mode::create | mpl::file::access_mode::read_write);
    file.preallocate(1024);
    mpl::strided_vector_layout<std::uint8_t> l(256, 1, 2);
    mpl::vector_layout<std::uint8_t> l_v(256);
    file.set_view(3, l, "native");
    if (comm_world.rank() == 0) {
      std::vector<std::uint8_t> v;
      for (int i{0}; i < 256; ++i)
        v.push_back(static_cast<std::uint8_t>(i));
      file.write_at(8, v.data(), l_v);
    }
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
  }
  return EXIT_SUCCESS;
}
