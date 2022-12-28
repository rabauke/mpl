#include <cstdlib>
#include <iostream>
// include MPL header file
#include <mpl/mpl.hpp>

int main(int argc, char *argv[]) {
  using namespace std::string_literals;

  // get a reference to communicator "world"
  [[maybe_unused]] const mpl::communicator &comm_world{mpl::environment::comm_world()};
  // get the parent inter-communicator
  auto &inter_comm{mpl::inter_communicator::parent()};
  std::cout << "Hello world! I am running on \"" << mpl::environment::processor_name()
            << "\". My rank is " << inter_comm.rank() << " out of " << inter_comm.size()
            << " processes.\n";
  std::cout << "commandline arguments: ";
  for (int i{0}; i < argc; ++i)
    std::cout << argv[i] << ' ';
  std::cout << std::endl;
  double message;
  inter_comm.bcast(0, message);
  std::cout << "got: " << message << '\n';
  return EXIT_SUCCESS;
}
