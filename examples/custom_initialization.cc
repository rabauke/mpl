#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <mpl/mpl.hpp>


// a custom initializer
class my_initializer final {
private:
  explicit my_initializer(int *argc, char **argv[]) {
    // initialize MPI by calling MPI_Init or MPI_Init_thread
    MPI_Init(argc, argv);
  }

  ~my_initializer() {
    // finalize MPI
    MPI_Finalize();
  }

public:
  static void init(int *argc, char **argv[]) {
    // variable must be static
    static const my_initializer init{argc, argv};
  }
};


int main(int argc, char *argv[]) {
  // custom initialization of the MPI environment before any MPL call
  my_initializer::init(&argc, &argv);

  // do some MPL operations
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  std::cout << "Hello world! I am running on \"" << mpl::environment::processor_name()
            << "\". My rank is " << comm_world.rank() << " out of " << comm_world.size()
            << " processes.\n";
  if (comm_world.size() >= 2) {
    if (comm_world.rank() == 0) {
      const std::string message{"Hello world!"};
      comm_world.send(message, 1);
    } else if (comm_world.rank() == 1) {
      std::string message;
      comm_world.recv(message, 0);
      std::cout << "got: \"" << message << "\"\n";
    }
  }

  // exit the program and implicitly deinitialize MPL first and MPI afterward
  return EXIT_SUCCESS;
}
