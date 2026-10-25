#include <catch2/catch_test_macros.hpp>
#include <mpl/mpl.hpp>


namespace {

class my_mpi_environment {
public:
  my_mpi_environment(int *argc, char ***argv) {
    MPI_Init(argc, argv);
  }

  ~my_mpi_environment() {
    MPI_Finalize();
  }
};


// test manual external initialization and finalization of the MPI environment
bool initialization_test() {
  // Create a static my_mpi_environment object on block scope before any call to MPL.  The
  // object will initialize the MPI environment in its constructor.  The object's destructor
  // finalizes the MPI environment.  The object will be destroyed after all MPL singletons have
  // been freed because static objects on block scope are destroyed in reverse order compared to
  // creation order.
  const static my_mpi_environment environment{nullptr, nullptr};

  // Perform some MPI operations.
  int size{0};
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size <= 0)
    return false;
  // Do some MPL stuff.
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  return comm_world.is_valid();
}

}  // namespace


SCENARIO("external initialization of the MPI environment", "[initialization]") {
  GIVEN("an external MPI initialization before any MPL calls") {
    WHEN("performing MPI and MPL operations") {
      THEN("communicator is valid and operations succeed") {
        REQUIRE(initialization_test());
      }
    }
  }
}
