#define BOOST_TEST_MODULE communicator

#include <boost/test/included/unit_test.hpp>
#include <mpl/mpl.hpp>

// test manual external initialization of MPI environment
bool initialization_test() {
  MPI_Init(nullptr, nullptr);
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  return comm_world.is_valid();
}


BOOST_AUTO_TEST_CASE(initialization) {
  BOOST_TEST(initialization_test());
}
