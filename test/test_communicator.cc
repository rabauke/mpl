#define BOOST_TEST_MODULE communicator

#include <boost/test/included/unit_test.hpp>
#include <mpl/mpl.hpp>

bool communicator_comm_world_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  if (not comm_world.is_valid())
    return false;
  int size{comm_world.size()};
  int rank{comm_world.rank()};
  if (rank < 0 or rank >= size)
    return false;
  if (size < 1)
    return false;
  const mpl::communicator &comm_word2 = mpl::environment::comm_world();
  if (comm_word2 != comm_world)
    return false;
  const mpl::communicator &comm_self = mpl::environment::comm_self();
  if (comm_self == comm_world)
    return false;
  if (comm_world.size() == 1 and comm_self.compare(comm_world) != mpl::communicator::congruent)
    return false;
  if (comm_world.size() > 1 and comm_self.compare(comm_world) != mpl::communicator::unequal)
    return false;
  if (size > 1) {
    mpl::communicator comm_new{mpl::communicator::split, comm_world, rank % 2 == 0};
    if (comm_world.size() % 2 == 0) {
      if (comm_new.size() != size / 2)
        return false;
    } else {
      if (rank % 2 == 0 && comm_new.size() != (size + 1) / 2)
        return false;
      if (rank % 2 == 1 && comm_new.size() != (size - 1) / 2)
        return false;
    }
  }
  return true;
}

bool communicator_comm_self_test() {
  const mpl::communicator &comm_self = mpl::environment::comm_self();
  if (not comm_self.is_valid())
    return false;
  int size{comm_self.size()};
  int rank{comm_self.rank()};
  if (rank != 0 or rank >= size)
    return false;
  if (size != 1)
    return false;
  const mpl::communicator &comm_self2 = mpl::environment::comm_self();
  if (comm_self2 != comm_self)
    return false;
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  if (comm_self == comm_world)
    return false;
  if (comm_world.size() == 1 and comm_self.compare(comm_world) != mpl::communicator::congruent)
    return false;
  if (comm_world.size() > 1 and comm_self.compare(comm_world) != mpl::communicator::unequal)
    return false;
  return true;
}

BOOST_AUTO_TEST_CASE(communicator) {
  BOOST_TEST(communicator_comm_world_test());
  BOOST_TEST(communicator_comm_self_test());
}
