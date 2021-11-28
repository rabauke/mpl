#define BOOST_TEST_MODULE inter_communicator

#include <boost/test/included/unit_test.hpp>
#include <mpl/mpl.hpp>

// test inter-communicator creation
BOOST_AUTO_TEST_CASE(inter_communicator_create) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  // split communicator comm_world into two groups consisting of processes with odd and even
  // rank in comm_world
  const int world_rank{comm_world.rank()};
  const int world_size{comm_world.size()};
  const int my_group{world_rank % 2};
  mpl::communicator local_communicator{mpl::communicator::split, comm_world, my_group};
  const int local_leader{0};
  const int remote_leader{my_group == 0 ? 1 : 0};
  // comm_world is used as the communicator that can communicate with processes in the local
  // group as well as in the remote group
  mpl::inter_communicator inter_com{local_communicator, local_leader, comm_world, remote_leader};
  BOOST_TEST((inter_com.size() + inter_com.remote_size() == world_size));
  if (my_group == 0) {
    BOOST_TEST((inter_com.size() == (world_size + 1) / 2));
    BOOST_TEST((inter_com.remote_size() == world_size / 2));
  } else {
    BOOST_TEST((inter_com.remote_size() == (world_size + 1) / 2));
    BOOST_TEST((inter_com.size() == world_size / 2));
  }
}


// test inter-communicator merge
BOOST_AUTO_TEST_CASE(inter_communicator_merge) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  // split communicator comm_world into two groups consisting of processes with odd and even
  // rank in comm_world
  const int world_rank{comm_world.rank()};
  const int my_group{world_rank % 2};
  mpl::communicator local_communicator{mpl::communicator::split, comm_world, my_group};
  const int local_leader{0};
  const int remote_leader{my_group == 0 ? 1 : 0};
  // comm_world is used as the communicator that can communicate with processes in the local
  // group as well as in the remote group
  mpl::inter_communicator inter_comm{local_communicator, local_leader, comm_world, remote_leader};
  mpl::communicator com{inter_comm, mpl::communicator::order_low};
  const auto communicator_equality{com.compare(comm_world)};
  BOOST_TEST(
      (communicator_equality == mpl::communicator::congruent or
              communicator_equality == mpl::communicator::similar));
}
