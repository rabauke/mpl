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
  mpl::inter_communicator icom{local_communicator, local_leader, comm_world, remote_leader};
  BOOST_TEST((icom.size() + icom.remote_size() == world_size));
  if (my_group == 0) {
    BOOST_TEST((icom.size() == (world_size + 1) / 2));
    BOOST_TEST((icom.remote_size() == world_size / 2));
  } else {
    BOOST_TEST((icom.remote_size() == (world_size + 1) / 2));
    BOOST_TEST((icom.size() == world_size / 2));
  }
}
