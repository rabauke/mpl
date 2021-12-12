#define BOOST_TEST_MODULE group

#include <boost/test/included/unit_test.hpp>
#include <mpl/mpl.hpp>


BOOST_AUTO_TEST_CASE(group) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  const mpl::communicator &comm_self{mpl::environment::comm_self()};

  mpl::group group_world{comm_world};
  mpl::group group_self{comm_self};

  BOOST_TEST((group_world.size() == comm_world.size()));
  BOOST_TEST((group_world.rank() == comm_world.rank()));
  BOOST_TEST((group_self.size() == comm_self.size()));

  mpl::group group_world_copy{group_world};
  BOOST_TEST((group_world == group_world_copy));

  if (comm_world.size() > 1)
    BOOST_TEST((group_world != group_self));
  else
    BOOST_TEST((group_world == group_self));
  if (comm_world.size() > 1)
    BOOST_TEST((group_world.compare(group_self) == mpl::group::unequal));
  else
    BOOST_TEST((group_world.compare(group_self) == mpl::group::identical));

  BOOST_TEST((group_self.translate(0, group_world) == group_world.rank()));

  mpl::group group_union(mpl::group::Union, group_world, group_self);
  mpl::group group_intersection(mpl::group::intersection, group_world, group_self);
  mpl::group group_difference(mpl::group::difference, group_world, group_self);
  mpl::group group_with_0(mpl::group::include, group_world, {0});
  mpl::group group_without_0(mpl::group::exclude, group_world, {0});

  BOOST_TEST((group_union.size() == group_world.size()));
  BOOST_TEST((group_intersection.size() == 1));
  BOOST_TEST((group_difference.size() == group_world.size() - 1));
  BOOST_TEST((group_with_0.size() == 1));
  BOOST_TEST((group_without_0.size() == group_world.size() - 1));
}
