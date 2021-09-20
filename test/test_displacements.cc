#define BOOST_TEST_MODULE displacements

#include <boost/test/included/unit_test.hpp>
#include <mpl/mpl.hpp>

BOOST_AUTO_TEST_CASE(displacements) {
  mpl::displacements displacements_1(10);
  mpl::displacements displacements_2{1, 2, 3};
  displacements_2.push_back(10);
  BOOST_TEST(displacements_1.size() == 10);
  BOOST_TEST(displacements_2.size() == 4);
}
