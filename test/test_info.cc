#define BOOST_TEST_MODULE info

#include <boost/test/included/unit_test.hpp>
#include <mpl/mpl.hpp>

BOOST_AUTO_TEST_CASE(info) {
  [[maybe_unused]] const mpl::communicator &comm_world{mpl::environment::comm_world()};

  mpl::info info_1;
  BOOST_TEST(info_1.size() == 0);
  info_1.set("Douglas Adams", "The Hitchhiker's Guide to the Galaxy");
  info_1.set("Isaac Asimov", "Nightfall");
  BOOST_TEST(info_1.size() == 2);
  BOOST_TEST(info_1.value("Isaac Asimov").value() == "Nightfall");

  mpl::info info_2{info_1};
  BOOST_TEST(info_1.size() == 2);
  BOOST_TEST(info_2.value("Isaac Asimov").value() == "Nightfall");
}
