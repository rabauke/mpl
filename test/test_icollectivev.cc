#define BOOST_TEST_MODULE icollectivev

#include <boost/test/included/unit_test.hpp>
#include <mpl/mpl.hpp>

template<typename T>
bool ialltoallv_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  std::vector<T> v_send,
      v_recv((comm_world.size() * comm_world.size() + comm_world.size()) / 2), v_expected;
  mpl::layouts<T> l_send, l_recv;
  for (int i = 0, i_end = comm_world.size(); i < i_end; ++i)
    for (int j = 0, j_end = comm_world.rank() + 1; j < j_end; ++j)
      v_send.push_back(comm_world.rank() + 1 + i);
  for (int i = 0, i_end = comm_world.size(); i < i_end; ++i) {
    l_send.push_back(
        mpl::indexed_layout<T>({{comm_world.rank() + 1, (comm_world.rank() + 1) * i}}));
    l_recv.push_back(mpl::indexed_layout<T>({{i + 1, (i * i + i) / 2}}));
  }
  auto r{comm_world.ialltoallv(v_send.data(), l_send, v_recv.data(), l_recv)};
  r.wait();
  for (int i = 0, i_end = comm_world.size(); i < i_end; ++i)
    for (int j = 0, j_end = i + 1; j < j_end; ++j)
      v_expected.push_back(i + 1 + comm_world.rank());
  return v_recv == v_expected;
}


BOOST_AUTO_TEST_CASE(icollectivev) {
  BOOST_TEST(ialltoallv_test<double>());
}
