#define BOOST_TEST_MODULE communicator_gatherv

#include <boost/test/included/unit_test.hpp>
#include <mpl/mpl.hpp>
#include <tuple>
#include "test_helper.hpp"


template<typename T>
bool gatherv_test(const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  const int N{(comm_world.size() * comm_world.size() + comm_world.size()) / 2};
  std::vector<T> v_gather(N);
  std::vector<T> v_send(comm_world.rank() + 1);
  std::vector<T> v_expected(N);
  std::iota(begin(v_expected), end(v_expected), val);
  mpl::layouts<T> layouts;
  for (int i{0}, i_end{comm_world.size()}, offset{0}; i < i_end; ++i) {
    layouts.push_back(mpl::indexed_layout<T>({{i + 1, offset}}));
    offset += i + 1;
  }
  T t_val{val};
  for (int i{0}, i_end{(comm_world.rank() * comm_world.rank() + comm_world.rank()) / 2};
       i < i_end; ++i)
    ++t_val;
  std::iota(begin(v_send), end(v_send), t_val);
  mpl::vector_layout<T> layout(comm_world.rank() + 1);
  if (comm_world.rank() == 0) {
    comm_world.gatherv(0, v_send.data(), layout, v_gather.data(), layouts);
    return v_gather == v_expected;
  } else {
    comm_world.gatherv(0, v_send.data(), layout);
    return true;
  }
}


template<typename T>
bool igatherv_test(const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  const int N{(comm_world.size() * comm_world.size() + comm_world.size()) / 2};
  std::vector<T> v_gather(N);
  std::vector<T> v_send(comm_world.rank() + 1);
  std::vector<T> v_expected(N);
  std::iota(begin(v_expected), end(v_expected), val);
  mpl::layouts<T> layouts;
  for (int i{0}, i_end{comm_world.size()}, offset{0}; i < i_end; ++i) {
    layouts.push_back(mpl::indexed_layout<T>({{i + 1, offset}}));
    offset += i + 1;
  }
  T t_val{val};
  for (int i{0}, i_end{(comm_world.rank() * comm_world.rank() + comm_world.rank()) / 2};
       i < i_end; ++i)
    ++t_val;
  std::iota(begin(v_send), end(v_send), t_val);
  mpl::vector_layout<T> layout(comm_world.rank() + 1);
  if (comm_world.rank() == 0) {
    auto r{comm_world.igatherv(0, v_send.data(), layout, v_gather.data(), layouts)};
    r.wait();
    return v_gather == v_expected;
  } else {
    auto r{comm_world.igatherv(0, v_send.data(), layout)};
    r.wait();
    return true;
  }
}


BOOST_AUTO_TEST_CASE(gatherv) {
  BOOST_TEST(gatherv_test(1.0));
  BOOST_TEST(gatherv_test(tuple{1, 2.0}));

#if !(defined MPICH)
  BOOST_TEST(igatherv_test(1.0));
  BOOST_TEST(igatherv_test(tuple{1, 2.0}));
#endif
}
