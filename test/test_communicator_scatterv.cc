#define BOOST_TEST_MODULE communicator_scatterv

#include <boost/test/included/unit_test.hpp>
#include <mpl/mpl.hpp>
#include <tuple>
#include "test_helper.hpp"


template<use_non_root_overload variant, typename T>
bool scatterv_test(const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  const int N{(comm_world.size() * comm_world.size() + comm_world.size()) / 2};
  std::vector<T> v_scatter(N);
  std::vector<T> v_recv(comm_world.rank() + 1);
  std::vector<T> v_expected(comm_world.rank() + 1);
  std::iota(begin(v_scatter), end(v_scatter), val);
  mpl::layouts<T> layouts;
  for (int i{0}, i_end{comm_world.size()}, offset{0}; i < i_end; ++i) {
    layouts.push_back(mpl::indexed_layout<T>({{i + 1, offset}}));
    offset += i + 1;
  }
  T t_val{val};
  for (int i{0}, i_end{(comm_world.rank() * comm_world.rank() + comm_world.rank()) / 2};
       i < i_end; ++i)
    ++t_val;
  std::iota(begin(v_expected), end(v_expected), t_val);
  mpl::vector_layout<T> layout(comm_world.rank() + 1);
  if constexpr (variant == use_non_root_overload::yes) {
    if (comm_world.rank() == 0)
      comm_world.scatterv(0, v_scatter.data(), layouts, v_recv.data(), layout);
    else
      comm_world.scatterv(0, v_recv.data(), layout);
  } else
    comm_world.scatterv(0, v_scatter.data(), layouts, v_recv.data(), layout);
  return v_recv == v_expected;
}


template<use_non_root_overload variant, typename T>
bool iscatterv_test(const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  const int N{(comm_world.size() * comm_world.size() + comm_world.size()) / 2};
  std::vector<T> v_scatter(N);
  std::vector<T> v_recv(comm_world.rank() + 1);
  std::vector<T> v_expected(comm_world.rank() + 1);
  std::iota(begin(v_scatter), end(v_scatter), val);
  mpl::layouts<T> layouts;
  for (int i{0}, i_end{comm_world.size()}, offset{0}; i < i_end; ++i) {
    layouts.push_back(mpl::indexed_layout<T>({{i + 1, offset}}));
    offset += i + 1;
  }
  T t_val{val};
  for (int i{0}, i_end{(comm_world.rank() * comm_world.rank() + comm_world.rank()) / 2};
       i < i_end; ++i)
    ++t_val;
  std::iota(begin(v_expected), end(v_expected), t_val);
  mpl::vector_layout<T> layout(comm_world.rank() + 1);
  if constexpr (variant == use_non_root_overload::yes) {
    if (comm_world.rank() == 0) {
      auto r{comm_world.iscatterv(0, v_scatter.data(), layouts, v_recv.data(), layout)};
      r.wait();
    } else {
      auto r{comm_world.iscatterv(0, v_recv.data(), layout)};
      r.wait();
    }
  } else {
    auto r{comm_world.iscatterv(0, v_scatter.data(), layouts, v_recv.data(), layout)};
    r.wait();
  }
  return v_recv == v_expected;
}


BOOST_AUTO_TEST_CASE(scatterv) {
  BOOST_TEST(scatterv_test<use_non_root_overload::no>(1.0));
  BOOST_TEST(scatterv_test<use_non_root_overload::no>(tuple{1, 2.0}));

  BOOST_TEST(scatterv_test<use_non_root_overload::yes>(1.0));
  BOOST_TEST(scatterv_test<use_non_root_overload::yes>(tuple{1, 2.0}));

#if !defined MPICH || MPICH_NUMVERSION >= 40101000
  BOOST_TEST(iscatterv_test<use_non_root_overload::no>(1.0));
  BOOST_TEST(iscatterv_test<use_non_root_overload::no>(tuple{1, 2.0}));

  BOOST_TEST(iscatterv_test<use_non_root_overload::yes>(1.0));
  BOOST_TEST(iscatterv_test<use_non_root_overload::yes>(tuple{1, 2.0}));
#endif
}
