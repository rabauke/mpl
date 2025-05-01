#define BOOST_TEST_MODULE communicator_gatherv

#include <boost/test/included/unit_test.hpp>
#include <mpl/mpl.hpp>
#include <tuple>
#include "test_helper.hpp"


template<use_non_root_overload variant, typename T>
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
  const mpl::vector_layout<T> layout(comm_world.rank() + 1);
  if constexpr (variant == use_non_root_overload::yes) {
    if (comm_world.rank() == 0)
      comm_world.gatherv(0, v_send.data(), layout, v_gather.data(), layouts);
    else
      comm_world.gatherv(0, v_send.data(), layout);
  } else {
    comm_world.gatherv(0, v_send.data(), layout, v_gather.data(), layouts);
  }
  return comm_world.rank() == 0 ? v_gather == v_expected : true;
}


template<use_non_root_overload variant, typename T>
bool gatherv_contiguous_test(const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  const int N{(comm_world.size() * comm_world.size() + comm_world.size()) / 2};
  std::vector<T> v_gather(N);
  std::vector<T> v_send(comm_world.rank() + 1);
  std::vector<T> v_expected(N);
  std::iota(begin(v_expected), end(v_expected), val);
  mpl::contiguous_layouts<T> layouts;
  mpl::displacements displacements;
  for (int i{0}, i_end{comm_world.size()}, offset{0}; i < i_end; ++i) {
    layouts.push_back(mpl::contiguous_layout<T>(i + 1));
    displacements.push_back(sizeof(T) * offset);
    offset += i + 1;
  }
  T t_val{val};
  for (int i{0}, i_end{(comm_world.rank() * comm_world.rank() + comm_world.rank()) / 2};
       i < i_end; ++i)
    ++t_val;
  std::iota(begin(v_send), end(v_send), t_val);
  const mpl::contiguous_layout<T> layout(comm_world.rank() + 1);
  if constexpr (variant == use_non_root_overload::yes) {
    if (comm_world.rank() == 0)
      comm_world.gatherv(0, v_send.data(), layout, v_gather.data(), layouts, displacements);
    else
      comm_world.gatherv(0, v_send.data(), layout);
  } else {
    comm_world.gatherv(0, v_send.data(), layout, v_gather.data(), layouts, displacements);
  }
  return comm_world.rank() == 0 ? v_gather == v_expected : true;
}


template<use_non_root_overload variant, typename T>
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
  const mpl::vector_layout<T> layout(comm_world.rank() + 1);
  if constexpr (variant == use_non_root_overload::yes) {
    if (comm_world.rank() == 0) {
      auto r{comm_world.igatherv(0, v_send.data(), layout, v_gather.data(), layouts)};
      r.wait();
    } else {
      auto r{comm_world.igatherv(0, v_send.data(), layout)};
      r.wait();
    }
  } else {
    auto r{comm_world.igatherv(0, v_send.data(), layout, v_gather.data(), layouts)};
    r.wait();
  }
  return comm_world.rank() == 0 ? v_gather == v_expected : true;
}


template<use_non_root_overload variant, typename T>
bool igatherv_contiguous_test(const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  const int N{(comm_world.size() * comm_world.size() + comm_world.size()) / 2};
  std::vector<T> v_gather(N);
  std::vector<T> v_send(comm_world.rank() + 1);
  std::vector<T> v_expected(N);
  std::iota(begin(v_expected), end(v_expected), val);
  mpl::contiguous_layouts<T> layouts;
  mpl::displacements displacements;
  for (int i{0}, i_end{comm_world.size()}, offset{0}; i < i_end; ++i) {
    layouts.push_back(mpl::contiguous_layout<T>(i + 1));
    displacements.push_back(sizeof(T) * offset);
    offset += i + 1;
  }
  T t_val{val};
  for (int i{0}, i_end{(comm_world.rank() * comm_world.rank() + comm_world.rank()) / 2};
       i < i_end; ++i)
    ++t_val;
  std::iota(begin(v_send), end(v_send), t_val);
  const mpl::contiguous_layout<T> layout(comm_world.rank() + 1);
  if constexpr (variant == use_non_root_overload::yes) {
    if (comm_world.rank() == 0) {
      auto r{comm_world.igatherv(0, v_send.data(), layout, v_gather.data(), layouts,
                                 displacements)};
      r.wait();
    } else {
      auto r{comm_world.igatherv(0, v_send.data(), layout)};
      r.wait();
    }
  } else {
    auto r{
        comm_world.igatherv(0, v_send.data(), layout, v_gather.data(), layouts, displacements)};
    r.wait();
  }
  return comm_world.rank() == 0 ? v_gather == v_expected : true;
}


BOOST_AUTO_TEST_CASE(gatherv) {
  BOOST_TEST(gatherv_test<use_non_root_overload::no>(1.0));
  BOOST_TEST(gatherv_test<use_non_root_overload::no>(tuple{1, 2.0}));

  BOOST_TEST(gatherv_test<use_non_root_overload::yes>(1.0));
  BOOST_TEST(gatherv_test<use_non_root_overload::yes>(tuple{1, 2.0}));

  BOOST_TEST(gatherv_contiguous_test<use_non_root_overload::no>(1.0));
  BOOST_TEST(gatherv_contiguous_test<use_non_root_overload::no>(tuple{1, 2.0}));

  BOOST_TEST(gatherv_contiguous_test<use_non_root_overload::yes>(1.0));
  BOOST_TEST(gatherv_contiguous_test<use_non_root_overload::yes>(tuple{1, 2.0}));

  // skip tests for older versions of MPICH due to a bug in MPICH's implementation of Alltoallw
#if !defined MPICH || MPICH_NUMVERSION >= 40101000
  BOOST_TEST(igatherv_test<use_non_root_overload::no>(1.0));
  BOOST_TEST(igatherv_test<use_non_root_overload::no>(tuple{1, 2.0}));

  BOOST_TEST(igatherv_test<use_non_root_overload::yes>(1.0));
  BOOST_TEST(igatherv_test<use_non_root_overload::yes>(tuple{1, 2.0}));
#endif

  BOOST_TEST(igatherv_contiguous_test<use_non_root_overload::no>(1.0));
  BOOST_TEST(igatherv_contiguous_test<use_non_root_overload::no>(tuple{1, 2.0}));

  BOOST_TEST(igatherv_contiguous_test<use_non_root_overload::yes>(1.0));
  BOOST_TEST(igatherv_contiguous_test<use_non_root_overload::yes>(tuple{1, 2.0}));
}
