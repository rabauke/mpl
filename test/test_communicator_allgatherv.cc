#define BOOST_TEST_MODULE communicator_allgatherv

#include <boost/test/included/unit_test.hpp>
#include <mpl/mpl.hpp>
#include <numeric>
#include <vector>
#include "test_helper.hpp"


template<typename T>
bool allgatherv_test(const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  const int N{(comm_world.size() * comm_world.size() + comm_world.size()) / 2};
  std::vector<T> v1(N);
  std::vector<T> v2(N);
  std::iota(begin(v1), end(v1), val);
  mpl::layouts<T> l;
  for (int i{0}, i_end{comm_world.size()}, offset{0}; i < i_end; ++i) {
    l.push_back(mpl::indexed_layout<T>({{i + 1, offset}}));
    offset += i + 1;
  }
  comm_world.allgatherv(v1.data(), l[comm_world.rank()], v2.data(), l);
  return v1 == v2;
}


template<typename T>
bool allgatherv_contiguous_test(const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  const int N{(comm_world.size() * comm_world.size() + comm_world.size()) / 2};
  std::vector<T> v1(N);
  std::vector<T> v2(N);
  std::iota(begin(v1), end(v1), val);
  mpl::contiguous_layouts<T> l;
  mpl::displacements displacements;
  for (int i{0}, i_end{comm_world.size()}, offset{0}; i < i_end; ++i) {
    l.push_back(mpl::contiguous_layout<T>(i + 1));
    displacements.push_back(sizeof(T) * offset);
    offset += i + 1;
  }
  const auto rank{comm_world.rank()};
  comm_world.allgatherv(v1.data() + displacements[rank] / sizeof(T), l[rank], v2.data(), l,
                        displacements);
  return v1 == v2;
}


template<typename T>
bool iallgatherv_test(const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  const int N{(comm_world.size() * comm_world.size() + comm_world.size()) / 2};
  std::vector<T> v1(N);
  std::vector<T> v2(N);
  std::iota(begin(v1), end(v1), val);
  mpl::layouts<T> l;
  for (int i{0}, i_end{comm_world.size()}, offset{0}; i < i_end; ++i) {
    l.push_back(mpl::indexed_layout<T>({{i + 1, offset}}));
    offset += i + 1;
  }
  auto r{comm_world.iallgatherv(v1.data(), l[comm_world.rank()], v2.data(), l)};
  r.wait();
  return v1 == v2;
}


template<typename T>
bool iallgatherv_contiguous_test(const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  const int N{(comm_world.size() * comm_world.size() + comm_world.size()) / 2};
  std::vector<T> v1(N);
  std::vector<T> v2(N);
  std::iota(begin(v1), end(v1), val);
  mpl::contiguous_layouts<T> l;
  mpl::displacements displacements;
  for (int i{0}, i_end{comm_world.size()}, offset{0}; i < i_end; ++i) {
    l.push_back(mpl::contiguous_layout<T>(i + 1));
    displacements.push_back(sizeof(T) * offset);
    offset += i + 1;
  }
  const auto rank{comm_world.rank()};
  auto r{comm_world.iallgatherv(v1.data() + displacements[rank] / sizeof(T), l[rank], v2.data(),
                                l, displacements)};
  r.wait();
  return v1 == v2;
}


BOOST_AUTO_TEST_CASE(allgatherv) {
  BOOST_TEST(allgatherv_test(1.0));
  BOOST_TEST(allgatherv_test(tuple{1, 2.0}));

  BOOST_TEST(allgatherv_contiguous_test(1.0));
  BOOST_TEST(allgatherv_contiguous_test(tuple{1, 2.0}));

  BOOST_TEST(iallgatherv_test(1.0));
  BOOST_TEST(iallgatherv_test(tuple{1, 2.0}));

  BOOST_TEST(allgatherv_contiguous_test(1.0));
  BOOST_TEST(allgatherv_contiguous_test(tuple{1, 2.0}));
}
