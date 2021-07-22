#define BOOST_TEST_MODULE communicator_allgatherv

#include <boost/test/included/unit_test.hpp>
#include <mpl/mpl.hpp>
#include <algorithm>
#include <array>
#include <vector>


template<typename T>
bool allgatherv_test(const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  const int N{(comm_world.size() * comm_world.size() + comm_world.size()) / 2};
  std::vector<T> v1(N), v2(N);
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
bool iallgatherv_test(const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  const int N{(comm_world.size() * comm_world.size() + comm_world.size()) / 2};
  std::vector<T> v1(N), v2(N);
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


struct tuple {
  int a;
  double b;
  tuple &operator++() {
    ++a;
    ++b;
    return *this;
  }
};

bool operator==(const tuple&t1, const tuple &t2)
{
  return t1.a== t2.a and t1.b == t2.b;
}

MPL_REFLECTION(tuple, a, b)


BOOST_AUTO_TEST_CASE(allgatherv) {
  BOOST_TEST(allgatherv_test(1.0));
  BOOST_TEST(allgatherv_test(tuple{1, 2.0}));

  BOOST_TEST(iallgatherv_test(1.0));
  BOOST_TEST(iallgatherv_test(tuple{1, 2.0}));
}
