#define BOOST_TEST_MODULE communicator_gather

#include <boost/test/included/unit_test.hpp>
#include <mpl/mpl.hpp>
#include <iterator>


template<typename T>
bool gather_test(const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  if (comm_world.rank() == 0) {
    std::vector<T> v(comm_world.size());
    comm_world.gather(0, val, v.data());
    std::vector<T> v_expected;
    for (int i{0}; i < comm_world.size(); ++i)
      v_expected.push_back(val);
    return v == v_expected;
  } else {
    comm_world.gather(0, val);
    return true;
  }
}


template<typename T>
bool gather_test(const std::vector<T> &send, const std::vector<T> &expected,
                 const mpl::layout<T> &l) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  if (comm_world.rank() == 0) {
    std::vector<T> v(comm_world.size() * send.size());
    comm_world.gather(0, send.data(), l, v.data(), l);
    std::vector<T> v_expected;
    for (int i{0}; i < comm_world.size(); ++i)
      std::copy(expected.begin(), expected.end(), std::back_inserter(v_expected));
    return v == v_expected;
  } else {
    comm_world.gather(0, send.data(), l);
    return true;
  }
}


template<typename T>
bool igather_test(const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  if (comm_world.rank() == 0) {
    std::vector<T> v(comm_world.size());
    auto r{comm_world.igather(0, val, v.data())};
    std::vector<T> v_expected;
    for (int i{0}; i < comm_world.size(); ++i)
      v_expected.push_back(val);
    r.wait();
    return v == v_expected;
  } else {
    auto r{comm_world.igather(0, val)};
    r.wait();
    return true;
  }
}


template<typename T>
bool igather_test(const std::vector<T> &send, const std::vector<T> &expected,
                  const mpl::layout<T> &l) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  if (comm_world.rank() == 0) {
    std::vector<T> v(comm_world.size() * send.size());
    auto r{comm_world.igather(0, send.data(), l, v.data(), l)};
    std::vector<T> v_expected;
    for (int i{0}; i < comm_world.size(); ++i)
      std::copy(expected.begin(), expected.end(), std::back_inserter(v_expected));
    r.wait();
    return v == v_expected;
  } else {
    auto r{comm_world.igather(0, send.data(), l)};
    r.wait();
    return true;
  }
}


BOOST_AUTO_TEST_CASE(gather) {
  BOOST_TEST(gather_test(1.0));
  BOOST_TEST(gather_test(std::array{1, 2, 3, 4}));
  {
    const std::vector send{1, 2, 3, 4, 5, 6};
    const std::vector expected{0, 2, 3, 0, 5, 0};
    mpl::indexed_layout<int> l{{{2, 1}, {1, 4}}};
    l.resize(0, 6);
    BOOST_TEST(gather_test(send, expected, l));
  }

  BOOST_TEST(igather_test(1.0));
  BOOST_TEST(igather_test(std::array{1, 2, 3, 4}));
  {
    const std::vector send{1, 2, 3, 4, 5, 6};
    const std::vector expected{0, 2, 3, 0, 5, 0};
    mpl::indexed_layout<int> l{{{2, 1}, {1, 4}}};
    l.resize(0, 6);
    BOOST_TEST(igather_test(send, expected, l));
  }
}
