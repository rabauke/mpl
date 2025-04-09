#define BOOST_TEST_MODULE communicator_mprobe_mrecv

#include <boost/test/included/unit_test.hpp>
#include <limits>
#include <cstddef>
#include <complex>
#include <vector>
#include <list>
#include <tuple>
#include <utility>
#include <mpl/mpl.hpp>
#include "test_helper.hpp"


template<typename T>
bool mprobe_test(const T &data) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0)
    comm_world.send(data, 1);
  if (comm_world.rank() == 1) {
    auto [m, s]{comm_world.mprobe(0)};
    if (s.source() != 0)
      return false;
    if constexpr (has_size<T>::value) {
      if (s.get_count<typename T::value_type>() != static_cast<int>(data.size()))
        return false;
    } else {
      if (s.get_count<T>() != 1)
        return false;
      T data_r;
      comm_world.mrecv(data_r, m);
      return data_r == data;
    }
  }
  return true;
}


template<typename T>
bool mprobe_iter_test(const T &data) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0)
    comm_world.send(data, 1);
  if (comm_world.rank() == 1) {
    auto [m, s]{comm_world.mprobe(0)};
    if (s.source() != 0)
      return false;
    int count{s.get_count<typename T::value_type>()};
    if (count != static_cast<int>(data.size()))
      return false;
    T data_r;
    if constexpr (has_resize<T>())
      data_r.resize(count);
    comm_world.mrecv(std::begin(data_r), std::end(data_r), m);
    return data_r == data;
  }
  return true;
}


template<typename T>
bool improbe_test(const T &data) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0)
    comm_world.send(data, 1);
  if (comm_world.rank() == 1) {
    while (true) {
      auto res{comm_world.improbe(0)};
      if (res) {
        auto [m, s]{res.value()};
        if (s.source() != 0)
          return false;
        if constexpr (has_size<T>::value) {
          if (s.get_count<typename T::value_type>() != static_cast<int>(data.size()))
            return false;
        } else {
          if (s.get_count<T>() != 1)
            return false;
        }
        T data_r;
        mpl::irequest request{comm_world.imrecv(data_r, m)};
        request.wait();
        return data_r == data;
      }
    }
  }
  return true;
}


template<typename T>
bool improbe_iter_test(const T &data) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0)
    comm_world.send(data, 1);
  if (comm_world.rank() == 1) {
    while (true) {
      auto res{comm_world.improbe(0)};
      if (res) {
        auto [m, s]{res.value()};
        if (s.source() != 0)
          return false;
        int count{s.get_count<typename T::value_type>()};
        if (count != static_cast<int>(data.size()))
          return false;
        T data_r;
        if constexpr (has_resize<T>())
          data_r.resize(count);
        mpl::irequest request{comm_world.imrecv(std::begin(data_r), std::end(data_r), m)};
        request.wait();
        return data_r == data;
      }
    }
  }
  return true;
}


BOOST_AUTO_TEST_CASE(mprobe) {
  // integer types
  BOOST_TEST(mprobe_test(std::byte(77)));
  BOOST_TEST(mprobe_test(std::numeric_limits<char>::max() - 1));
  BOOST_TEST(mprobe_test(std::numeric_limits<signed char>::max() - 1));
  BOOST_TEST(mprobe_test(std::numeric_limits<unsigned char>::max() - 1));
  BOOST_TEST(mprobe_test(std::numeric_limits<signed short>::max() - 1));
  BOOST_TEST(mprobe_test(std::numeric_limits<unsigned short>::max() - 1));
  BOOST_TEST(mprobe_test(std::numeric_limits<signed int>::max() - 1));
  BOOST_TEST(mprobe_test(std::numeric_limits<unsigned int>::max() - 1));
  BOOST_TEST(mprobe_test(std::numeric_limits<signed long>::max() - 1));
  BOOST_TEST(mprobe_test(std::numeric_limits<unsigned long>::max() - 1));
  BOOST_TEST(mprobe_test(std::numeric_limits<signed long long>::max() - 1));
  BOOST_TEST(mprobe_test(std::numeric_limits<unsigned long long>::max() - 1));
  // character types
  BOOST_TEST(mprobe_test(static_cast<wchar_t>('A')));
  BOOST_TEST(mprobe_test(static_cast<char16_t>('A')));
  BOOST_TEST(mprobe_test(static_cast<char32_t>('A')));
  // floating point number types
  BOOST_TEST(mprobe_test(static_cast<float>(3.14)));
  BOOST_TEST(mprobe_test(static_cast<double>(3.14)));
  BOOST_TEST(mprobe_test(static_cast<long double>(3.14)));
  BOOST_TEST(mprobe_test(std::complex<float>(3.14, 2.72)));
  BOOST_TEST(mprobe_test(std::complex<double>(3.14, 2.72)));
  BOOST_TEST(mprobe_test(std::complex<long double>(3.14, 2.72)));
  // logical type
  BOOST_TEST(mprobe_test(true));
  // enums
  enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };
  BOOST_TEST(mprobe_test(my_enum::val));
  // pairs, tuples and arrays
  BOOST_TEST(mprobe_test(std::pair<int, double>{1, 2.3}));
  BOOST_TEST(mprobe_test(std::tuple<int, double, bool>{1, 2.3, true}));
  BOOST_TEST(mprobe_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  // iterators
  BOOST_TEST(mprobe_iter_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  BOOST_TEST(mprobe_iter_test(std::vector<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(mprobe_iter_test(std::list<int>{1, 2, 3, 4, 5}));
}


BOOST_AUTO_TEST_CASE(improbe) {
  // integer types
  BOOST_TEST(improbe_test(std::byte(77)));
  BOOST_TEST(improbe_test(std::numeric_limits<char>::max() - 1));
  BOOST_TEST(improbe_test(std::numeric_limits<signed char>::max() - 1));
  BOOST_TEST(improbe_test(std::numeric_limits<unsigned char>::max() - 1));
  BOOST_TEST(improbe_test(std::numeric_limits<signed short>::max() - 1));
  BOOST_TEST(improbe_test(std::numeric_limits<unsigned short>::max() - 1));
  BOOST_TEST(improbe_test(std::numeric_limits<signed int>::max() - 1));
  BOOST_TEST(improbe_test(std::numeric_limits<unsigned int>::max() - 1));
  BOOST_TEST(improbe_test(std::numeric_limits<signed long>::max() - 1));
  BOOST_TEST(improbe_test(std::numeric_limits<unsigned long>::max() - 1));
  BOOST_TEST(improbe_test(std::numeric_limits<signed long long>::max() - 1));
  BOOST_TEST(improbe_test(std::numeric_limits<unsigned long long>::max() - 1));
  // character types
  BOOST_TEST(improbe_test(static_cast<wchar_t>('A')));
  BOOST_TEST(improbe_test(static_cast<char16_t>('A')));
  BOOST_TEST(improbe_test(static_cast<char32_t>('A')));
  // floating point number types
  BOOST_TEST(improbe_test(static_cast<float>(3.14)));
  BOOST_TEST(improbe_test(static_cast<double>(3.14)));
  BOOST_TEST(improbe_test(static_cast<long double>(3.14)));
  BOOST_TEST(improbe_test(std::complex<float>(3.14, 2.72)));
  BOOST_TEST(improbe_test(std::complex<double>(3.14, 2.72)));
  BOOST_TEST(improbe_test(std::complex<long double>(3.14, 2.72)));
  // logical type
  BOOST_TEST(improbe_test(true));
  // enums
  enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };
  BOOST_TEST(improbe_test(my_enum::val));
  // pairs, tuples and arrays
  BOOST_TEST(improbe_test(std::pair<int, double>{1, 2.3}));
  BOOST_TEST(improbe_test(std::tuple<int, double, bool>{1, 2.3, true}));
  BOOST_TEST(improbe_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  // iterators
  BOOST_TEST(improbe_iter_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  BOOST_TEST(improbe_iter_test(std::vector<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(improbe_iter_test(std::list<int>{1, 2, 3, 4, 5}));
}
