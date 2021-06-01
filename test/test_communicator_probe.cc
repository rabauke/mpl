#define BOOST_TEST_MODULE communicator_probe

#include <boost/test/included/unit_test.hpp>
#include <limits>
#include <cstddef>
#include <complex>
#include <string>
#include <vector>
#include <list>
#include <set>
#include <tuple>
#include <utility>
#include <mpl/mpl.hpp>
#include "test_helper.hpp"


template<typename T>
bool probe_test(const T &data) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0)
    comm_world.send(data, 1);
  if (comm_world.rank() == 1) {
    mpl::status_t s{comm_world.probe(0)};
    if (s.source() != 0)
      return false;
    if constexpr (has_size<T>::value) {
      if (s.template get_count<typename T::value_type>() != data.size())
        return false;
    } else {
      if (s.template get_count<T>() != 1)
        return false;
    }
    T data_r;
    comm_world.recv(data_r, 0);
    return data_r == data;
  }
  return true;
}


template<typename T>
bool probe_iter_test(const T &data) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0)
    comm_world.send(data, 1);
  if (comm_world.rank() == 1) {
    mpl::status_t s{comm_world.probe(0)};
    if (s.source() != 0)
      return false;
    int count{s.template get_count<typename T::value_type>()};
    if (count != data.size())
      return false;
    T data_r;
    if constexpr (has_resize<T>())
      data_r.resize(count);
    comm_world.recv(std::begin(data_r), std::end(data_r), 0);
    return data_r == data;
  }
  return true;
}


template<typename T>
bool iprobe_test(const T &data) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0)
    comm_world.send(data, 1);
  if (comm_world.rank() == 1) {
    while (true) {
      auto s{comm_world.iprobe(0)};
      if (s) {
        if (s->source() != 0)
          return false;
        if constexpr (has_size<T>::value) {
          if (s->template get_count<typename T::value_type>() != data.size())
            return false;
        } else {
          if (s->template get_count<T>() != 1)
            return false;
        }
        T data_r;
        mpl::irequest request{comm_world.irecv(data_r, 0)};
        request.wait();
        return data_r == data;
      }
    }
  }
  return true;
}


template<typename T>
bool iprobe_iter_test(const T &data) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0)
    comm_world.send(data, 1);
  if (comm_world.rank() == 1) {
    while (true) {
      auto s{comm_world.iprobe(0)};
      if (s) {
        if (s->source() != 0)
          return false;
        int count{s->template get_count<typename T::value_type>()};
        if (count != data.size())
          return false;
        T data_r;
        if constexpr (has_resize<T>())
          data_r.resize(count);
        mpl::irequest request{comm_world.irecv(std::begin(data_r), std::end(data_r), 0)};
        request.wait();
        return data_r == data;
      }
    }
  }
  return true;
}


BOOST_AUTO_TEST_CASE(probe) {
  // integer types
  BOOST_TEST(probe_test(std::byte(77)));
  BOOST_TEST(probe_test(std::numeric_limits<char>::max() - 1));
  BOOST_TEST(probe_test(std::numeric_limits<signed char>::max() - 1));
  BOOST_TEST(probe_test(std::numeric_limits<unsigned char>::max() - 1));
  BOOST_TEST(probe_test(std::numeric_limits<signed short>::max() - 1));
  BOOST_TEST(probe_test(std::numeric_limits<unsigned short>::max() - 1));
  BOOST_TEST(probe_test(std::numeric_limits<signed int>::max() - 1));
  BOOST_TEST(probe_test(std::numeric_limits<unsigned int>::max() - 1));
  BOOST_TEST(probe_test(std::numeric_limits<signed long>::max() - 1));
  BOOST_TEST(probe_test(std::numeric_limits<unsigned long>::max() - 1));
  BOOST_TEST(probe_test(std::numeric_limits<signed long long>::max() - 1));
  BOOST_TEST(probe_test(std::numeric_limits<unsigned long long>::max() - 1));
  // character types
  BOOST_TEST(probe_test(static_cast<wchar_t>('A')));
  BOOST_TEST(probe_test(static_cast<char16_t>('A')));
  BOOST_TEST(probe_test(static_cast<char32_t>('A')));
  // floating point number types
  BOOST_TEST(probe_test(static_cast<float>(3.14)));
  BOOST_TEST(probe_test(static_cast<double>(3.14)));
  BOOST_TEST(probe_test(static_cast<long double>(3.14)));
  BOOST_TEST(probe_test(std::complex<float>(3.14, 2.72)));
  BOOST_TEST(probe_test(std::complex<double>(3.14, 2.72)));
  BOOST_TEST(probe_test(std::complex<long double>(3.14, 2.72)));
  // logical type
  BOOST_TEST(probe_test(true));
  // enums
  enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };
  BOOST_TEST(probe_test(my_enum::val));
  // pairs, tuples and arrays
  BOOST_TEST(probe_test(std::pair<int, double>{1, 2.3}));
  BOOST_TEST(probe_test(std::tuple<int, double, bool>{1, 2.3, true}));
  BOOST_TEST(probe_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  // strings and STL containers
  BOOST_TEST(probe_test(std::string{"Hello World"}));
  BOOST_TEST(probe_test(std::wstring{L"Hello World"}));
  BOOST_TEST(probe_test(std::vector<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(probe_test(std::list<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(probe_test(std::set<int>{1, 2, 3, 4, 5}));
  // iterators
  BOOST_TEST(probe_iter_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  BOOST_TEST(probe_iter_test(std::vector<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(probe_iter_test(std::list<int>{1, 2, 3, 4, 5}));
}


BOOST_AUTO_TEST_CASE(iprobe) {
  // integer types
  BOOST_TEST(iprobe_test(std::byte(77)));
  BOOST_TEST(iprobe_test(std::numeric_limits<char>::max() - 1));
  BOOST_TEST(iprobe_test(std::numeric_limits<signed char>::max() - 1));
  BOOST_TEST(iprobe_test(std::numeric_limits<unsigned char>::max() - 1));
  BOOST_TEST(iprobe_test(std::numeric_limits<signed short>::max() - 1));
  BOOST_TEST(iprobe_test(std::numeric_limits<unsigned short>::max() - 1));
  BOOST_TEST(iprobe_test(std::numeric_limits<signed int>::max() - 1));
  BOOST_TEST(iprobe_test(std::numeric_limits<unsigned int>::max() - 1));
  BOOST_TEST(iprobe_test(std::numeric_limits<signed long>::max() - 1));
  BOOST_TEST(iprobe_test(std::numeric_limits<unsigned long>::max() - 1));
  BOOST_TEST(iprobe_test(std::numeric_limits<signed long long>::max() - 1));
  BOOST_TEST(iprobe_test(std::numeric_limits<unsigned long long>::max() - 1));
  // character types
  BOOST_TEST(iprobe_test(static_cast<wchar_t>('A')));
  BOOST_TEST(iprobe_test(static_cast<char16_t>('A')));
  BOOST_TEST(iprobe_test(static_cast<char32_t>('A')));
  // floating point number types
  BOOST_TEST(iprobe_test(static_cast<float>(3.14)));
  BOOST_TEST(iprobe_test(static_cast<double>(3.14)));
  BOOST_TEST(iprobe_test(static_cast<long double>(3.14)));
  BOOST_TEST(iprobe_test(std::complex<float>(3.14, 2.72)));
  BOOST_TEST(iprobe_test(std::complex<double>(3.14, 2.72)));
  BOOST_TEST(iprobe_test(std::complex<long double>(3.14, 2.72)));
  // logical type
  BOOST_TEST(iprobe_test(true));
  // enums
  enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };
  BOOST_TEST(iprobe_test(my_enum::val));
  // pairs, tuples, arrays
  BOOST_TEST(iprobe_test(std::pair<int, double>{1, 2.3}));
  BOOST_TEST(iprobe_test(std::tuple<int, double, bool>{1, 2.3, true}));
  BOOST_TEST(iprobe_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  // strings and STL containers
  BOOST_TEST(iprobe_test(std::string{"Hello World"}));
  BOOST_TEST(iprobe_test(std::wstring{L"Hello World"}));
  BOOST_TEST(iprobe_test(std::vector<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(iprobe_test(std::list<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(iprobe_test(std::set<int>{1, 2, 3, 4, 5}));
  // iterators
  BOOST_TEST(iprobe_iter_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  BOOST_TEST(iprobe_iter_test(std::vector<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(iprobe_iter_test(std::list<int>{1, 2, 3, 4, 5}));
}
