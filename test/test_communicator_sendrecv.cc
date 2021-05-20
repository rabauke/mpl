#define BOOST_TEST_MODULE communicator_sendrecv

#include <boost/test/included/unit_test.hpp>
#include <limits>
#include <cstddef>
#include <complex>
#include <mpl/mpl.hpp>


template<typename T>
bool sendrecv_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  const int rank{comm_world.rank()};
  const int size{comm_world.size()};
  T x[2]{T(rank), T(rank)};
  comm_world.sendrecv(x[1], (rank + 1) % size, mpl::tag(0), x[0], (rank - 1 + size) % size,
                      mpl::tag(0));
  return x[0] == T((rank - 1 + size) % size) and x[1] == T(rank);
}


template<typename T>
bool sendrecv_replace_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  const int rank{comm_world.rank()};
  int size{comm_world.size()};
  T x{T(rank)};
  comm_world.sendrecv_replace(x, (rank + 1) % size, mpl::tag(0), (rank - 1 + size) % size,
                              mpl::tag(0));
  return x == T((rank - 1 + size) % size);
}


BOOST_AUTO_TEST_CASE(sendrecv) {
  // integer types
  BOOST_TEST(sendrecv_test<std::byte>());
  BOOST_TEST(sendrecv_test<char>());
  BOOST_TEST(sendrecv_test<signed char>());
  BOOST_TEST(sendrecv_test<unsigned char>());
  BOOST_TEST(sendrecv_test<signed short>());
  BOOST_TEST(sendrecv_test<unsigned short>());
  BOOST_TEST(sendrecv_test<signed int>());
  BOOST_TEST(sendrecv_test<unsigned int>());
  BOOST_TEST(sendrecv_test<signed long>());
  BOOST_TEST(sendrecv_test<unsigned long>());
  BOOST_TEST(sendrecv_test<signed long long>());
  BOOST_TEST(sendrecv_test<unsigned long long>());
  // character types
  BOOST_TEST(sendrecv_test<wchar_t>());
  BOOST_TEST(sendrecv_test<char16_t>());
  BOOST_TEST(sendrecv_test<char32_t>());
  // floating point number types
  BOOST_TEST(sendrecv_test<float>());
  BOOST_TEST(sendrecv_test<double>());
  BOOST_TEST(sendrecv_test<long double>());
  BOOST_TEST(sendrecv_test<float>());
  BOOST_TEST(sendrecv_test<double>());
  BOOST_TEST(sendrecv_test<long double>());
  // logical type
  BOOST_TEST(sendrecv_test<bool>());
}


BOOST_AUTO_TEST_CASE(sendrecv_replace) {
  // integer types
  BOOST_TEST(sendrecv_replace_test<std::byte>());
  BOOST_TEST(sendrecv_replace_test<char>());
  BOOST_TEST(sendrecv_replace_test<signed char>());
  BOOST_TEST(sendrecv_replace_test<unsigned char>());
  BOOST_TEST(sendrecv_replace_test<signed short>());
  BOOST_TEST(sendrecv_replace_test<unsigned short>());
  BOOST_TEST(sendrecv_replace_test<signed int>());
  BOOST_TEST(sendrecv_replace_test<unsigned int>());
  BOOST_TEST(sendrecv_replace_test<signed long>());
  BOOST_TEST(sendrecv_replace_test<unsigned long>());
  BOOST_TEST(sendrecv_replace_test<signed long long>());
  BOOST_TEST(sendrecv_replace_test<unsigned long long>());
  // character types
  BOOST_TEST(sendrecv_replace_test<wchar_t>());
  BOOST_TEST(sendrecv_replace_test<char16_t>());
  BOOST_TEST(sendrecv_replace_test<char32_t>());
  // floating point number types
  BOOST_TEST(sendrecv_replace_test<float>());
  BOOST_TEST(sendrecv_replace_test<double>());
  BOOST_TEST(sendrecv_replace_test<long double>());
  BOOST_TEST(sendrecv_replace_test<float>());
  BOOST_TEST(sendrecv_replace_test<double>());
  BOOST_TEST(sendrecv_replace_test<long double>());
  // logical type
  BOOST_TEST(sendrecv_replace_test<bool>());
}
