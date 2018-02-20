#define BOOST_TEST_MODULE send recv
#include <boost/test/included/unit_test.hpp>
#include <iostream>
#include <limits>
#include <cstddef>
#include <complex>
#include <mpl/mpl.hpp>

template<typename T>
bool isend_irecv_test(const T &data) {
  const mpl::communicator &comm_world=mpl::environment::comm_world();
  if (comm_world.size()<2)
    false;
  if (comm_world.rank()==0) {
    mpl::irequest r{ comm_world.isend(data, 1) };
    r.wait();
  }
  if (comm_world.rank()==1) {
    T data_r;
    mpl::irequest r{ comm_world.irecv(data_r, 0) };
    r.wait();
    return data_r==data;
  }
  return true;
}

BOOST_AUTO_TEST_CASE(isend_irecv)
{
  // integer types
#if __cplusplus>=201703L
  BOOST_TEST(isend_irecv_test(std::byte(77)));
#endif
  BOOST_TEST(isend_irecv_test(std::numeric_limits<char>::max()-1));
  BOOST_TEST(isend_irecv_test(std::numeric_limits<signed char>::max()-1));
  BOOST_TEST(isend_irecv_test(std::numeric_limits<unsigned char>::max()-1));
  BOOST_TEST(isend_irecv_test(std::numeric_limits<signed short>::max()-1));
  BOOST_TEST(isend_irecv_test(std::numeric_limits<unsigned short>::max()-1));
  BOOST_TEST(isend_irecv_test(std::numeric_limits<signed int>::max()-1));
  BOOST_TEST(isend_irecv_test(std::numeric_limits<unsigned int>::max()-1));
  BOOST_TEST(isend_irecv_test(std::numeric_limits<signed long>::max()-1));
  BOOST_TEST(isend_irecv_test(std::numeric_limits<unsigned long>::max()-1));
  BOOST_TEST(isend_irecv_test(std::numeric_limits<signed long long>::max()-1));
  BOOST_TEST(isend_irecv_test(std::numeric_limits<unsigned long long>::max()-1));
  // character types
  BOOST_TEST(isend_irecv_test(static_cast<wchar_t>('A')));
  BOOST_TEST(isend_irecv_test(static_cast<char16_t>('A')));
  BOOST_TEST(isend_irecv_test(static_cast<char32_t>('A')));
  // floating point number types
  BOOST_TEST(isend_irecv_test(static_cast<float>(3.14)));
  BOOST_TEST(isend_irecv_test(static_cast<double>(3.14)));
  BOOST_TEST(isend_irecv_test(static_cast<long double>(3.14)));
  BOOST_TEST(isend_irecv_test(std::complex<float>(3.14, 2.72)));
  BOOST_TEST(isend_irecv_test(std::complex<double>(3.14, 2.72)));
  BOOST_TEST(isend_irecv_test(std::complex<long double>(3.14, 2.72)));
  // logical type
  BOOST_TEST(isend_irecv_test(true));
  // enums
  enum class my_enum : int { val=std::numeric_limits<int>::max()-1 };
  BOOST_TEST(isend_irecv_test(my_enum::val));
}
