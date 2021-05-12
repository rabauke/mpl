#define BOOST_TEST_MODULE isend_irecv

#include <boost/test/included/unit_test.hpp>
#include <limits>
#include <cstddef>
#include <complex>
#include <mpl/mpl.hpp>

template<typename T>
bool isend_irecv_test(const T &data) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    auto r{comm_world.isend(data, 1)};
    r.wait();
  }
  if (comm_world.rank() == 1) {
    T data_r;
    auto r{comm_world.irecv(data_r, 0)};
    while (not r.test().first) {
    }
    return data_r == data;
  }
  return true;
}

template<typename T>
bool ibsend_irecv_test(const T &data) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    const int size{comm_world.bsend_size<T>()};
    mpl::bsend_buffer<> buff(size);
    auto r{comm_world.ibsend(data, 1)};
    r.wait();
  }
  if (comm_world.rank() == 1) {
    T data_r;
    auto r{comm_world.irecv(data_r, 0)};
    while (not r.test().first) {
    }
    return data_r == data;
  }
  return true;
}

template<typename T>
bool issend_irecv_test(const T &data) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    auto r{comm_world.issend(data, 1)};
    r.wait();
  }
  if (comm_world.rank() == 1) {
    T data_r;
    auto r{comm_world.irecv(data_r, 0)};
    while (not r.test().first) {
    }
    return data_r == data;
  }
  return true;
}

template<typename T>
bool irsend_irecv_test(const T &data) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    comm_world.barrier();
    auto r{comm_world.irsend(data, 1)};
    r.wait();
  } else if (comm_world.rank() == 1) {
    T data_r;
    auto r{comm_world.irecv(data_r, 0)};
    comm_world.barrier();
    while (not r.test().first) {
    }
    return data_r == data;
  } else
    comm_world.barrier();
  return true;
}

BOOST_AUTO_TEST_CASE(isend_irecv) {
  // integer types
  BOOST_TEST(isend_irecv_test(std::byte(77)));
  BOOST_TEST(isend_irecv_test(std::numeric_limits<char>::max() - 1));
  BOOST_TEST(isend_irecv_test(std::numeric_limits<signed char>::max() - 1));
  BOOST_TEST(isend_irecv_test(std::numeric_limits<unsigned char>::max() - 1));
  BOOST_TEST(isend_irecv_test(std::numeric_limits<signed short>::max() - 1));
  BOOST_TEST(isend_irecv_test(std::numeric_limits<unsigned short>::max() - 1));
  BOOST_TEST(isend_irecv_test(std::numeric_limits<signed int>::max() - 1));
  BOOST_TEST(isend_irecv_test(std::numeric_limits<unsigned int>::max() - 1));
  BOOST_TEST(isend_irecv_test(std::numeric_limits<signed long>::max() - 1));
  BOOST_TEST(isend_irecv_test(std::numeric_limits<unsigned long>::max() - 1));
  BOOST_TEST(isend_irecv_test(std::numeric_limits<signed long long>::max() - 1));
  BOOST_TEST(isend_irecv_test(std::numeric_limits<unsigned long long>::max() - 1));
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
  enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };
  BOOST_TEST(isend_irecv_test(my_enum::val));
}

BOOST_AUTO_TEST_CASE(ibsend_irecv) {
  // integer types
  BOOST_TEST(ibsend_irecv_test(std::byte(77)));
  BOOST_TEST(ibsend_irecv_test(std::numeric_limits<char>::max() - 1));
  BOOST_TEST(ibsend_irecv_test(std::numeric_limits<signed char>::max() - 1));
  BOOST_TEST(ibsend_irecv_test(std::numeric_limits<unsigned char>::max() - 1));
  BOOST_TEST(ibsend_irecv_test(std::numeric_limits<signed short>::max() - 1));
  BOOST_TEST(ibsend_irecv_test(std::numeric_limits<unsigned short>::max() - 1));
  BOOST_TEST(ibsend_irecv_test(std::numeric_limits<signed int>::max() - 1));
  BOOST_TEST(ibsend_irecv_test(std::numeric_limits<unsigned int>::max() - 1));
  BOOST_TEST(ibsend_irecv_test(std::numeric_limits<signed long>::max() - 1));
  BOOST_TEST(ibsend_irecv_test(std::numeric_limits<unsigned long>::max() - 1));
  BOOST_TEST(ibsend_irecv_test(std::numeric_limits<signed long long>::max() - 1));
  BOOST_TEST(ibsend_irecv_test(std::numeric_limits<unsigned long long>::max() - 1));
  // character types
  BOOST_TEST(ibsend_irecv_test(static_cast<wchar_t>('A')));
  BOOST_TEST(ibsend_irecv_test(static_cast<char16_t>('A')));
  BOOST_TEST(ibsend_irecv_test(static_cast<char32_t>('A')));
  // floating point number types
  BOOST_TEST(ibsend_irecv_test(static_cast<float>(3.14)));
  BOOST_TEST(ibsend_irecv_test(static_cast<double>(3.14)));
  BOOST_TEST(ibsend_irecv_test(static_cast<long double>(3.14)));
  BOOST_TEST(ibsend_irecv_test(std::complex<float>(3.14, 2.72)));
  BOOST_TEST(ibsend_irecv_test(std::complex<double>(3.14, 2.72)));
  BOOST_TEST(ibsend_irecv_test(std::complex<long double>(3.14, 2.72)));
  // logical type
  BOOST_TEST(ibsend_irecv_test(true));
  // enums
  enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };
  BOOST_TEST(ibsend_irecv_test(my_enum::val));
}

BOOST_AUTO_TEST_CASE(issend_irecv) {
  // integer types
  BOOST_TEST(issend_irecv_test(std::byte(77)));
  BOOST_TEST(issend_irecv_test(std::numeric_limits<char>::max() - 1));
  BOOST_TEST(issend_irecv_test(std::numeric_limits<signed char>::max() - 1));
  BOOST_TEST(issend_irecv_test(std::numeric_limits<unsigned char>::max() - 1));
  BOOST_TEST(issend_irecv_test(std::numeric_limits<signed short>::max() - 1));
  BOOST_TEST(issend_irecv_test(std::numeric_limits<unsigned short>::max() - 1));
  BOOST_TEST(issend_irecv_test(std::numeric_limits<signed int>::max() - 1));
  BOOST_TEST(issend_irecv_test(std::numeric_limits<unsigned int>::max() - 1));
  BOOST_TEST(issend_irecv_test(std::numeric_limits<signed long>::max() - 1));
  BOOST_TEST(issend_irecv_test(std::numeric_limits<unsigned long>::max() - 1));
  BOOST_TEST(issend_irecv_test(std::numeric_limits<signed long long>::max() - 1));
  BOOST_TEST(issend_irecv_test(std::numeric_limits<unsigned long long>::max() - 1));
  // character types
  BOOST_TEST(issend_irecv_test(static_cast<wchar_t>('A')));
  BOOST_TEST(issend_irecv_test(static_cast<char16_t>('A')));
  BOOST_TEST(issend_irecv_test(static_cast<char32_t>('A')));
  // floating point number types
  BOOST_TEST(issend_irecv_test(static_cast<float>(3.14)));
  BOOST_TEST(issend_irecv_test(static_cast<double>(3.14)));
  BOOST_TEST(issend_irecv_test(static_cast<long double>(3.14)));
  BOOST_TEST(issend_irecv_test(std::complex<float>(3.14, 2.72)));
  BOOST_TEST(issend_irecv_test(std::complex<double>(3.14, 2.72)));
  BOOST_TEST(issend_irecv_test(std::complex<long double>(3.14, 2.72)));
  // logical type
  BOOST_TEST(issend_irecv_test(true));
  // enums
  enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };
  BOOST_TEST(issend_irecv_test(my_enum::val));
}

BOOST_AUTO_TEST_CASE(irsend_irecv) {
  // integer types
  BOOST_TEST(irsend_irecv_test(std::byte(77)));
  BOOST_TEST(irsend_irecv_test(std::numeric_limits<char>::max() - 1));
  BOOST_TEST(irsend_irecv_test(std::numeric_limits<signed char>::max() - 1));
  BOOST_TEST(irsend_irecv_test(std::numeric_limits<unsigned char>::max() - 1));
  BOOST_TEST(irsend_irecv_test(std::numeric_limits<signed short>::max() - 1));
  BOOST_TEST(irsend_irecv_test(std::numeric_limits<unsigned short>::max() - 1));
  BOOST_TEST(irsend_irecv_test(std::numeric_limits<signed int>::max() - 1));
  BOOST_TEST(irsend_irecv_test(std::numeric_limits<unsigned int>::max() - 1));
  BOOST_TEST(irsend_irecv_test(std::numeric_limits<signed long>::max() - 1));
  BOOST_TEST(irsend_irecv_test(std::numeric_limits<unsigned long>::max() - 1));
  BOOST_TEST(irsend_irecv_test(std::numeric_limits<signed long long>::max() - 1));
  BOOST_TEST(irsend_irecv_test(std::numeric_limits<unsigned long long>::max() - 1));
  // character types
  BOOST_TEST(irsend_irecv_test(static_cast<wchar_t>('A')));
  BOOST_TEST(irsend_irecv_test(static_cast<char16_t>('A')));
  BOOST_TEST(irsend_irecv_test(static_cast<char32_t>('A')));
  // floating point number types
  BOOST_TEST(irsend_irecv_test(static_cast<float>(3.14)));
  BOOST_TEST(irsend_irecv_test(static_cast<double>(3.14)));
  BOOST_TEST(irsend_irecv_test(static_cast<long double>(3.14)));
  BOOST_TEST(irsend_irecv_test(std::complex<float>(3.14, 2.72)));
  BOOST_TEST(irsend_irecv_test(std::complex<double>(3.14, 2.72)));
  BOOST_TEST(irsend_irecv_test(std::complex<long double>(3.14, 2.72)));
  // logical type
  BOOST_TEST(irsend_irecv_test(true));
  // enums
  enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };
  BOOST_TEST(irsend_irecv_test(my_enum::val));
}
