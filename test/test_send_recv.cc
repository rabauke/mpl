#define BOOST_TEST_MODULE send_recv

#include <boost/test/included/unit_test.hpp>
#include <limits>
#include <cstddef>
#include <complex>
#include <mpl/mpl.hpp>

template<typename T>
bool send_recv_test(const T &data) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0)
    comm_world.send(data, 1);
  if (comm_world.rank() == 1) {
    T data_r;
    comm_world.recv(data_r, 0);
    return data_r == data;
  }
  return true;
}

template<typename T>
bool bsend_recv_test(const T &data) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    const int size{comm_world.bsend_size<T>()};
    mpl::bsend_buffer<> buff(size);
    comm_world.bsend(data, 1);
  }
  if (comm_world.rank() == 1) {
    T data_r;
    comm_world.recv(data_r, 0);
    return data_r == data;
  }
  return true;
}

template<typename T>
bool ssend_recv_test(const T &data) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0)
    comm_world.ssend(data, 1);
  if (comm_world.rank() == 1) {
    T data_r;
    comm_world.recv(data_r, 0);
    return data_r == data;
  }
  return true;
}

template<typename T>
bool rsend_recv_test(const T &data) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    comm_world.barrier();
    comm_world.rsend(data, 1);
  } else if (comm_world.rank() == 1) {
    T data_r;
    mpl::irequest r{comm_world.irecv(data_r, 0)};
    comm_world.barrier();
    r.wait();
    return data_r == data;
  } else
    comm_world.barrier();
  return true;
}

BOOST_AUTO_TEST_CASE(send_recv) {
  // integer types
  BOOST_TEST(send_recv_test(std::byte(77)));
  BOOST_TEST(send_recv_test(std::numeric_limits<char>::max() - 1));
  BOOST_TEST(send_recv_test(std::numeric_limits<signed char>::max() - 1));
  BOOST_TEST(send_recv_test(std::numeric_limits<unsigned char>::max() - 1));
  BOOST_TEST(send_recv_test(std::numeric_limits<signed short>::max() - 1));
  BOOST_TEST(send_recv_test(std::numeric_limits<unsigned short>::max() - 1));
  BOOST_TEST(send_recv_test(std::numeric_limits<signed int>::max() - 1));
  BOOST_TEST(send_recv_test(std::numeric_limits<unsigned int>::max() - 1));
  BOOST_TEST(send_recv_test(std::numeric_limits<signed long>::max() - 1));
  BOOST_TEST(send_recv_test(std::numeric_limits<unsigned long>::max() - 1));
  BOOST_TEST(send_recv_test(std::numeric_limits<signed long long>::max() - 1));
  BOOST_TEST(send_recv_test(std::numeric_limits<unsigned long long>::max() - 1));
  // character types
  BOOST_TEST(send_recv_test(static_cast<wchar_t>('A')));
  BOOST_TEST(send_recv_test(static_cast<char16_t>('A')));
  BOOST_TEST(send_recv_test(static_cast<char32_t>('A')));
  // floating point number types
  BOOST_TEST(send_recv_test(static_cast<float>(3.14)));
  BOOST_TEST(send_recv_test(static_cast<double>(3.14)));
  BOOST_TEST(send_recv_test(static_cast<long double>(3.14)));
  BOOST_TEST(send_recv_test(std::complex<float>(3.14, 2.72)));
  BOOST_TEST(send_recv_test(std::complex<double>(3.14, 2.72)));
  BOOST_TEST(send_recv_test(std::complex<long double>(3.14, 2.72)));
  // logical type
  BOOST_TEST(send_recv_test(true));
  // enums
  enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };
  BOOST_TEST(send_recv_test(my_enum::val));
}

BOOST_AUTO_TEST_CASE(bsend_recv) {
  // integer types
  BOOST_TEST(bsend_recv_test(std::byte(77)));
  BOOST_TEST(bsend_recv_test(std::numeric_limits<char>::max() - 1));
  BOOST_TEST(bsend_recv_test(std::numeric_limits<signed char>::max() - 1));
  BOOST_TEST(bsend_recv_test(std::numeric_limits<unsigned char>::max() - 1));
  BOOST_TEST(bsend_recv_test(std::numeric_limits<signed short>::max() - 1));
  BOOST_TEST(bsend_recv_test(std::numeric_limits<unsigned short>::max() - 1));
  BOOST_TEST(bsend_recv_test(std::numeric_limits<signed int>::max() - 1));
  BOOST_TEST(bsend_recv_test(std::numeric_limits<unsigned int>::max() - 1));
  BOOST_TEST(bsend_recv_test(std::numeric_limits<signed long>::max() - 1));
  BOOST_TEST(bsend_recv_test(std::numeric_limits<unsigned long>::max() - 1));
  BOOST_TEST(bsend_recv_test(std::numeric_limits<signed long long>::max() - 1));
  BOOST_TEST(bsend_recv_test(std::numeric_limits<unsigned long long>::max() - 1));
  // character types
  BOOST_TEST(bsend_recv_test(static_cast<wchar_t>('A')));
  BOOST_TEST(bsend_recv_test(static_cast<char16_t>('A')));
  BOOST_TEST(bsend_recv_test(static_cast<char32_t>('A')));
  // floating point number types
  BOOST_TEST(bsend_recv_test(static_cast<float>(3.14)));
  BOOST_TEST(bsend_recv_test(static_cast<double>(3.14)));
  BOOST_TEST(bsend_recv_test(static_cast<long double>(3.14)));
  BOOST_TEST(bsend_recv_test(std::complex<float>(3.14, 2.72)));
  BOOST_TEST(bsend_recv_test(std::complex<double>(3.14, 2.72)));
  BOOST_TEST(bsend_recv_test(std::complex<long double>(3.14, 2.72)));
  // logical type
  BOOST_TEST(bsend_recv_test(true));
  // enums
  enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };
  BOOST_TEST(bsend_recv_test(my_enum::val));
}

BOOST_AUTO_TEST_CASE(ssend_recv) {
  // integer types
  BOOST_TEST(ssend_recv_test(std::byte(77)));
  BOOST_TEST(ssend_recv_test(std::numeric_limits<char>::max() - 1));
  BOOST_TEST(ssend_recv_test(std::numeric_limits<signed char>::max() - 1));
  BOOST_TEST(ssend_recv_test(std::numeric_limits<unsigned char>::max() - 1));
  BOOST_TEST(ssend_recv_test(std::numeric_limits<signed short>::max() - 1));
  BOOST_TEST(ssend_recv_test(std::numeric_limits<unsigned short>::max() - 1));
  BOOST_TEST(ssend_recv_test(std::numeric_limits<signed int>::max() - 1));
  BOOST_TEST(ssend_recv_test(std::numeric_limits<unsigned int>::max() - 1));
  BOOST_TEST(ssend_recv_test(std::numeric_limits<signed long>::max() - 1));
  BOOST_TEST(ssend_recv_test(std::numeric_limits<unsigned long>::max() - 1));
  BOOST_TEST(ssend_recv_test(std::numeric_limits<signed long long>::max() - 1));
  BOOST_TEST(ssend_recv_test(std::numeric_limits<unsigned long long>::max() - 1));
  // character types
  BOOST_TEST(ssend_recv_test(static_cast<wchar_t>('A')));
  BOOST_TEST(ssend_recv_test(static_cast<char16_t>('A')));
  BOOST_TEST(ssend_recv_test(static_cast<char32_t>('A')));
  // floating point number types
  BOOST_TEST(ssend_recv_test(static_cast<float>(3.14)));
  BOOST_TEST(ssend_recv_test(static_cast<double>(3.14)));
  BOOST_TEST(ssend_recv_test(static_cast<long double>(3.14)));
  BOOST_TEST(ssend_recv_test(std::complex<float>(3.14, 2.72)));
  BOOST_TEST(ssend_recv_test(std::complex<double>(3.14, 2.72)));
  BOOST_TEST(ssend_recv_test(std::complex<long double>(3.14, 2.72)));
  // logical type
  BOOST_TEST(ssend_recv_test(true));
  // enums
  enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };
  BOOST_TEST(ssend_recv_test(my_enum::val));
}

BOOST_AUTO_TEST_CASE(rsend_recv) {
  // integer types
  BOOST_TEST(rsend_recv_test(std::byte(77)));
  BOOST_TEST(rsend_recv_test(std::numeric_limits<char>::max() - 1));
  BOOST_TEST(rsend_recv_test(std::numeric_limits<signed char>::max() - 1));
  BOOST_TEST(rsend_recv_test(std::numeric_limits<unsigned char>::max() - 1));
  BOOST_TEST(rsend_recv_test(std::numeric_limits<signed short>::max() - 1));
  BOOST_TEST(rsend_recv_test(std::numeric_limits<unsigned short>::max() - 1));
  BOOST_TEST(rsend_recv_test(std::numeric_limits<signed int>::max() - 1));
  BOOST_TEST(rsend_recv_test(std::numeric_limits<unsigned int>::max() - 1));
  BOOST_TEST(rsend_recv_test(std::numeric_limits<signed long>::max() - 1));
  BOOST_TEST(rsend_recv_test(std::numeric_limits<unsigned long>::max() - 1));
  BOOST_TEST(rsend_recv_test(std::numeric_limits<signed long long>::max() - 1));
  BOOST_TEST(rsend_recv_test(std::numeric_limits<unsigned long long>::max() - 1));
  // character types
  BOOST_TEST(rsend_recv_test(static_cast<wchar_t>('A')));
  BOOST_TEST(rsend_recv_test(static_cast<char16_t>('A')));
  BOOST_TEST(rsend_recv_test(static_cast<char32_t>('A')));
  // floating point number types
  BOOST_TEST(rsend_recv_test(static_cast<float>(3.14)));
  BOOST_TEST(rsend_recv_test(static_cast<double>(3.14)));
  BOOST_TEST(rsend_recv_test(static_cast<long double>(3.14)));
  BOOST_TEST(rsend_recv_test(std::complex<float>(3.14, 2.72)));
  BOOST_TEST(rsend_recv_test(std::complex<double>(3.14, 2.72)));
  BOOST_TEST(rsend_recv_test(std::complex<long double>(3.14, 2.72)));
  // logical type
  BOOST_TEST(rsend_recv_test(true));
  // enums
  enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };
  BOOST_TEST(rsend_recv_test(my_enum::val));
}
