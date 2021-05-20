#define BOOST_TEST_MODULE communicator_init_send_init_recv

#include <boost/test/included/unit_test.hpp>
#include <limits>
#include <cstddef>
#include <complex>
#include <mpl/mpl.hpp>


template<typename T>
bool send_init_recv_init_test(const T &data) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    auto r{comm_world.send_init(data, 1)};
    r.start();
    r.wait();
  }
  if (comm_world.rank() == 1) {
    T data_r;
    auto r{comm_world.recv_init(data_r, 0)};
    r.start();
    while (not r.test().first) {
    }
    return data_r == data;
  }
  return true;
}


template<typename T>
bool bsend_init_recv_init_test(const T &data) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    const int size{comm_world.bsend_size<T>()};
    mpl::bsend_buffer<> buff(size);
    auto r{comm_world.bsend_init(data, 1)};
    r.start();
    r.wait();
  }
  if (comm_world.rank() == 1) {
    T data_r;
    auto r{comm_world.recv_init(data_r, 0)};
    r.start();
    while (not r.test().first) {
    }
    return data_r == data;
  }
  return true;
}


template<typename T>
bool ssend_init_recv_init_test(const T &data) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    auto r{comm_world.ssend_init(data, 1)};
    r.start();
    r.wait();
  }
  if (comm_world.rank() == 1) {
    T data_r;
    auto r{comm_world.recv_init(data_r, 0)};
    r.start();
    while (not r.test().first) {
    }
    return data_r == data;
  }
  return true;
}


template<typename T>
bool rsend_init_recv_init_test(const T &data) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    comm_world.barrier();
    auto r{comm_world.rsend_init(data, 1)};
    r.start();
    r.wait();
  } else if (comm_world.rank() == 1) {
    T data_r;
    auto r{comm_world.recv_init(data_r, 0)};
    r.start();
    comm_world.barrier();
    while (not r.test().first) {
    }
    return data_r == data;
  } else
    comm_world.barrier();
  return true;
}


BOOST_AUTO_TEST_CASE(send_init_recv_init) {
  // integer types
  BOOST_TEST(send_init_recv_init_test(std::byte(77)));
  BOOST_TEST(send_init_recv_init_test(std::numeric_limits<char>::max() - 1));
  BOOST_TEST(send_init_recv_init_test(std::numeric_limits<signed char>::max() - 1));
  BOOST_TEST(send_init_recv_init_test(std::numeric_limits<unsigned char>::max() - 1));
  BOOST_TEST(send_init_recv_init_test(std::numeric_limits<signed short>::max() - 1));
  BOOST_TEST(send_init_recv_init_test(std::numeric_limits<unsigned short>::max() - 1));
  BOOST_TEST(send_init_recv_init_test(std::numeric_limits<signed int>::max() - 1));
  BOOST_TEST(send_init_recv_init_test(std::numeric_limits<unsigned int>::max() - 1));
  BOOST_TEST(send_init_recv_init_test(std::numeric_limits<signed long>::max() - 1));
  BOOST_TEST(send_init_recv_init_test(std::numeric_limits<unsigned long>::max() - 1));
  BOOST_TEST(send_init_recv_init_test(std::numeric_limits<signed long long>::max() - 1));
  BOOST_TEST(send_init_recv_init_test(std::numeric_limits<unsigned long long>::max() - 1));
  // character types
  BOOST_TEST(send_init_recv_init_test(static_cast<wchar_t>('A')));
  BOOST_TEST(send_init_recv_init_test(static_cast<char16_t>('A')));
  BOOST_TEST(send_init_recv_init_test(static_cast<char32_t>('A')));
  // floating point number types
  BOOST_TEST(send_init_recv_init_test(static_cast<float>(3.14)));
  BOOST_TEST(send_init_recv_init_test(static_cast<double>(3.14)));
  BOOST_TEST(send_init_recv_init_test(static_cast<long double>(3.14)));
  BOOST_TEST(send_init_recv_init_test(std::complex<float>(3.14, 2.72)));
  BOOST_TEST(send_init_recv_init_test(std::complex<double>(3.14, 2.72)));
  BOOST_TEST(send_init_recv_init_test(std::complex<long double>(3.14, 2.72)));
  // logical type
  BOOST_TEST(send_init_recv_init_test(true));
  // enums
  enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };
  BOOST_TEST(send_init_recv_init_test(my_enum::val));
}


BOOST_AUTO_TEST_CASE(bsend_init_recv_init) {
  // integer types
  BOOST_TEST(bsend_init_recv_init_test(std::byte(77)));
  BOOST_TEST(bsend_init_recv_init_test(std::numeric_limits<char>::max() - 1));
  BOOST_TEST(bsend_init_recv_init_test(std::numeric_limits<signed char>::max() - 1));
  BOOST_TEST(bsend_init_recv_init_test(std::numeric_limits<unsigned char>::max() - 1));
  BOOST_TEST(bsend_init_recv_init_test(std::numeric_limits<signed short>::max() - 1));
  BOOST_TEST(bsend_init_recv_init_test(std::numeric_limits<unsigned short>::max() - 1));
  BOOST_TEST(bsend_init_recv_init_test(std::numeric_limits<signed int>::max() - 1));
  BOOST_TEST(bsend_init_recv_init_test(std::numeric_limits<unsigned int>::max() - 1));
  BOOST_TEST(bsend_init_recv_init_test(std::numeric_limits<signed long>::max() - 1));
  BOOST_TEST(bsend_init_recv_init_test(std::numeric_limits<unsigned long>::max() - 1));
  BOOST_TEST(bsend_init_recv_init_test(std::numeric_limits<signed long long>::max() - 1));
  BOOST_TEST(bsend_init_recv_init_test(std::numeric_limits<unsigned long long>::max() - 1));
  // character types
  BOOST_TEST(bsend_init_recv_init_test(static_cast<wchar_t>('A')));
  BOOST_TEST(bsend_init_recv_init_test(static_cast<char16_t>('A')));
  BOOST_TEST(bsend_init_recv_init_test(static_cast<char32_t>('A')));
  // floating point number types
  BOOST_TEST(bsend_init_recv_init_test(static_cast<float>(3.14)));
  BOOST_TEST(bsend_init_recv_init_test(static_cast<double>(3.14)));
  BOOST_TEST(bsend_init_recv_init_test(static_cast<long double>(3.14)));
  BOOST_TEST(bsend_init_recv_init_test(std::complex<float>(3.14, 2.72)));
  BOOST_TEST(bsend_init_recv_init_test(std::complex<double>(3.14, 2.72)));
  BOOST_TEST(bsend_init_recv_init_test(std::complex<long double>(3.14, 2.72)));
  // logical type
  BOOST_TEST(bsend_init_recv_init_test(true));
  // enums
  enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };
  BOOST_TEST(bsend_init_recv_init_test(my_enum::val));
}


BOOST_AUTO_TEST_CASE(ssend_init_recv_init) {
  // integer types
  BOOST_TEST(ssend_init_recv_init_test(std::byte(77)));
  BOOST_TEST(ssend_init_recv_init_test(std::numeric_limits<char>::max() - 1));
  BOOST_TEST(ssend_init_recv_init_test(std::numeric_limits<signed char>::max() - 1));
  BOOST_TEST(ssend_init_recv_init_test(std::numeric_limits<unsigned char>::max() - 1));
  BOOST_TEST(ssend_init_recv_init_test(std::numeric_limits<signed short>::max() - 1));
  BOOST_TEST(ssend_init_recv_init_test(std::numeric_limits<unsigned short>::max() - 1));
  BOOST_TEST(ssend_init_recv_init_test(std::numeric_limits<signed int>::max() - 1));
  BOOST_TEST(ssend_init_recv_init_test(std::numeric_limits<unsigned int>::max() - 1));
  BOOST_TEST(ssend_init_recv_init_test(std::numeric_limits<signed long>::max() - 1));
  BOOST_TEST(ssend_init_recv_init_test(std::numeric_limits<unsigned long>::max() - 1));
  BOOST_TEST(ssend_init_recv_init_test(std::numeric_limits<signed long long>::max() - 1));
  BOOST_TEST(ssend_init_recv_init_test(std::numeric_limits<unsigned long long>::max() - 1));
  // character types
  BOOST_TEST(ssend_init_recv_init_test(static_cast<wchar_t>('A')));
  BOOST_TEST(ssend_init_recv_init_test(static_cast<char16_t>('A')));
  BOOST_TEST(ssend_init_recv_init_test(static_cast<char32_t>('A')));
  // floating point number types
  BOOST_TEST(ssend_init_recv_init_test(static_cast<float>(3.14)));
  BOOST_TEST(ssend_init_recv_init_test(static_cast<double>(3.14)));
  BOOST_TEST(ssend_init_recv_init_test(static_cast<long double>(3.14)));
  BOOST_TEST(ssend_init_recv_init_test(std::complex<float>(3.14, 2.72)));
  BOOST_TEST(ssend_init_recv_init_test(std::complex<double>(3.14, 2.72)));
  BOOST_TEST(ssend_init_recv_init_test(std::complex<long double>(3.14, 2.72)));
  // logical type
  BOOST_TEST(ssend_init_recv_init_test(true));
  // enums
  enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };
  BOOST_TEST(ssend_init_recv_init_test(my_enum::val));
}


BOOST_AUTO_TEST_CASE(rsend_init_recv_init) {
  // integer types
  BOOST_TEST(rsend_init_recv_init_test(std::byte(77)));
  BOOST_TEST(rsend_init_recv_init_test(std::numeric_limits<char>::max() - 1));
  BOOST_TEST(rsend_init_recv_init_test(std::numeric_limits<signed char>::max() - 1));
  BOOST_TEST(rsend_init_recv_init_test(std::numeric_limits<unsigned char>::max() - 1));
  BOOST_TEST(rsend_init_recv_init_test(std::numeric_limits<signed short>::max() - 1));
  BOOST_TEST(rsend_init_recv_init_test(std::numeric_limits<unsigned short>::max() - 1));
  BOOST_TEST(rsend_init_recv_init_test(std::numeric_limits<signed int>::max() - 1));
  BOOST_TEST(rsend_init_recv_init_test(std::numeric_limits<unsigned int>::max() - 1));
  BOOST_TEST(rsend_init_recv_init_test(std::numeric_limits<signed long>::max() - 1));
  BOOST_TEST(rsend_init_recv_init_test(std::numeric_limits<unsigned long>::max() - 1));
  BOOST_TEST(rsend_init_recv_init_test(std::numeric_limits<signed long long>::max() - 1));
  BOOST_TEST(rsend_init_recv_init_test(std::numeric_limits<unsigned long long>::max() - 1));
  // character types
  BOOST_TEST(rsend_init_recv_init_test(static_cast<wchar_t>('A')));
  BOOST_TEST(rsend_init_recv_init_test(static_cast<char16_t>('A')));
  BOOST_TEST(rsend_init_recv_init_test(static_cast<char32_t>('A')));
  // floating point number types
  BOOST_TEST(rsend_init_recv_init_test(static_cast<float>(3.14)));
  BOOST_TEST(rsend_init_recv_init_test(static_cast<double>(3.14)));
  BOOST_TEST(rsend_init_recv_init_test(static_cast<long double>(3.14)));
  BOOST_TEST(rsend_init_recv_init_test(std::complex<float>(3.14, 2.72)));
  BOOST_TEST(rsend_init_recv_init_test(std::complex<double>(3.14, 2.72)));
  BOOST_TEST(rsend_init_recv_init_test(std::complex<long double>(3.14, 2.72)));
  // logical type
  BOOST_TEST(rsend_init_recv_init_test(true));
  // enums
  enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };
  BOOST_TEST(rsend_init_recv_init_test(my_enum::val));
}
