#define BOOST_TEST_MODULE send_recv

#include <boost/test/included/unit_test.hpp>
#include <iostream>
#include <limits>
#include <cstddef>
#include <complex>
#include <mpl/mpl.hpp>

template<typename T>
bool send_recv_test(const T &data) {
  const mpl::communicator &comm_world=mpl::environment::comm_world();
  if (comm_world.size()<2)
    false;
  if (comm_world.rank()==0)
    comm_world.send(data, 1);
  if (comm_world.rank()==1) {
    T data_r;
    comm_world.recv(data_r, 0);
    return data_r==data;
  }
  return true;
}

template<typename T>
bool bsend_recv_test(const T &data) {
  const mpl::communicator &comm_world=mpl::environment::comm_world();
  if (comm_world.size()<2)
    false;
  if (comm_world.rank()==0) {
    const int size{comm_world.bsend_size<T>()};
    mpl::bsend_buffer<> buff(size);
    comm_world.bsend(data, 1);
  }
  if (comm_world.rank()==1) {
    T data_r;
    comm_world.recv(data_r, 0);
    return data_r==data;
  }
  return true;
}

template<typename T>
bool ssend_recv_test(const T &data) {
  const mpl::communicator &comm_world=mpl::environment::comm_world();
  if (comm_world.size()<2)
    false;
  if (comm_world.rank()==0)
    comm_world.ssend(data, 1);
  if (comm_world.rank()==1) {
    T data_r;
    comm_world.recv(data_r, 0);
    return data_r==data;
  }
  return true;
}

template<typename T>
bool rsend_recv_test(const T &data) {
  const mpl::communicator &comm_world=mpl::environment::comm_world();
  if (comm_world.size()<2)
    false;
  if (comm_world.rank()==0)
    comm_world.rsend(data, 1);
  if (comm_world.rank()==1) {
    T data_r;
    comm_world.recv(data_r, 0);
    return data_r==data;
  }
  return true;
}

template<typename T>
bool sendrecv_test() {
  const mpl::communicator &comm_world=mpl::environment::comm_world();
  int rank=comm_world.rank();
  int size=comm_world.size();
  T x[2]={T(rank), T(rank)};
  comm_world.sendrecv(x[1], (rank+1)%size, mpl::tag(0),
                      x[0], (rank-1+size)%size, mpl::tag(0));
  return x[0]==T((rank-1+size)%size) and x[1]==T(rank);
}


BOOST_AUTO_TEST_CASE(send_recv) {
  // integer types
#if __cplusplus>=201703L
  BOOST_TEST(send_recv_test(std::byte(77)));
#endif
  BOOST_TEST(send_recv_test(std::numeric_limits<char>::max()-1));
  BOOST_TEST(send_recv_test(std::numeric_limits<signed char>::max()-1));
  BOOST_TEST(send_recv_test(std::numeric_limits<unsigned char>::max()-1));
  BOOST_TEST(send_recv_test(std::numeric_limits<signed short>::max()-1));
  BOOST_TEST(send_recv_test(std::numeric_limits<unsigned short>::max()-1));
  BOOST_TEST(send_recv_test(std::numeric_limits<signed int>::max()-1));
  BOOST_TEST(send_recv_test(std::numeric_limits<unsigned int>::max()-1));
  BOOST_TEST(send_recv_test(std::numeric_limits<signed long>::max()-1));
  BOOST_TEST(send_recv_test(std::numeric_limits<unsigned long>::max()-1));
  BOOST_TEST(send_recv_test(std::numeric_limits<signed long long>::max()-1));
  BOOST_TEST(send_recv_test(std::numeric_limits<unsigned long long>::max()-1));
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
  enum class my_enum : int {
    val=std::numeric_limits<int>::max()-1
  };
  BOOST_TEST(send_recv_test(my_enum::val));
}

BOOST_AUTO_TEST_CASE(bsend_recv) {
  // integer types
#if __cplusplus>=201703L
  BOOST_TEST(bsend_recv_test(std::byte(77)));
#endif
  BOOST_TEST(bsend_recv_test(std::numeric_limits<char>::max()-1));
  BOOST_TEST(bsend_recv_test(std::numeric_limits<signed char>::max()-1));
  BOOST_TEST(bsend_recv_test(std::numeric_limits<unsigned char>::max()-1));
  BOOST_TEST(bsend_recv_test(std::numeric_limits<signed short>::max()-1));
  BOOST_TEST(bsend_recv_test(std::numeric_limits<unsigned short>::max()-1));
  BOOST_TEST(bsend_recv_test(std::numeric_limits<signed int>::max()-1));
  BOOST_TEST(bsend_recv_test(std::numeric_limits<unsigned int>::max()-1));
  BOOST_TEST(bsend_recv_test(std::numeric_limits<signed long>::max()-1));
  BOOST_TEST(bsend_recv_test(std::numeric_limits<unsigned long>::max()-1));
  BOOST_TEST(bsend_recv_test(std::numeric_limits<signed long long>::max()-1));
  BOOST_TEST(bsend_recv_test(std::numeric_limits<unsigned long long>::max()-1));
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
  enum class my_enum : int {
    val=std::numeric_limits<int>::max()-1
  };
  BOOST_TEST(bsend_recv_test(my_enum::val));
}

BOOST_AUTO_TEST_CASE(ssend_recv) {
  // integer types
#if __cplusplus>=201703L
  BOOST_TEST(ssend_recv_test(std::byte(77)));
#endif
  BOOST_TEST(ssend_recv_test(std::numeric_limits<char>::max()-1));
  BOOST_TEST(ssend_recv_test(std::numeric_limits<signed char>::max()-1));
  BOOST_TEST(ssend_recv_test(std::numeric_limits<unsigned char>::max()-1));
  BOOST_TEST(ssend_recv_test(std::numeric_limits<signed short>::max()-1));
  BOOST_TEST(ssend_recv_test(std::numeric_limits<unsigned short>::max()-1));
  BOOST_TEST(ssend_recv_test(std::numeric_limits<signed int>::max()-1));
  BOOST_TEST(ssend_recv_test(std::numeric_limits<unsigned int>::max()-1));
  BOOST_TEST(ssend_recv_test(std::numeric_limits<signed long>::max()-1));
  BOOST_TEST(ssend_recv_test(std::numeric_limits<unsigned long>::max()-1));
  BOOST_TEST(ssend_recv_test(std::numeric_limits<signed long long>::max()-1));
  BOOST_TEST(ssend_recv_test(std::numeric_limits<unsigned long long>::max()-1));
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
  enum class my_enum : int {
    val=std::numeric_limits<int>::max()-1
  };
  BOOST_TEST(ssend_recv_test(my_enum::val));
}

BOOST_AUTO_TEST_CASE(rsend_recv) {
  // integer types
#if __cplusplus>=201703L
  BOOST_TEST(rsend_recv_test(std::byte(77)));
#endif
  BOOST_TEST(rsend_recv_test(std::numeric_limits<char>::max()-1));
  BOOST_TEST(rsend_recv_test(std::numeric_limits<signed char>::max()-1));
  BOOST_TEST(rsend_recv_test(std::numeric_limits<unsigned char>::max()-1));
  BOOST_TEST(rsend_recv_test(std::numeric_limits<signed short>::max()-1));
  BOOST_TEST(rsend_recv_test(std::numeric_limits<unsigned short>::max()-1));
  BOOST_TEST(rsend_recv_test(std::numeric_limits<signed int>::max()-1));
  BOOST_TEST(rsend_recv_test(std::numeric_limits<unsigned int>::max()-1));
  BOOST_TEST(rsend_recv_test(std::numeric_limits<signed long>::max()-1));
  BOOST_TEST(rsend_recv_test(std::numeric_limits<unsigned long>::max()-1));
  BOOST_TEST(rsend_recv_test(std::numeric_limits<signed long long>::max()-1));
  BOOST_TEST(rsend_recv_test(std::numeric_limits<unsigned long long>::max()-1));
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
  enum class my_enum : int {
    val=std::numeric_limits<int>::max()-1
  };
  BOOST_TEST(rsend_recv_test(my_enum::val));
}

BOOST_AUTO_TEST_CASE(sendrecv) {
  // integer types
#if __cplusplus>=201703L
  BOOST_TEST(sendrecv_test<std::byte>());
#endif
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
