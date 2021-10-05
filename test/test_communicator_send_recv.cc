#define BOOST_TEST_MODULE communicator_send_recv

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
#include <algorithm>
#include <mpl/mpl.hpp>
#include "test_helper.hpp"


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
bool send_recv_iter_test(const T &data) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0)
    comm_world.send(std::begin(data), std::end(data), 1);
  if (comm_world.rank() == 1) {
    T data_r;
    if constexpr (std::is_const_v<std::remove_reference_t<decltype(*std::begin(data_r))>>) {
      comm_world.recv(data_r, 0);
      return data_r == data;
    } else {
      if constexpr (has_resize<T>())
        data_r.resize(data.size());
      comm_world.recv(std::begin(data_r), std::end(data_r), 0);
      return data_r == data;
    }
  }
  return true;
}


template<typename T>
bool bsend_recv_test(const T &data) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    int size;
    if constexpr (has_size<T>::value)
      size = comm_world.bsend_size<typename T::value_type>(data.size());
    else
      size = comm_world.bsend_size<T>();
    mpl::bsend_buffer buff(size);
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
bool bsend_recv_iter_test(const T &data) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    int size;
    if constexpr (has_size<T>::value)
      size = comm_world.bsend_size<typename T::value_type>(data.size());
    else
      size = comm_world.bsend_size<T>();
    mpl::bsend_buffer buff(size);
    comm_world.bsend(std::begin(data), std::end(data), 1);
  }
  if (comm_world.rank() == 1) {
    T data_r;
    if constexpr (std::is_const_v<std::remove_reference_t<decltype(*std::begin(data_r))>>) {
      T data_r;
      comm_world.recv(data_r, 0);
      return data_r == data;
    } else {
      if constexpr (has_resize<T>())
        data_r.resize(data.size());
      comm_world.recv(std::begin(data_r), std::end(data_r), 0);
      return data_r == data;
    }
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
bool ssend_recv_iter_test(const T &data) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0)
    comm_world.ssend(std::begin(data), std::end(data), 1);
  if (comm_world.rank() == 1) {
    T data_r;
    if constexpr (std::is_const_v<std::remove_reference_t<decltype(*std::begin(data_r))>>) {
      comm_world.recv(data_r, 0);
      return data_r == data;
    } else {
      if constexpr (has_resize<T>())
        data_r.resize(data.size());
      comm_world.recv(std::begin(data_r), std::end(data_r), 0);
      return data_r == data;
    }
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
    // must ensure that MPI_Recv is called before mpl::communicator::rsend
    if constexpr (has_begin_end<T>() and has_size<T>() and has_resize<T>()) {
      // T is an STL container
      T data_r;
      data_r.resize(data.size());
      mpl::irequest r{comm_world.irecv(begin(data_r), end(data_r), 0)};
      comm_world.barrier();
      r.wait();
      return data_r == data;
    } else if constexpr (has_begin_end<T>() and has_size<T>()) {
      // T is an STL container without resize member, e.g., std::set
      std::vector<typename T::value_type> data_r;
      data_r.resize(data.size());
      mpl::irequest r{comm_world.irecv(begin(data_r), end(data_r), 0)};
      comm_world.barrier();
      r.wait();
      return std::equal(begin(data_r), end(data_r), begin(data));
    } else {
      // T is some fundamental type
      // mpl::communicator::irecv does not suffice in the cases above as the irecv performs a
      // probe first to receive STL containers
      T data_r;
      mpl::irequest r{comm_world.irecv(data_r, 0)};
      comm_world.barrier();
      r.wait();
      return data_r == data;
    }
  } else
    comm_world.barrier();
  return true;
}


template<typename T>
bool rsend_recv_iter_test(const T &data) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    comm_world.barrier();
    comm_world.rsend(std::begin(data), std::end(data), 1);
  } else if (comm_world.rank() == 1) {
    // must ensure that MPI_Recv is called before mpl::communicator::rsend
    if constexpr (has_begin_end<T>() and has_size<T>() and has_resize<T>()) {
      T data_r;
      data_r.resize(data.size());
      mpl::irequest r{comm_world.irecv(begin(data_r), end(data_r), 0)};
      comm_world.barrier();
      r.wait();
      return data_r == data;
    } else if constexpr (has_begin_end<T>() and has_size<T>()) {
      // T is an STL container without resize member, e.g., std::set
      std::vector<typename T::value_type> data_r;
      data_r.resize(data.size());
      mpl::irequest r{comm_world.irecv(begin(data_r), end(data_r), 0)};
      comm_world.barrier();
      r.wait();
      return std::equal(begin(data_r), end(data_r), begin(data));
    } else {
      // T is some fundamental type
      // mpl::communicator::irecv does not suffice in the cases above as the irecv performs a
      // probe first to receive STL containers
      T data_r;
      mpl::irequest r{comm_world.irecv(data_r, 0)};
      comm_world.barrier();
      r.wait();
      return data_r == data;
    }
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
  // pairs, tuples and arrays
  BOOST_TEST(send_recv_test(std::pair<int, double>{1, 2.3}));
  BOOST_TEST(send_recv_test(std::tuple<int, double, bool>{1, 2.3, true}));
  BOOST_TEST(send_recv_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  // strings and STL containers
  BOOST_TEST(send_recv_test(std::string{"Hello World"}));
  BOOST_TEST(send_recv_test(std::wstring{L"Hello World"}));
  BOOST_TEST(send_recv_test(std::vector<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(send_recv_test(std::list<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(send_recv_test(std::set<int>{1, 2, 3, 4, 5}));
  // iterators
  BOOST_TEST(send_recv_iter_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  BOOST_TEST(send_recv_iter_test(std::vector<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(send_recv_iter_test(std::list<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(send_recv_iter_test(std::set<int>{1, 2, 3, 4, 5}));
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
  // pairs, tuples and arrays
  BOOST_TEST(bsend_recv_test(std::pair<int, double>{1, 2.3}));
  BOOST_TEST(bsend_recv_test(std::tuple<int, double, bool>{1, 2.3, true}));
  BOOST_TEST(bsend_recv_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  // strings and STL containers
  BOOST_TEST(bsend_recv_test(std::string{"Hello World"}));
  BOOST_TEST(bsend_recv_test(std::wstring{L"Hello World"}));
  BOOST_TEST(bsend_recv_test(std::vector<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(bsend_recv_test(std::list<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(bsend_recv_test(std::set<int>{1, 2, 3, 4, 5}));
  // iterators
  BOOST_TEST(bsend_recv_iter_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  BOOST_TEST(bsend_recv_iter_test(std::vector<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(bsend_recv_iter_test(std::list<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(bsend_recv_iter_test(std::set<int>{1, 2, 3, 4, 5}));
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
  // pairs, tuples and arrays
  BOOST_TEST(ssend_recv_test(std::pair<int, double>{1, 2.3}));
  BOOST_TEST(ssend_recv_test(std::tuple<int, double, bool>{1, 2.3, true}));
  BOOST_TEST(ssend_recv_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  // strings and STL containers
  BOOST_TEST(ssend_recv_test(std::string{"Hello World"}));
  BOOST_TEST(ssend_recv_test(std::wstring{L"Hello World"}));
  BOOST_TEST(ssend_recv_test(std::vector<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(ssend_recv_test(std::list<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(ssend_recv_test(std::set<int>{1, 2, 3, 4, 5}));
  // iterators
  BOOST_TEST(ssend_recv_iter_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  BOOST_TEST(ssend_recv_iter_test(std::vector<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(ssend_recv_iter_test(std::list<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(ssend_recv_iter_test(std::set<int>{1, 2, 3, 4, 5}));
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
  // pairs, tuples and arrays
  BOOST_TEST(rsend_recv_test(std::pair<int, double>{1, 2.3}));
  BOOST_TEST(rsend_recv_test(std::tuple<int, double, bool>{1, 2.3, true}));
  BOOST_TEST(rsend_recv_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  // strings and STL containers
  BOOST_TEST(rsend_recv_test(std::string{"Hello World"}));
  BOOST_TEST(rsend_recv_test(std::wstring{L"Hello World"}));
  BOOST_TEST(rsend_recv_test(std::vector<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(rsend_recv_test(std::list<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(rsend_recv_test(std::set<int>{1, 2, 3, 4, 5}));
  // iterators
  BOOST_TEST(rsend_recv_iter_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  BOOST_TEST(rsend_recv_iter_test(std::vector<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(rsend_recv_iter_test(std::list<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(rsend_recv_iter_test(std::set<int>{1, 2, 3, 4, 5}));
}
