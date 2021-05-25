#define BOOST_TEST_MODULE communicator_init_send_init_recv

#include <boost/test/included/unit_test.hpp>
#include <limits>
#include <cstddef>
#include <complex>
#include <tuple>
#include <utility>
#include <mpl/mpl.hpp>


template<typename, typename = void>
struct has_size : std::false_type {};

template<typename T>
struct has_size<T, std::void_t<decltype(T().size())>> : std::true_type {};


template<typename, typename = void>
struct has_resize : std::false_type {};

template<typename T>
struct has_resize<T, std::void_t<decltype(T().resize(1))>> : std::true_type {};


template<typename T>
bool send_init_recv_init_test(const T &data) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    auto r{comm_world.send_init(data, 1)};
    r.start();
    r.wait();
    r.start();
    r.wait();
  }
  if (comm_world.rank() == 1) {
    bool ok{true};
    T data_r;
    auto r{comm_world.recv_init(data_r, 0)};
    r.start();
    while (not r.test().first) {
    }
    ok = ok and data_r == data;
    r.start();
    while (not r.test().first) {
    }
    ok = ok and data_r == data;
    return ok;
  }
  return true;
}


template<typename T>
bool send_init_recv_init_iter_test(const T &data) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    auto r{comm_world.send_init(std::begin(data), std::end(data), 1)};
    r.start();
    r.wait();
    r.start();
    r.wait();
  }
  if (comm_world.rank() == 1) {
    bool ok{true};
    T data_r;
    if constexpr (std::is_const_v<std::remove_reference_t<decltype(*std::begin(data_r))>>) {
      std::vector<typename T::value_type> data_t(data.size());
      auto r{comm_world.recv_init(std::begin(data_t), std::end(data_t), 0)};
      r.start();
      while (not r.test().first) {
      }
      data_r = T(std::begin(data_t), std::end(data_t));
      ok = ok and data_r == data;
      r.start();
      while (not r.test().first) {
      }
      data_r = T(std::begin(data_t), std::end(data_t));
      ok = ok and data_r == data;
      return ok;
    } else {
      if constexpr (has_resize<T>())
        data_r.resize(data.size());
      auto r{comm_world.recv_init(std::begin(data_r), std::end(data_r), 0)};
      r.start();
      while (not r.test().first) {
      }
      ok = ok and data_r == data;
      r.start();
      while (not r.test().first) {
      }
      ok = ok and data_r == data;
      return ok;
    }
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
    r.start();
    r.wait();
  }
  if (comm_world.rank() == 1) {
    bool ok{true};
    T data_r;
    auto r{comm_world.recv_init(data_r, 0)};
    r.start();
    while (not r.test().first) {
    }
    ok = ok and data_r == data;
    r.start();
    while (not r.test().first) {
    }
    ok = ok and data_r == data;
    return ok;
  }
  return true;
}


template<typename T>
bool bsend_init_recv_init_iter_test(const T &data) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    int size;
    if constexpr (has_size<T>::value)
      size = comm_world.bsend_size<typename T::value_type>(data.size());
    else
      size = comm_world.bsend_size<T>();
    mpl::bsend_buffer<> buff(size);
    auto r{comm_world.bsend_init(std::begin(data), std::end(data), 1)};
    r.start();
    r.wait();
    r.start();
    r.wait();
  }
  if (comm_world.rank() == 1) {
    bool ok{true};
    T data_r;
    if constexpr (std::is_const_v<std::remove_reference_t<decltype(*std::begin(data_r))>>) {
      std::vector<typename T::value_type> data_t(data.size());
      auto r{comm_world.recv_init(std::begin(data_t), std::end(data_t), 0)};
      r.start();
      while (not r.test().first) {
      }
      data_r = T(std::begin(data_t), std::end(data_t));
      ok = ok and data_r == data;
      r.start();
      while (not r.test().first) {
      }
      data_r = T(std::begin(data_t), std::end(data_t));
      ok = ok and data_r == data;
      return ok;
    } else {
      if constexpr (has_resize<T>())
        data_r.resize(data.size());
      auto r{comm_world.recv_init(std::begin(data_r), std::end(data_r), 0)};
      r.start();
      while (not r.test().first) {
      }
      ok = ok and data_r == data;
      r.start();
      while (not r.test().first) {
      }
      ok = ok and data_r == data;
      return ok;
    }
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
    r.start();
    r.wait();
  }
  if (comm_world.rank() == 1) {
    bool ok{true};
    T data_r;
    auto r{comm_world.recv_init(data_r, 0)};
    r.start();
    while (not r.test().first) {
    }
    ok = ok and data_r == data;
    r.start();
    while (not r.test().first) {
    }
    ok = ok and data_r == data;
    return ok;
  }
  return true;
}


template<typename T>
bool ssend_init_recv_init_iter_test(const T &data) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    auto r{comm_world.ssend_init(std::begin(data), std::end(data), 1)};
    r.start();
    r.wait();
    r.start();
    r.wait();
  }
  if (comm_world.rank() == 1) {
    bool ok{true};
    T data_r;
    if constexpr (std::is_const_v<std::remove_reference_t<decltype(*std::begin(data_r))>>) {
      std::vector<typename T::value_type> data_t(data.size());
      auto r{comm_world.recv_init(std::begin(data_t), std::end(data_t), 0)};
      r.start();
      while (not r.test().first) {
      }
      data_r = T(std::begin(data_t), std::end(data_t));
      ok = ok and data_r == data;
      r.start();
      while (not r.test().first) {
      }
      data_r = T(std::begin(data_t), std::end(data_t));
      ok = ok and data_r == data;
      return ok;
    } else {
      if constexpr (has_resize<T>())
        data_r.resize(data.size());
      auto r{comm_world.recv_init(std::begin(data_r), std::end(data_r), 0)};
      r.start();
      while (not r.test().first) {
      }
      ok = ok and data_r == data;
      r.start();
      while (not r.test().first) {
      }
      ok = ok and data_r == data;
      return ok;
    }
  }
  return true;
}


template<typename T>
bool rsend_init_recv_init_test(const T &data) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    auto r{comm_world.rsend_init(data, 1)};
    comm_world.barrier();
    r.start();
    r.wait();
    comm_world.barrier();
    r.start();
    r.wait();
  } else if (comm_world.rank() == 1) {
    bool ok{true};
    T data_r;
    auto r{comm_world.recv_init(data_r, 0)};
    r.start();
    comm_world.barrier();
    while (not r.test().first) {
    }
    ok = ok and data_r == data;
    comm_world.barrier();
    r.start();
    while (not r.test().first) {
    }
    ok = ok and data_r == data;
    return ok;
  } else {
    comm_world.barrier();
    comm_world.barrier();
  }
  return true;
}


template<typename T>
bool rsend_init_recv_init_iter_test(const T &data) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    auto r{comm_world.rsend_init(std::begin(data), std::end(data), 1)};
    comm_world.barrier();
    r.start();
    r.wait();
    comm_world.barrier();
    r.start();
    r.wait();
  } else if (comm_world.rank() == 1) {
    bool ok{true};
    T data_r;
    if constexpr (std::is_const_v<std::remove_reference_t<decltype(*std::begin(data_r))>>) {
      std::vector<typename T::value_type> data_t(data.size());
      auto r{comm_world.recv_init(std::begin(data_t), std::end(data_t), 0)};
      r.start();
      comm_world.barrier();
      while (not r.test().first) {
      }
      data_r = T(std::begin(data_t), std::end(data_t));
      ok = ok and data_r == data;
      r.start();
      comm_world.barrier();
      while (not r.test().first) {
      }
      data_r = T(std::begin(data_t), std::end(data_t));
      ok = ok and data_r == data;
      return ok;
    } else {
      if constexpr (has_resize<T>())
        data_r.resize(data.size());
      auto r{comm_world.recv_init(std::begin(data_r), std::end(data_r), 0)};
      r.start();
      comm_world.barrier();
      while (not r.test().first) {
      }
      ok = ok and data_r == data;
      r.start();
      comm_world.barrier();
      while (not r.test().first) {
      }
      ok = ok and data_r == data;
      return ok;
    }
  } else {
    comm_world.barrier();
    comm_world.barrier();
  }
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
  // pairs and tuples
  BOOST_TEST(send_init_recv_init_test(std::pair<int, double>{1, 2.3}));
  BOOST_TEST(send_init_recv_init_test(std::tuple<int, double, bool>{1, 2.3, true}));
  // iterators
  BOOST_TEST(send_init_recv_init_iter_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  BOOST_TEST(send_init_recv_init_iter_test(std::vector<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(send_init_recv_init_iter_test(std::list<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(send_init_recv_init_iter_test(std::set<int>{1, 2, 3, 4, 5}));
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
  // pairs and tuples
  BOOST_TEST(bsend_init_recv_init_test(std::pair<int, double>{1, 2.3}));
  BOOST_TEST(bsend_init_recv_init_test(std::tuple<int, double, bool>{1, 2.3, true}));
  // iterators
  BOOST_TEST(bsend_init_recv_init_iter_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  BOOST_TEST(bsend_init_recv_init_iter_test(std::vector<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(bsend_init_recv_init_iter_test(std::list<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(bsend_init_recv_init_iter_test(std::set<int>{1, 2, 3, 4, 5}));
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
  // pairs and tuples
  BOOST_TEST(ssend_init_recv_init_test(std::pair<int, double>{1, 2.3}));
  BOOST_TEST(ssend_init_recv_init_test(std::tuple<int, double, bool>{1, 2.3, true}));
  // iterators
  BOOST_TEST(ssend_init_recv_init_iter_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  BOOST_TEST(ssend_init_recv_init_iter_test(std::vector<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(ssend_init_recv_init_iter_test(std::list<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(ssend_init_recv_init_iter_test(std::set<int>{1, 2, 3, 4, 5}));
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
  // pairs and tuples
  BOOST_TEST(rsend_init_recv_init_test(std::pair<int, double>{1, 2.3}));
  BOOST_TEST(rsend_init_recv_init_test(std::tuple<int, double, bool>{1, 2.3, true}));
  // iterators
  BOOST_TEST(rsend_init_recv_init_iter_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  BOOST_TEST(rsend_init_recv_init_iter_test(std::vector<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(rsend_init_recv_init_iter_test(std::list<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(rsend_init_recv_init_iter_test(std::set<int>{1, 2, 3, 4, 5}));
}
