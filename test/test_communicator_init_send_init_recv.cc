#include <catch2/catch_test_macros.hpp>
#include <complex>
#include <cstddef>
#include <limits>
#include <list>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <mpl/mpl.hpp>
#include "test_helper.hpp"


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
    while (not r.test()) {
    }
    ok = ok and data_r == data;
    r.start();
    while (not r.test()) {
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
      while (not r.test()) {
      }
      data_r = T(std::begin(data_t), std::end(data_t));
      ok = ok and data_r == data;
      r.start();
      while (not r.test()) {
      }
      data_r = T(std::begin(data_t), std::end(data_t));
      ok = ok and data_r == data;
      return ok;
    } else {
      if constexpr (has_resize<T>())
        data_r.resize(data.size());
      auto r{comm_world.recv_init(std::begin(data_r), std::end(data_r), 0)};
      r.start();
      while (not r.test()) {
      }
      ok = ok and data_r == data;
      r.start();
      while (not r.test()) {
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
    int size;
    if constexpr (has_size<T>::value)
      size = comm_world.bsend_size<typename T::value_type>(data.size());
    else
      size = comm_world.bsend_size<T>();
    mpl::bsend_buffer buff(size);
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
    while (not r.test()) {
    }
    ok = ok and data_r == data;
    r.start();
    while (not r.test()) {
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
    mpl::bsend_buffer buff(size);
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
      while (not r.test()) {
      }
      data_r = T(std::begin(data_t), std::end(data_t));
      ok = ok and data_r == data;
      r.start();
      while (not r.test()) {
      }
      data_r = T(std::begin(data_t), std::end(data_t));
      ok = ok and data_r == data;
      return ok;
    } else {
      if constexpr (has_resize<T>())
        data_r.resize(data.size());
      auto r{comm_world.recv_init(std::begin(data_r), std::end(data_r), 0)};
      r.start();
      while (not r.test()) {
      }
      ok = ok and data_r == data;
      r.start();
      while (not r.test()) {
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
    while (not r.test()) {
    }
    ok = ok and data_r == data;
    r.start();
    while (not r.test()) {
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
      while (not r.test()) {
      }
      data_r = T(std::begin(data_t), std::end(data_t));
      ok = ok and data_r == data;
      r.start();
      while (not r.test()) {
      }
      data_r = T(std::begin(data_t), std::end(data_t));
      ok = ok and data_r == data;
      return ok;
    } else {
      if constexpr (has_resize<T>())
        data_r.resize(data.size());
      auto r{comm_world.recv_init(std::begin(data_r), std::end(data_r), 0)};
      r.start();
      while (not r.test()) {
      }
      ok = ok and data_r == data;
      r.start();
      while (not r.test()) {
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
    // must ensure that MPI_Recv_init is called before mpl::communicator::rsend_init
    if constexpr (has_begin_end<T>() and has_size<T>() and has_resize<T>()) {
      // T is an STL container
      T data_r;
      data_r.resize(data.size());
      auto r{comm_world.recv_init(std::begin(data_r), std::end(data_r), 0)};
      r.start();
      comm_world.barrier();
      while (not r.test()) {
      }
      ok = ok and data_r == data;
      r.start();
      comm_world.barrier();
      while (not r.test()) {
      }
      ok = ok and data_r == data;
      return ok;
    } else if constexpr (has_begin_end<T>() and has_size<T>() and not has_resize<T>()) {
      if constexpr (std::is_const_v<std::remove_reference_t<decltype(*std::begin(data))>>) {
        T data_r;
        auto r{comm_world.recv_init(data_r, 0)};
        r.start();
        comm_world.barrier();
        while (not r.test()) {
        }
        ok = ok and data_r == data;
        r.start();
        comm_world.barrier();
        while (not r.test()) {
        }
        ok = ok and data_r == data;
        return ok;
      } else {
        // T is an std::array
        T data_r;
        auto r{comm_world.recv_init(std::begin(data_r), std::end(data_r), 0)};
        r.start();
        comm_world.barrier();
        while (not r.test()) {
        }
        ok = ok and data_r == data;
        r.start();
        comm_world.barrier();
        while (not r.test()) {
        }
        ok = ok and data_r == data;
        return ok;
      }
    } else {
      // T is some fundamental type
      T data_r;
      auto r{comm_world.recv_init(data_r, 0)};
      r.start();
      comm_world.barrier();
      while (not r.test()) {
      }
      ok = ok and data_r == data;
      r.start();
      comm_world.barrier();
      while (not r.test()) {
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
      while (not r.test()) {
      }
      data_r = T(std::begin(data_t), std::end(data_t));
      ok = ok and data_r == data;
      r.start();
      comm_world.barrier();
      while (not r.test()) {
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
      while (not r.test()) {
      }
      ok = ok and data_r == data;
      r.start();
      comm_world.barrier();
      while (not r.test()) {
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


bool prequest_pool_individual_test() {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    int val1{10};
    int val2{20};
    mpl::prequest_pool pool;
    pool.push(comm_world.send_init(val1, 1, mpl::tag_t(301)));
    pool.push(comm_world.send_init(val2, 1, mpl::tag_t(302)));
    for (int iter{0}; iter < 3; ++iter) {
      val1 = 10 * (iter + 1);
      val2 = 20 * (iter + 1);
      pool.start(0);
      pool.wait(0);
      pool.start(1);
      while (not pool.test(1)) {
      }
    }
    return true;
  }
  if (comm_world.rank() == 1) {
    int val1{0};
    int val2{0};
    mpl::prequest_pool pool;
    pool.push(comm_world.recv_init(val1, 0, mpl::tag_t(301)));
    pool.push(comm_world.recv_init(val2, 0, mpl::tag_t(302)));
    bool ok{true};
    for (int iter{0}; iter < 3; ++iter) {
      pool.start(0);
      auto status0{pool.wait(0)};
      ok = ok and (val1 == 10 * (iter + 1));
      ok = ok and
           (status0.tag() == mpl::tag_t(301) and pool.get_status(0).tag() == mpl::tag_t(301));
      pool.start(1);
      while (not pool.test(1)) {
      }
      ok = ok and (val2 == 20 * (iter + 1));
      ok = ok and (pool.get_status(1).tag() == mpl::tag_t(302));
    }
    return ok;
  }
  return true;
}


SCENARIO("persistent communication requests and request pools",
         "[communicator][init_send_init_recv]") {
  const mpl::communicator &comm_world = mpl::environment::comm_world();

  GIVEN("a world communicator configured for persistent communication") {
    if (comm_world.size() < 2)
      return;

    WHEN("transmitting data iteratively using send_init and recv_init") {
      THEN("data is received across multiple start/wait cycles correctly") {
        REQUIRE(send_init_recv_init_test(std::byte(77)));
        REQUIRE(send_init_recv_init_test(std::numeric_limits<char>::max() - 1));
        REQUIRE(send_init_recv_init_test(std::numeric_limits<signed char>::max() - 1));
        REQUIRE(send_init_recv_init_test(std::numeric_limits<unsigned char>::max() - 1));
        REQUIRE(send_init_recv_init_test(std::numeric_limits<signed short>::max() - 1));
        REQUIRE(send_init_recv_init_test(std::numeric_limits<unsigned short>::max() - 1));
        REQUIRE(send_init_recv_init_test(std::numeric_limits<signed int>::max() - 1));
        REQUIRE(send_init_recv_init_test(std::numeric_limits<unsigned int>::max() - 1));
        REQUIRE(send_init_recv_init_test(std::numeric_limits<signed long>::max() - 1));
        REQUIRE(send_init_recv_init_test(std::numeric_limits<unsigned long>::max() - 1));
        REQUIRE(send_init_recv_init_test(std::numeric_limits<signed long long>::max() - 1));
        REQUIRE(send_init_recv_init_test(std::numeric_limits<unsigned long long>::max() - 1));
        REQUIRE(send_init_recv_init_test(static_cast<wchar_t>('A')));
        REQUIRE(send_init_recv_init_test(static_cast<char16_t>('A')));
        REQUIRE(send_init_recv_init_test(static_cast<char32_t>('A')));
        REQUIRE(send_init_recv_init_test(static_cast<float>(3.14)));
        REQUIRE(send_init_recv_init_test(static_cast<double>(3.14)));
        REQUIRE(send_init_recv_init_test(static_cast<long double>(3.14)));
        REQUIRE(send_init_recv_init_test(std::complex<float>(3.14, 2.72)));
        REQUIRE(send_init_recv_init_test(std::complex<double>(3.14, 2.72)));
        REQUIRE(send_init_recv_init_test(std::complex<long double>(3.14, 2.72)));
        REQUIRE(send_init_recv_init_test(true));
        enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };
        REQUIRE(send_init_recv_init_test(my_enum::val));
        REQUIRE(send_init_recv_init_test(std::pair<int, double>{1, 2.3}));
        REQUIRE(send_init_recv_init_test(std::tuple<int, double, bool>{1, 2.3, true}));
        REQUIRE(send_init_recv_init_test(std::array<int, 5>{1, 2, 3, 4, 5}));
        REQUIRE(send_init_recv_init_iter_test(std::array<int, 5>{1, 2, 3, 4, 5}));
        REQUIRE(send_init_recv_init_iter_test(std::vector<int>{1, 2, 3, 4, 5}));
        REQUIRE(send_init_recv_init_iter_test(std::list<int>{1, 2, 3, 4, 5}));
        REQUIRE(send_init_recv_init_iter_test(std::set<int>{1, 2, 3, 4, 5}));
      }
    }

    WHEN("transmitting data iteratively using bsend_init and recv_init") {
      THEN("data is received across multiple start/wait cycles correctly") {
        REQUIRE(bsend_init_recv_init_test(std::byte(77)));
        REQUIRE(bsend_init_recv_init_test(std::numeric_limits<char>::max() - 1));
        REQUIRE(bsend_init_recv_init_test(std::numeric_limits<signed char>::max() - 1));
        REQUIRE(bsend_init_recv_init_test(std::numeric_limits<unsigned char>::max() - 1));
        REQUIRE(bsend_init_recv_init_test(std::numeric_limits<signed short>::max() - 1));
        REQUIRE(bsend_init_recv_init_test(std::numeric_limits<unsigned short>::max() - 1));
        REQUIRE(bsend_init_recv_init_test(std::numeric_limits<signed int>::max() - 1));
        REQUIRE(bsend_init_recv_init_test(std::numeric_limits<unsigned int>::max() - 1));
        REQUIRE(bsend_init_recv_init_test(std::numeric_limits<signed long>::max() - 1));
        REQUIRE(bsend_init_recv_init_test(std::numeric_limits<unsigned long>::max() - 1));
        REQUIRE(bsend_init_recv_init_test(std::numeric_limits<signed long long>::max() - 1));
        REQUIRE(bsend_init_recv_init_test(std::numeric_limits<unsigned long long>::max() - 1));
        REQUIRE(bsend_init_recv_init_test(static_cast<wchar_t>('A')));
        REQUIRE(bsend_init_recv_init_test(static_cast<char16_t>('A')));
        REQUIRE(bsend_init_recv_init_test(static_cast<char32_t>('A')));
        REQUIRE(bsend_init_recv_init_test(static_cast<float>(3.14)));
        REQUIRE(bsend_init_recv_init_test(static_cast<double>(3.14)));
        REQUIRE(bsend_init_recv_init_test(static_cast<long double>(3.14)));
        REQUIRE(bsend_init_recv_init_test(std::complex<float>(3.14, 2.72)));
        REQUIRE(bsend_init_recv_init_test(std::complex<double>(3.14, 2.72)));
        REQUIRE(bsend_init_recv_init_test(std::complex<long double>(3.14, 2.72)));
        REQUIRE(bsend_init_recv_init_test(true));
        enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };
        REQUIRE(bsend_init_recv_init_test(my_enum::val));
        REQUIRE(bsend_init_recv_init_test(std::pair<int, double>{1, 2.3}));
        REQUIRE(bsend_init_recv_init_test(std::tuple<int, double, bool>{1, 2.3, true}));
        REQUIRE(bsend_init_recv_init_test(std::array<int, 5>{1, 2, 3, 4, 5}));
        REQUIRE(bsend_init_recv_init_iter_test(std::array<int, 5>{1, 2, 3, 4, 5}));
        REQUIRE(bsend_init_recv_init_iter_test(std::vector<int>{1, 2, 3, 4, 5}));
        REQUIRE(bsend_init_recv_init_iter_test(std::list<int>{1, 2, 3, 4, 5}));
        REQUIRE(bsend_init_recv_init_iter_test(std::set<int>{1, 2, 3, 4, 5}));
      }
    }

    WHEN("transmitting data iteratively using ssend_init and recv_init") {
      THEN("data is received across multiple start/wait cycles correctly") {
        REQUIRE(ssend_init_recv_init_test(std::byte(77)));
        REQUIRE(ssend_init_recv_init_test(std::numeric_limits<char>::max() - 1));
        REQUIRE(ssend_init_recv_init_test(std::numeric_limits<signed char>::max() - 1));
        REQUIRE(ssend_init_recv_init_test(std::numeric_limits<unsigned char>::max() - 1));
        REQUIRE(ssend_init_recv_init_test(std::numeric_limits<signed short>::max() - 1));
        REQUIRE(ssend_init_recv_init_test(std::numeric_limits<unsigned short>::max() - 1));
        REQUIRE(ssend_init_recv_init_test(std::numeric_limits<signed int>::max() - 1));
        REQUIRE(ssend_init_recv_init_test(std::numeric_limits<unsigned int>::max() - 1));
        REQUIRE(ssend_init_recv_init_test(std::numeric_limits<signed long>::max() - 1));
        REQUIRE(ssend_init_recv_init_test(std::numeric_limits<unsigned long>::max() - 1));
        REQUIRE(ssend_init_recv_init_test(std::numeric_limits<signed long long>::max() - 1));
        REQUIRE(ssend_init_recv_init_test(std::numeric_limits<unsigned long long>::max() - 1));
        REQUIRE(ssend_init_recv_init_test(static_cast<wchar_t>('A')));
        REQUIRE(ssend_init_recv_init_test(static_cast<char16_t>('A')));
        REQUIRE(ssend_init_recv_init_test(static_cast<char32_t>('A')));
        REQUIRE(ssend_init_recv_init_test(static_cast<float>(3.14)));
        REQUIRE(ssend_init_recv_init_test(static_cast<double>(3.14)));
        REQUIRE(ssend_init_recv_init_test(static_cast<long double>(3.14)));
        REQUIRE(ssend_init_recv_init_test(std::complex<float>(3.14, 2.72)));
        REQUIRE(ssend_init_recv_init_test(std::complex<double>(3.14, 2.72)));
        REQUIRE(ssend_init_recv_init_test(std::complex<long double>(3.14, 2.72)));
        REQUIRE(ssend_init_recv_init_test(true));
        enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };
        REQUIRE(ssend_init_recv_init_test(my_enum::val));
        REQUIRE(ssend_init_recv_init_test(std::pair<int, double>{1, 2.3}));
        REQUIRE(ssend_init_recv_init_test(std::tuple<int, double, bool>{1, 2.3, true}));
        REQUIRE(ssend_init_recv_init_test(std::array<int, 5>{1, 2, 3, 4, 5}));
        REQUIRE(ssend_init_recv_init_iter_test(std::array<int, 5>{1, 2, 3, 4, 5}));
        REQUIRE(ssend_init_recv_init_iter_test(std::vector<int>{1, 2, 3, 4, 5}));
        REQUIRE(ssend_init_recv_init_iter_test(std::list<int>{1, 2, 3, 4, 5}));
        REQUIRE(ssend_init_recv_init_iter_test(std::set<int>{1, 2, 3, 4, 5}));
      }
    }

    WHEN("transmitting data iteratively using rsend_init and recv_init") {
      THEN("data is received across multiple start/wait cycles correctly") {
        REQUIRE(rsend_init_recv_init_test(std::byte(77)));
        REQUIRE(rsend_init_recv_init_test(std::numeric_limits<char>::max() - 1));
        REQUIRE(rsend_init_recv_init_test(std::numeric_limits<signed char>::max() - 1));
        REQUIRE(rsend_init_recv_init_test(std::numeric_limits<unsigned char>::max() - 1));
        REQUIRE(rsend_init_recv_init_test(std::numeric_limits<signed short>::max() - 1));
        REQUIRE(rsend_init_recv_init_test(std::numeric_limits<unsigned short>::max() - 1));
        REQUIRE(rsend_init_recv_init_test(std::numeric_limits<signed int>::max() - 1));
        REQUIRE(rsend_init_recv_init_test(std::numeric_limits<unsigned int>::max() - 1));
        REQUIRE(rsend_init_recv_init_test(std::numeric_limits<signed long>::max() - 1));
        REQUIRE(rsend_init_recv_init_test(std::numeric_limits<unsigned long>::max() - 1));
        REQUIRE(rsend_init_recv_init_test(std::numeric_limits<signed long long>::max() - 1));
        REQUIRE(rsend_init_recv_init_test(std::numeric_limits<unsigned long long>::max() - 1));
        REQUIRE(rsend_init_recv_init_test(static_cast<wchar_t>('A')));
        REQUIRE(rsend_init_recv_init_test(static_cast<char16_t>('A')));
        REQUIRE(rsend_init_recv_init_test(static_cast<char32_t>('A')));
        REQUIRE(rsend_init_recv_init_test(static_cast<float>(3.14)));
        REQUIRE(rsend_init_recv_init_test(static_cast<double>(3.14)));
        REQUIRE(rsend_init_recv_init_test(static_cast<long double>(3.14)));
        REQUIRE(rsend_init_recv_init_test(std::complex<float>(3.14, 2.72)));
        REQUIRE(rsend_init_recv_init_test(std::complex<double>(3.14, 2.72)));
        REQUIRE(rsend_init_recv_init_test(std::complex<long double>(3.14, 2.72)));
        REQUIRE(rsend_init_recv_init_test(true));
        enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };
        REQUIRE(rsend_init_recv_init_test(my_enum::val));
        REQUIRE(rsend_init_recv_init_test(std::pair<int, double>{1, 2.3}));
        REQUIRE(rsend_init_recv_init_test(std::tuple<int, double, bool>{1, 2.3, true}));
        REQUIRE(rsend_init_recv_init_test(std::array<int, 5>{1, 2, 3, 4, 5}));
        REQUIRE(rsend_init_recv_init_iter_test(std::array<int, 5>{1, 2, 3, 4, 5}));
        REQUIRE(rsend_init_recv_init_iter_test(std::vector<int>{1, 2, 3, 4, 5}));
        REQUIRE(rsend_init_recv_init_iter_test(std::list<int>{1, 2, 3, 4, 5}));
        REQUIRE(rsend_init_recv_init_iter_test(std::set<int>{1, 2, 3, 4, 5}));
      }
    }

    WHEN("managing persistent requests in a prequest_pool") {
      THEN("individual persistent start, wait, and test operations work correctly") {
        REQUIRE(prequest_pool_individual_test());
      }
    }
  }
}
