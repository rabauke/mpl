#include <catch2/catch_test_macros.hpp>
#include <complex>
#include <cstddef>
#include <limits>
#include <list>
#include <tuple>
#include <utility>
#include <vector>
#include <mpl/mpl.hpp>
#include "test_helper.hpp"


template<typename T>
bool mprobe_test(const T &data) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0)
    comm_world.send(data, 1);
  if (comm_world.rank() == 1) {
    auto [m, s]{comm_world.mprobe(0)};
    if (s.source() != 0)
      return false;
    if constexpr (has_size<T>::value) {
      if (s.get_count<typename T::value_type>() != static_cast<int>(data.size()))
        return false;
    } else {
      if (s.get_count<T>() != 1)
        return false;
      T data_r;
      comm_world.mrecv(data_r, m);
      return data_r == data;
    }
  }
  return true;
}


template<typename T>
bool mprobe_iter_test(const T &data) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0)
    comm_world.send(data, 1);
  if (comm_world.rank() == 1) {
    auto [m, s]{comm_world.mprobe(0)};
    if (s.source() != 0)
      return false;
    int count{s.get_count<typename T::value_type>()};
    if (count != static_cast<int>(data.size()))
      return false;
    T data_r;
    if constexpr (has_resize<T>())
      data_r.resize(count);
    comm_world.mrecv(std::begin(data_r), std::end(data_r), m);
    return data_r == data;
  }
  return true;
}


template<typename T>
bool improbe_test(const T &data) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0)
    comm_world.send(data, 1);
  if (comm_world.rank() == 1) {
    while (true) {
      auto res{comm_world.improbe(0)};
      if (res) {
        auto [m, s]{res.value()};
        if (s.source() != 0)
          return false;
        if constexpr (has_size<T>::value) {
          if (s.get_count<typename T::value_type>() != static_cast<int>(data.size()))
            return false;
        } else {
          if (s.get_count<T>() != 1)
            return false;
        }
        T data_r;
        mpl::irequest request{comm_world.imrecv(data_r, m)};
        request.wait();
        return data_r == data;
      }
    }
  }
  return true;
}


template<typename T>
bool improbe_iter_test(const T &data) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0)
    comm_world.send(data, 1);
  if (comm_world.rank() == 1) {
    while (true) {
      auto res{comm_world.improbe(0)};
      if (res) {
        auto [m, s]{res.value()};
        if (s.source() != 0)
          return false;
        int count{s.get_count<typename T::value_type>()};
        if (count != static_cast<int>(data.size()))
          return false;
        T data_r;
        if constexpr (has_resize<T>())
          data_r.resize(count);
        mpl::irequest request{comm_world.imrecv(std::begin(data_r), std::end(data_r), m)};
        request.wait();
        return data_r == data;
      }
    }
  }
  return true;
}


SCENARIO("matched message probing and receiving", "[communicator][mprobe_mrecv]") {
  const mpl::communicator &comm_world = mpl::environment::comm_world();

  GIVEN("a world communicator with sender rank 0 and receiver rank 1") {
    if (comm_world.size() < 2)
      return;

    WHEN("performing blocking matched probing (mprobe) and receive (mrecv)") {
      THEN("message is matched and payload received accurately") {
        REQUIRE(mprobe_test(std::byte(77)));
        REQUIRE(mprobe_test(std::numeric_limits<char>::max() - 1));
        REQUIRE(mprobe_test(std::numeric_limits<signed char>::max() - 1));
        REQUIRE(mprobe_test(std::numeric_limits<unsigned char>::max() - 1));
        REQUIRE(mprobe_test(std::numeric_limits<signed short>::max() - 1));
        REQUIRE(mprobe_test(std::numeric_limits<unsigned short>::max() - 1));
        REQUIRE(mprobe_test(std::numeric_limits<signed int>::max() - 1));
        REQUIRE(mprobe_test(std::numeric_limits<unsigned int>::max() - 1));
        REQUIRE(mprobe_test(std::numeric_limits<signed long>::max() - 1));
        REQUIRE(mprobe_test(std::numeric_limits<unsigned long>::max() - 1));
        REQUIRE(mprobe_test(std::numeric_limits<signed long long>::max() - 1));
        REQUIRE(mprobe_test(std::numeric_limits<unsigned long long>::max() - 1));
        REQUIRE(mprobe_test(static_cast<wchar_t>('A')));
        REQUIRE(mprobe_test(static_cast<char16_t>('A')));
        REQUIRE(mprobe_test(static_cast<char32_t>('A')));
        REQUIRE(mprobe_test(static_cast<float>(3.14)));
        REQUIRE(mprobe_test(static_cast<double>(3.14)));
        REQUIRE(mprobe_test(static_cast<long double>(3.14)));
        REQUIRE(mprobe_test(std::complex<float>(3.14, 2.72)));
        REQUIRE(mprobe_test(std::complex<double>(3.14, 2.72)));
        REQUIRE(mprobe_test(std::complex<long double>(3.14, 2.72)));
        REQUIRE(mprobe_test(std::pair<int, double>{1, 2.3}));
        REQUIRE(mprobe_test(std::tuple<int, double, bool>{1, 2.3, true}));
        REQUIRE(mprobe_iter_test(std::vector<int>{1, 2, 3, 4, 5}));
        REQUIRE(mprobe_iter_test(std::list<int>{1, 2, 3, 4, 5}));
      }
    }

    WHEN("performing non-blocking matched probing (improbe) and receive (imrecv)") {
      THEN("message is polled, matched, and payload received accurately") {
        REQUIRE(improbe_test(std::byte(77)));
        REQUIRE(improbe_test(std::numeric_limits<char>::max() - 1));
        REQUIRE(improbe_test(std::numeric_limits<signed char>::max() - 1));
        REQUIRE(improbe_test(std::numeric_limits<unsigned char>::max() - 1));
        REQUIRE(improbe_test(std::numeric_limits<signed short>::max() - 1));
        REQUIRE(improbe_test(std::numeric_limits<unsigned short>::max() - 1));
        REQUIRE(improbe_test(std::numeric_limits<signed int>::max() - 1));
        REQUIRE(improbe_test(std::numeric_limits<unsigned int>::max() - 1));
        REQUIRE(improbe_test(std::numeric_limits<signed long>::max() - 1));
        REQUIRE(improbe_test(std::numeric_limits<unsigned long>::max() - 1));
        REQUIRE(improbe_test(std::numeric_limits<signed long long>::max() - 1));
        REQUIRE(improbe_test(std::numeric_limits<unsigned long long>::max() - 1));
        REQUIRE(improbe_test(static_cast<wchar_t>('A')));
        REQUIRE(improbe_test(static_cast<char16_t>('A')));
        REQUIRE(improbe_test(static_cast<char32_t>('A')));
        REQUIRE(improbe_test(static_cast<float>(3.14)));
        REQUIRE(improbe_test(static_cast<double>(3.14)));
        REQUIRE(improbe_test(static_cast<long double>(3.14)));
        REQUIRE(improbe_test(std::complex<float>(3.14, 2.72)));
        REQUIRE(improbe_test(std::complex<double>(3.14, 2.72)));
        REQUIRE(improbe_test(std::complex<long double>(3.14, 2.72)));
        REQUIRE(improbe_test(std::pair<int, double>{1, 2.3}));
        REQUIRE(improbe_test(std::tuple<int, double, bool>{1, 2.3, true}));
        REQUIRE(improbe_iter_test(std::vector<int>{1, 2, 3, 4, 5}));
        REQUIRE(improbe_iter_test(std::list<int>{1, 2, 3, 4, 5}));
      }
    }
  }
}
