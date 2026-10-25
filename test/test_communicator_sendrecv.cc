#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <complex>
#include <cstddef>
#include <limits>
#include <list>
#include <set>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include <mpl/mpl.hpp>
#include "test_helper.hpp"


namespace {

enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };

template<typename T>
bool sendrecv_test(const T &data) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  const int rank{comm_world.rank()};
  const int size{comm_world.size()};
  T data_r;
  comm_world.sendrecv(data, (rank + 1) % size, mpl::tag_t(0), data_r, (rank - 1 + size) % size,
                      mpl::tag_t(0));
  return data_r == data;
}


template<typename T>
bool sendrecv_iter_test(const T &data) {
  static_assert(has_size<T>());
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  if (comm_world.size() < 2)
    return false;
  std::vector<typename T::value_type> data_r(data.size());
  if (comm_world.rank() == 0) {
    comm_world.sendrecv(std::begin(data), std::end(data), 1, mpl::tag_t(4), std::begin(data_r),
                        std::end(data_r), 1, mpl::tag_t(4));
    return std::equal(std::begin(data), std::end(data), std::begin(data_r));
  }
  if (comm_world.rank() == 1) {
    comm_world.sendrecv(std::begin(data), std::end(data), 0, mpl::tag_t(4), std::begin(data_r),
                        std::end(data_r), 0, mpl::tag_t(4));
    return std::equal(std::begin(data), std::end(data), std::begin(data_r));
  }
  return true;
}


template<typename T>
struct data_type_helper;

template<typename T, std::size_t n>
struct data_type_helper<std::array<T, n>> {
  static std::array<T, n> get(int val) {
    std::array<T, n> res;
    std::fill(std::begin(res), std::end(res), val);
    return res;
  }
};


template<typename T1, typename T2>
struct data_type_helper<std::pair<T1, T2>> {
  static std::pair<T1, T2> get(int val) {
    return std::make_pair<T1, T2>(T1(val), T2(val));
  }
};


template<typename... Ts>
struct data_type_helper<std::tuple<Ts...>> {
  static std::tuple<Ts...> get(int val) {
    return std::make_tuple<Ts...>(Ts(val)...);
  }
};


template<typename T>
struct data_type_helper<std::vector<T>> {
  static std::vector<T> get(int val) {
    return {T(val), T(val), T(val), T(val), T(val)};
  }
};


template<typename T>
struct data_type_helper<std::list<T>> {
  static std::list<T> get(int val) {
    return {T(val), T(val), T(val), T(val), T(val)};
  }
};


template<typename T>
bool sendrecv_replace_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  const int rank{comm_world.rank()};
  int size{comm_world.size()};
  T x, expected;
  if constexpr (std::is_constructible_v<T, int> or std::is_enum_v<T>) {
    x = T(rank);
    expected = T((rank + size - 1) % size);
  } else {
    x = data_type_helper<T>::get(rank);
    expected = data_type_helper<T>::get((rank + size - 1) % size);
  }
  comm_world.sendrecv_replace(x, (rank + 1) % size, mpl::tag_t(0), (rank - 1 + size) % size,
                              mpl::tag_t(0));
  return x == expected;
}


template<typename T>
bool sendrecv_replace_iter_test() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  const int rank{comm_world.rank()};
  int size{comm_world.size()};
  T x = data_type_helper<T>::get(rank);
  T expected = data_type_helper<T>::get((rank + size - 1) % size);
  comm_world.sendrecv_replace(std::begin(x), std::end(x), (rank + 1) % size, mpl::tag_t(0),
                              (rank - 1 + size) % size, mpl::tag_t(0));
  return x == expected;
}

}  // namespace


SCENARIO("combined send-and-receive operations exchange data circularly",
         "[communicator][sendrecv]") {
  const mpl::communicator &comm_world = mpl::environment::comm_world();

  GIVEN("a world communicator configured for ring exchange") {
    WHEN("performing separate buffer sendrecv operations") {
      THEN("all data types are shifted across ranks accurately") {
        REQUIRE(sendrecv_test(std::byte(77)));
        REQUIRE(sendrecv_test(std::numeric_limits<char>::max() - 1));
        REQUIRE(sendrecv_test(std::numeric_limits<signed char>::max() - 1));
        REQUIRE(sendrecv_test(std::numeric_limits<unsigned char>::max() - 1));
        REQUIRE(sendrecv_test(std::numeric_limits<signed short>::max() - 1));
        REQUIRE(sendrecv_test(std::numeric_limits<unsigned short>::max() - 1));
        REQUIRE(sendrecv_test(std::numeric_limits<signed int>::max() - 1));
        REQUIRE(sendrecv_test(std::numeric_limits<unsigned int>::max() - 1));
        REQUIRE(sendrecv_test(std::numeric_limits<signed long>::max() - 1));
        REQUIRE(sendrecv_test(std::numeric_limits<unsigned long>::max() - 1));
        REQUIRE(sendrecv_test(std::numeric_limits<signed long long>::max() - 1));
        REQUIRE(sendrecv_test(std::numeric_limits<unsigned long long>::max() - 1));
        REQUIRE(sendrecv_test(static_cast<wchar_t>('A')));
        REQUIRE(sendrecv_test(static_cast<char16_t>('A')));
        REQUIRE(sendrecv_test(static_cast<char32_t>('A')));
        REQUIRE(sendrecv_test(static_cast<float>(3.14)));
        REQUIRE(sendrecv_test(static_cast<double>(3.14)));
        REQUIRE(sendrecv_test(static_cast<long double>(3.14)));
        REQUIRE(sendrecv_test(std::complex<float>(3.14, 2.72)));
        REQUIRE(sendrecv_test(std::complex<double>(3.14, 2.72)));
        REQUIRE(sendrecv_test(std::complex<long double>(3.14, 2.72)));
        REQUIRE(sendrecv_test(true));
        REQUIRE(sendrecv_test(my_enum::val));
        REQUIRE(sendrecv_test(std::pair<int, double>{1, 2.3}));
        REQUIRE(sendrecv_test(std::tuple<int, double, bool>{1, 2.3, true}));
        REQUIRE(sendrecv_test(std::array<int, 5>{1, 2, 3, 4, 5}));
        if (comm_world.size() >= 2) {
          REQUIRE(sendrecv_iter_test(std::array<int, 5>{1, 2, 3, 4, 5}));
          REQUIRE(sendrecv_iter_test(std::vector<int>{1, 2, 3, 4, 5}));
          REQUIRE(sendrecv_iter_test(std::list<int>{1, 2, 3, 4, 5}));
          REQUIRE(sendrecv_iter_test(std::set<int>{1, 2, 3, 4, 5}));
        }
      }
    }

    WHEN("performing in-place sendrecv_replace operations") {
      THEN("all data types are replaced in-place with received neighbor values") {
        REQUIRE(sendrecv_replace_test<std::byte>());
        REQUIRE(sendrecv_replace_test<char>());
        REQUIRE(sendrecv_replace_test<signed char>());
        REQUIRE(sendrecv_replace_test<unsigned char>());
        REQUIRE(sendrecv_replace_test<signed short>());
        REQUIRE(sendrecv_replace_test<unsigned short>());
        REQUIRE(sendrecv_replace_test<signed int>());
        REQUIRE(sendrecv_replace_test<unsigned int>());
        REQUIRE(sendrecv_replace_test<signed long>());
        REQUIRE(sendrecv_replace_test<unsigned long>());
        REQUIRE(sendrecv_replace_test<signed long long>());
        REQUIRE(sendrecv_replace_test<unsigned long long>());
        REQUIRE(sendrecv_replace_test<wchar_t>());
        REQUIRE(sendrecv_replace_test<char16_t>());
        REQUIRE(sendrecv_replace_test<char32_t>());
        REQUIRE(sendrecv_replace_test<float>());
        REQUIRE(sendrecv_replace_test<double>());
        REQUIRE(sendrecv_replace_test<long double>());
        REQUIRE(sendrecv_replace_test<bool>());
        REQUIRE(sendrecv_replace_test<my_enum>());
        using std_pair = std::pair<int, double>;
        REQUIRE(sendrecv_replace_test<std_pair>());
        using std_tuple = std::tuple<int, double, bool>;
        REQUIRE(sendrecv_replace_test<std_tuple>());
        using array = std::array<int, 5>;
        REQUIRE(sendrecv_replace_test<array>());
        REQUIRE(sendrecv_replace_iter_test<array>());
        using vector = std::vector<int>;
        REQUIRE(sendrecv_replace_iter_test<vector>());
        using list = std::list<int>;
        REQUIRE(sendrecv_replace_iter_test<list>());
      }
    }
  }
}
