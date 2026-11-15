#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <limits>
#include <list>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include <mpl/mpl.hpp>
#include "test_helper.hpp"


// tests excluded for MPICH and Intel-MPI
// tests fail, most likely due the limitations of current versions of MPICH and Intel-MPI
#if !(defined MPICH_VERSION || defined I_MPI_VERSION)

namespace {

  enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };

  template<typename T>
  bool isendrecv_test(const T &data) {
    const mpl::communicator &comm_world = mpl::environment::comm_world();
    const int rank{comm_world.rank()};
    const int size{comm_world.size()};
    T data_r;
    mpl::irequest r{comm_world.isendrecv(data, (rank + 1) % size, mpl::tag_t(3), data_r,
                                         (rank - 1 + size) % size, mpl::tag_t(3))};
    const mpl::status_t s{r.wait()};
    return data_r == data and s.source() == (rank - 1 + size) % size and
           s.tag() == mpl::tag_t(3);
  }


  template<typename T>
  bool isendrecv_layout_test() {
    const mpl::communicator &comm_world = mpl::environment::comm_world();
    const int rank{comm_world.rank()};
    const int size{comm_world.size()};
    const int n{6};
    std::vector<T> send(n);
    std::vector<T> recv(n, T(-1));
    for (int i{0}; i < n; ++i)
      send[i] = T(rank * 10 + i);
    const mpl::vector_layout<T> l(n);
    mpl::irequest r{comm_world.isendrecv(send.data(), l, (rank + 1) % size, mpl::tag_t(1),
                                         recv.data(), l, (rank - 1 + size) % size,
                                         mpl::tag_t(1))};
    r.wait();
    for (int i{0}; i < n; ++i)
      if (recv[i] != T(((rank - 1 + size) % size) * 10 + i))
        return false;
    return true;
  }


  template<typename C1, typename C2>
  bool isendrecv_iter_test() {
    const mpl::communicator &comm_world = mpl::environment::comm_world();
    const int rank{comm_world.rank()};
    const int size{comm_world.size()};
    const int src{(rank - 1 + size) % size};
    C1 send;
    for (int i{0}; i < 5; ++i)
      send.push_back(rank * 10 + i);
    C2 recv;
    for (int i{0}; i < 5; ++i)
      recv.push_back(-1);
    mpl::irequest r{comm_world.isendrecv(std::begin(send), std::end(send), (rank + 1) % size,
                                         mpl::tag_t(2), std::begin(recv), std::end(recv), src,
                                         mpl::tag_t(2))};
    r.wait();
    int i{0};
    for (const int v : recv)
      if (v != src * 10 + i++)
        return false;
    return true;
  }


  template<typename T>
  struct data_type_helper;

  template<typename T, std::size_t n>
  struct data_type_helper<std::array<T, n>> {
    static std::array<T, n> get(int val) {
      std::array<T, n> res{};
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
  bool isendrecv_replace_test() {
    const mpl::communicator &comm_world = mpl::environment::comm_world();
    const int rank{comm_world.rank()};
    const int size{comm_world.size()};
    T x{};
    T expected{};
    if constexpr (std::is_constructible_v<T, int> or std::is_enum_v<T>) {
      x = T(rank);
      expected = T((rank + size - 1) % size);
    } else {
      x = data_type_helper<T>::get(rank);
      expected = data_type_helper<T>::get((rank + size - 1) % size);
    }
    mpl::irequest r{comm_world.isendrecv_replace(x, (rank + 1) % size, mpl::tag_t(0),
                                                 (rank - 1 + size) % size, mpl::tag_t(0))};
    r.wait();
    return x == expected;
  }


  bool isendrecv_replace_layout_test() {
    const mpl::communicator &comm_world = mpl::environment::comm_world();
    const int rank{comm_world.rank()};
    const int size{comm_world.size()};
    std::vector<int> x(5, rank);
    const mpl::contiguous_layout<int> l(5);
    mpl::irequest r{comm_world.isendrecv_replace(x.data(), l, (rank + 1) % size, mpl::tag_t(0),
                                                 (rank - 1 + size) % size, mpl::tag_t(0))};
    r.wait();
    return x == std::vector<int>(5, (rank + size - 1) % size);
  }


  template<typename T>
  bool isendrecv_replace_iter_test() {
    const mpl::communicator &comm_world = mpl::environment::comm_world();
    const int rank{comm_world.rank()};
    const int size{comm_world.size()};
    T x{data_type_helper<T>::get(rank)};
    T expected{data_type_helper<T>::get((rank + size - 1) % size)};
    mpl::irequest r{comm_world.isendrecv_replace(std::begin(x), std::end(x), (rank + 1) % size,
                                                 mpl::tag_t(0), (rank - 1 + size) % size,
                                                 mpl::tag_t(0))};
    r.wait();
    return x == expected;
  }

}  // namespace


SCENARIO("non-blocking combined send-and-receive operations exchange data circularly",
         "[communicator][isendrecv]") {
  const mpl::communicator &comm_world = mpl::environment::comm_world();

  GIVEN("a world communicator configured for ring exchange") {
    WHEN("performing separate buffer isendrecv operations") {
      THEN("all data types are shifted across ranks accurately") {
        REQUIRE(isendrecv_test(std::byte(77)));
        REQUIRE(isendrecv_test(std::numeric_limits<char>::max() - 1));
        REQUIRE(isendrecv_test(std::numeric_limits<signed char>::max() - 1));
        REQUIRE(isendrecv_test(std::numeric_limits<unsigned char>::max() - 1));
        REQUIRE(isendrecv_test(std::numeric_limits<signed short>::max() - 1));
        REQUIRE(isendrecv_test(std::numeric_limits<unsigned short>::max() - 1));
        REQUIRE(isendrecv_test(std::numeric_limits<signed int>::max() - 1));
        REQUIRE(isendrecv_test(std::numeric_limits<unsigned int>::max() - 1));
        REQUIRE(isendrecv_test(std::numeric_limits<signed long>::max() - 1));
        REQUIRE(isendrecv_test(std::numeric_limits<unsigned long>::max() - 1));
        REQUIRE(isendrecv_test(std::numeric_limits<signed long long>::max() - 1));
        REQUIRE(isendrecv_test(std::numeric_limits<unsigned long long>::max() - 1));
        REQUIRE(isendrecv_test(static_cast<float>(3.14)));
        REQUIRE(isendrecv_test(static_cast<double>(3.14)));
        REQUIRE(isendrecv_test(std::complex<double>(3.14, 2.72)));
        REQUIRE(isendrecv_test(true));
        REQUIRE(isendrecv_test(my_enum::val));
        REQUIRE(isendrecv_test(std::pair<int, double>{1, 2.3}));
        REQUIRE(isendrecv_test(std::tuple<int, double, bool>{1, 2.3, true}));
        REQUIRE(isendrecv_test(std::array<int, 5>{1, 2, 3, 4, 5}));
      }
    }

    WHEN("performing isendrecv operations with layouts and iterators") {
      THEN("data is shifted across ranks accurately") {
        REQUIRE(isendrecv_layout_test<int>());
        REQUIRE(isendrecv_layout_test<double>());
        using vector = std::vector<int>;
        using list = std::list<int>;
        REQUIRE((isendrecv_iter_test<vector, vector>()));
        REQUIRE((isendrecv_iter_test<vector, list>()));
        REQUIRE((isendrecv_iter_test<list, vector>()));
        REQUIRE((isendrecv_iter_test<list, list>()));
      }
    }

    WHEN("performing in-place isendrecv_replace operations") {
      THEN("all data types are replaced in-place with received neighbor values") {
        REQUIRE(isendrecv_replace_test<std::byte>());
        REQUIRE(isendrecv_replace_test<char>());
        REQUIRE(isendrecv_replace_test<signed int>());
        REQUIRE(isendrecv_replace_test<unsigned long long>());
        REQUIRE(isendrecv_replace_test<float>());
        REQUIRE(isendrecv_replace_test<double>());
        REQUIRE(isendrecv_replace_test<bool>());
        REQUIRE(isendrecv_replace_test<my_enum>());
        using std_pair = std::pair<int, double>;
        REQUIRE(isendrecv_replace_test<std_pair>());
        using std_tuple = std::tuple<int, double, bool>;
        REQUIRE(isendrecv_replace_test<std_tuple>());
        using array = std::array<int, 5>;
        REQUIRE(isendrecv_replace_test<array>());
        REQUIRE(isendrecv_replace_layout_test());
        REQUIRE(isendrecv_replace_iter_test<array>());
        REQUIRE(isendrecv_replace_iter_test<std::vector<int>>());
        REQUIRE(isendrecv_replace_iter_test<std::list<int>>());
      }
    }

    WHEN("polling a pending request with test") {
      THEN("the request eventually completes") {
        const int rank{comm_world.rank()};
        const int size{comm_world.size()};
        int send{rank};
        int recv{-1};
        mpl::irequest r{comm_world.isendrecv(send, (rank + 1) % size, mpl::tag_t(5), recv,
                                             (rank - 1 + size) % size, mpl::tag_t(5))};
        while (not r.test()) {
        }
        REQUIRE(recv == (rank - 1 + size) % size);
      }
    }

    WHEN("communicating with mpl::proc_null") {
      THEN("the operations complete immediately and leave the data untouched") {
        int send{1};
        int recv{42};
        mpl::irequest r_1{comm_world.isendrecv(send, mpl::proc_null, mpl::tag_t(0), recv,
                                               mpl::proc_null, mpl::tag_t(0))};
        r_1.wait();
        REQUIRE(recv == 42);
        mpl::irequest r_2{comm_world.isendrecv_replace(recv, mpl::proc_null, mpl::tag_t(0),
                                                       mpl::proc_null, mpl::tag_t(0))};
        r_2.wait();
        REQUIRE(recv == 42);
      }
    }
  }
}

#else

SCENARIO("non-blocking combined send-and-receive operations exchange data circularly skipped",
         "[communicator][isendrecv]") {
}

#endif
