#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <complex>
#include <cstddef>
#include <limits>
#include <list>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include <mpl/mpl.hpp>


#if __cplusplus >= 202002L
#include <span>

template<typename T>
struct is_span : public std::false_type {};

template<typename T, std::size_t N>
struct is_span<std::span<T, N>> : public std::true_type {};

template<typename T>
inline constexpr bool is_span_v = is_span<T>::value;

template<typename T>
struct span_size;

template<typename T, std::size_t N>
struct span_size<std::span<T, N>> {
  static constexpr std::size_t value = N;
};
#endif


template<typename T>
void send_recv_test(const T &data) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  if (comm_world.size() < 2)
    return;

  MPI_Comm new_comm;
  MPI_Comm_dup(MPI_COMM_WORLD, &new_comm);
  mpl::mpi_communicator mpi_comm{new_comm};

  if (mpi_comm.rank() == 0) {
    mpi_comm.send(data, 1);
  }
  if (mpi_comm.rank() == 1) {
#if __cplusplus >= 202002L
    if constexpr (is_span_v<T>) {
      using element_type = typename T::element_type;
      constexpr auto size = span_size<T>::value;
      std::array<element_type, size> array;
      std::span data_r(array);
      mpi_comm.recv(data_r, 0);
      MPI_Comm_free(&new_comm);
      CHECK(std::equal(data.begin(), data.end(), data_r.begin(), data_r.end()));
      return;
    } else
#endif
    {
      T data_r;
      mpi_comm.recv(data_r, 0);
      MPI_Comm_free(&new_comm);
      CHECK(data_r == data);
      return;
    }
  }
  MPI_Comm_free(&new_comm);
}


SCENARIO("mpi_communicator wrapping raw MPI_Comm handles", "[mpi_communicator]") {
  const mpl::communicator &comm_world = mpl::environment::comm_world();

  GIVEN("a communicator created from a duplicated MPI_Comm handle") {
    REQUIRE(comm_world.size() >= 2);

    WHEN("transmitting integer types") {
      THEN("all integer types are transmitted and received correctly") {
        send_recv_test(std::byte(77));
        send_recv_test(std::numeric_limits<char>::max() - 1);
        send_recv_test(std::numeric_limits<signed char>::max() - 1);
        send_recv_test(std::numeric_limits<unsigned char>::max() - 1);
        send_recv_test(std::numeric_limits<signed short>::max() - 1);
        send_recv_test(std::numeric_limits<unsigned short>::max() - 1);
        send_recv_test(std::numeric_limits<signed int>::max() - 1);
        send_recv_test(std::numeric_limits<unsigned int>::max() - 1);
        send_recv_test(std::numeric_limits<signed long>::max() - 1);
        send_recv_test(std::numeric_limits<unsigned long>::max() - 1);
        send_recv_test(std::numeric_limits<signed long long>::max() - 1);
        send_recv_test(std::numeric_limits<unsigned long long>::max() - 1);
      }
    }

    WHEN("transmitting character types") {
      THEN("all character types are transmitted and received correctly") {
        send_recv_test(static_cast<wchar_t>('A'));
        send_recv_test(static_cast<char16_t>('A'));
        send_recv_test(static_cast<char32_t>('A'));
      }
    }

    WHEN("transmitting floating point and complex types") {
      THEN("all floating point and complex types are transmitted and received correctly") {
        send_recv_test(static_cast<float>(3.14));
        send_recv_test(static_cast<double>(3.14));
        send_recv_test(static_cast<long double>(3.14));
        send_recv_test(std::complex<float>(3.14, 2.72));
        send_recv_test(std::complex<double>(3.14, 2.72));
        send_recv_test(std::complex<long double>(3.14, 2.72));
      }
    }

    WHEN("transmitting logical, enum, tuple, array, and container types") {
      THEN("all data structures are transmitted and received correctly") {
        send_recv_test(true);
        enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };
        send_recv_test(my_enum::val);
        send_recv_test(std::pair<int, double>{1, 2.3});
        send_recv_test(std::tuple<int, double, bool>{1, 2.3, true});
        send_recv_test(std::array<int, 5>{1, 2, 3, 4, 5});
#if __cplusplus >= 202002L
        {
          int array[]{1, 2, 3, 4, 5};
          std::span span{array};
          send_recv_test(span);
        }
#endif
        send_recv_test(std::string{"Hello World"});
        send_recv_test(std::wstring{L"Hello World"});
        send_recv_test(std::vector<int>{1, 2, 3, 4, 5});
        send_recv_test(std::list<int>{1, 2, 3, 4, 5});
        send_recv_test(std::set<int>{1, 2, 3, 4, 5});
      }
    }
  }
}
