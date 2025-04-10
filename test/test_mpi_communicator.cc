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
#include <type_traits>
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
bool send_recv_test(const T &data) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  if (comm_world.size() < 2)
    return false;

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
      return std::equal(data.begin(), data.end(), data_r.begin(), data_r.end());
    } else
#endif
    {
      T data_r;
      mpi_comm.recv(data_r, 0);
      MPI_Comm_free(&new_comm);
      return data_r == data;
    }
  }
  MPI_Comm_free(&new_comm);
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
#if __cplusplus >= 202002L
  {
    int array[]{1, 2, 3, 4, 5};
    std::span span{array};
    BOOST_TEST(send_recv_test(span));
  }
#endif
  // strings and STL containers
  BOOST_TEST(send_recv_test(std::string{"Hello World"}));
  BOOST_TEST(send_recv_test(std::wstring{L"Hello World"}));
  BOOST_TEST(send_recv_test(std::vector<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(send_recv_test(std::list<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(send_recv_test(std::set<int>{1, 2, 3, 4, 5}));
}
