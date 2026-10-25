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
#include "test_helper.hpp"


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
  if (comm_world.rank() == 0)
    comm_world.send(data, 1);
  if (comm_world.rank() == 1) {
#if __cplusplus >= 202002L
    if constexpr (is_span_v<T>) {
      using element_type = typename T::element_type;
      constexpr auto size = span_size<T>::value;
      std::array<element_type, size> array;
      std::span data_r(array);
      comm_world.recv(data_r, 0);
      return std::equal(data.begin(), data.end(), data_r.begin(), data_r.end());
    } else
#endif
    {
      T data_r;
      comm_world.recv(data_r, 0);
      return data_r == data;
    }
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
#if __cplusplus >= 202002L
    if constexpr (is_span_v<T>) {
      using element_type = typename T::element_type;
      constexpr auto size = span_size<T>::value;
      std::array<element_type, size> array;
      std::span data_r(array);
      comm_world.recv(data_r, 0);
      return std::equal(data.begin(), data.end(), data_r.begin(), data_r.end());
    } else
#endif
    {
      T data_r;
      comm_world.recv(data_r, 0);
      return data_r == data;
    }
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
#if __cplusplus >= 202002L
    if constexpr (is_span_v<T>) {
      using element_type = typename T::element_type;
      constexpr auto size = span_size<T>::value;
      std::array<element_type, size> array;
      std::span data_r(array);
      comm_world.recv(data_r, 0);
      return std::equal(data.begin(), data.end(), data_r.begin(), data_r.end());
    } else
#endif
    {
      T data_r;
      comm_world.recv(data_r, 0);
      return data_r == data;
    }
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
      mpl::irequest r{comm_world.irecv(std::begin(data_r), std::end(data_r), 0)};
      comm_world.barrier();
      r.wait();
      return data_r == data;
    } else if constexpr (has_begin_end<T>() and has_size<T>() and not has_resize<T>()) {
      if constexpr (std::is_const_v<std::remove_reference_t<decltype(*std::begin(data))>>) {
        T data_r;
        mpl::irequest r{comm_world.irecv(data_r, 0)};
        comm_world.barrier();
        r.wait();
        return data_r == data;
      } else {
        // T is an std::array
        T data_r;
        mpl::irequest r{comm_world.irecv(std::begin(data_r), std::end(data_r), 0)};
        comm_world.barrier();
        r.wait();
        return data_r == data;
      }
#if __cplusplus >= 202002L
    } else if constexpr (is_span_v<T>) {
      // T is an std::span
      using element_type = typename T::element_type;
      constexpr auto size = span_size<T>::value;
      std::array<element_type, size> array;
      std::span data_r(array);
      mpl::irequest r{comm_world.irecv(data_r, 0)};
      comm_world.barrier();
      r.wait();
      return std::equal(data.begin(), data.end(), data_r.begin(), data_r.end());
#endif
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
      // T is an STL container
      T data_r;
      data_r.resize(data.size());
      mpl::irequest r{comm_world.irecv(std::begin(data_r), std::end(data_r), 0)};
      comm_world.barrier();
      r.wait();
      return data_r == data;
    } else if constexpr (has_begin_end<T>() and has_size<T>() and not has_resize<T>()) {
      if constexpr (std::is_const_v<std::remove_reference_t<decltype(*std::begin(data))>>) {
        T data_r;
        mpl::irequest r{comm_world.irecv(data_r, 0)};
        comm_world.barrier();
        r.wait();
        return data_r == data;
      } else {
        // T is an std::array
        T data_r;
        mpl::irequest r{comm_world.irecv(std::begin(data_r), std::end(data_r), 0)};
        comm_world.barrier();
        r.wait();
        return data_r == data;
      }
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


SCENARIO("blocking point-to-point communication modes", "[communicator][send_recv]") {
  const mpl::communicator &comm_world = mpl::environment::comm_world();

  GIVEN("a world communicator with sender rank 0 and receiver rank 1") {
    REQUIRE(comm_world.size() >= 2);

    WHEN("transmitting data using standard send and recv") {
      THEN("all data types and containers are received identically") {
        CHECK(send_recv_test(std::byte(77)));
        CHECK(send_recv_test(std::numeric_limits<char>::max() - 1));
        CHECK(send_recv_test(std::numeric_limits<signed char>::max() - 1));
        CHECK(send_recv_test(std::numeric_limits<unsigned char>::max() - 1));
        CHECK(send_recv_test(std::numeric_limits<signed short>::max() - 1));
        CHECK(send_recv_test(std::numeric_limits<unsigned short>::max() - 1));
        CHECK(send_recv_test(std::numeric_limits<signed int>::max() - 1));
        CHECK(send_recv_test(std::numeric_limits<unsigned int>::max() - 1));
        CHECK(send_recv_test(std::numeric_limits<signed long>::max() - 1));
        CHECK(send_recv_test(std::numeric_limits<unsigned long>::max() - 1));
        CHECK(send_recv_test(std::numeric_limits<signed long long>::max() - 1));
        CHECK(send_recv_test(std::numeric_limits<unsigned long long>::max() - 1));
        CHECK(send_recv_test(static_cast<wchar_t>('A')));
        CHECK(send_recv_test(static_cast<char16_t>('A')));
        CHECK(send_recv_test(static_cast<char32_t>('A')));
        CHECK(send_recv_test(static_cast<float>(3.14)));
        CHECK(send_recv_test(static_cast<double>(3.14)));
        CHECK(send_recv_test(static_cast<long double>(3.14)));
        CHECK(send_recv_test(std::complex<float>(3.14, 2.72)));
        CHECK(send_recv_test(std::complex<double>(3.14, 2.72)));
        CHECK(send_recv_test(std::complex<long double>(3.14, 2.72)));
        CHECK(send_recv_test(true));
        enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };
        CHECK(send_recv_test(my_enum::val));
        CHECK(send_recv_test(std::pair<int, double>{1, 2.3}));
        CHECK(send_recv_test(std::tuple<int, double, bool>{1, 2.3, true}));
        CHECK(send_recv_test(std::array<int, 5>{1, 2, 3, 4, 5}));
#if __cplusplus >= 202002L
        {
          int array[]{1, 2, 3, 4, 5};
          std::span span{array};
          CHECK(send_recv_test(span));
        }
#endif
        CHECK(send_recv_test(std::string{"Hello World"}));
        CHECK(send_recv_test(std::wstring{L"Hello World"}));
        CHECK(send_recv_test(std::vector<int>{1, 2, 3, 4, 5}));
        CHECK(send_recv_test(std::list<int>{1, 2, 3, 4, 5}));
        CHECK(send_recv_test(std::set<int>{1, 2, 3, 4, 5}));
        CHECK(send_recv_iter_test(std::array<int, 5>{1, 2, 3, 4, 5}));
        CHECK(send_recv_iter_test(std::vector<int>{1, 2, 3, 4, 5}));
        CHECK(send_recv_iter_test(std::list<int>{1, 2, 3, 4, 5}));
        CHECK(send_recv_iter_test(std::set<int>{1, 2, 3, 4, 5}));
      }
    }

    WHEN("transmitting data using buffered bsend and recv") {
      THEN("all data types and containers are received identically") {
        CHECK(bsend_recv_test(std::byte(77)));
        CHECK(bsend_recv_test(std::numeric_limits<char>::max() - 1));
        CHECK(bsend_recv_test(std::numeric_limits<signed char>::max() - 1));
        CHECK(bsend_recv_test(std::numeric_limits<unsigned char>::max() - 1));
        CHECK(bsend_recv_test(std::numeric_limits<signed short>::max() - 1));
        CHECK(bsend_recv_test(std::numeric_limits<unsigned short>::max() - 1));
        CHECK(bsend_recv_test(std::numeric_limits<signed int>::max() - 1));
        CHECK(bsend_recv_test(std::numeric_limits<unsigned int>::max() - 1));
        CHECK(bsend_recv_test(std::numeric_limits<signed long>::max() - 1));
        CHECK(bsend_recv_test(std::numeric_limits<unsigned long>::max() - 1));
        CHECK(bsend_recv_test(std::numeric_limits<signed long long>::max() - 1));
        CHECK(bsend_recv_test(std::numeric_limits<unsigned long long>::max() - 1));
        CHECK(bsend_recv_test(static_cast<wchar_t>('A')));
        CHECK(bsend_recv_test(static_cast<char16_t>('A')));
        CHECK(bsend_recv_test(static_cast<char32_t>('A')));
        CHECK(bsend_recv_test(static_cast<float>(3.14)));
        CHECK(bsend_recv_test(static_cast<double>(3.14)));
        CHECK(bsend_recv_test(static_cast<long double>(3.14)));
        CHECK(bsend_recv_test(std::complex<float>(3.14, 2.72)));
        CHECK(bsend_recv_test(std::complex<double>(3.14, 2.72)));
        CHECK(bsend_recv_test(std::complex<long double>(3.14, 2.72)));
        CHECK(bsend_recv_test(true));
        enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };
        CHECK(bsend_recv_test(my_enum::val));
        CHECK(bsend_recv_test(std::pair<int, double>{1, 2.3}));
        CHECK(bsend_recv_test(std::tuple<int, double, bool>{1, 2.3, true}));
        CHECK(bsend_recv_test(std::array<int, 5>{1, 2, 3, 4, 5}));
#if __cplusplus >= 202002L
        {
          int array[]{1, 2, 3, 4, 5};
          std::span span{array};
          CHECK(bsend_recv_test(span));
        }
#endif
        CHECK(bsend_recv_test(std::string{"Hello World"}));
        CHECK(bsend_recv_test(std::wstring{L"Hello World"}));
        CHECK(bsend_recv_test(std::vector<int>{1, 2, 3, 4, 5}));
        CHECK(bsend_recv_test(std::list<int>{1, 2, 3, 4, 5}));
        CHECK(bsend_recv_test(std::set<int>{1, 2, 3, 4, 5}));
        CHECK(bsend_recv_iter_test(std::array<int, 5>{1, 2, 3, 4, 5}));
        CHECK(bsend_recv_iter_test(std::vector<int>{1, 2, 3, 4, 5}));
        CHECK(bsend_recv_iter_test(std::list<int>{1, 2, 3, 4, 5}));
        CHECK(bsend_recv_iter_test(std::set<int>{1, 2, 3, 4, 5}));
      }
    }

    WHEN("transmitting data using synchronous ssend and recv") {
      THEN("all data types and containers are received identically") {
        CHECK(ssend_recv_test(std::byte(77)));
        CHECK(ssend_recv_test(std::numeric_limits<char>::max() - 1));
        CHECK(ssend_recv_test(std::numeric_limits<signed char>::max() - 1));
        CHECK(ssend_recv_test(std::numeric_limits<unsigned char>::max() - 1));
        CHECK(ssend_recv_test(std::numeric_limits<signed short>::max() - 1));
        CHECK(ssend_recv_test(std::numeric_limits<unsigned short>::max() - 1));
        CHECK(ssend_recv_test(std::numeric_limits<signed int>::max() - 1));
        CHECK(ssend_recv_test(std::numeric_limits<unsigned int>::max() - 1));
        CHECK(ssend_recv_test(std::numeric_limits<signed long>::max() - 1));
        CHECK(ssend_recv_test(std::numeric_limits<unsigned long>::max() - 1));
        CHECK(ssend_recv_test(std::numeric_limits<signed long long>::max() - 1));
        CHECK(ssend_recv_test(std::numeric_limits<unsigned long long>::max() - 1));
        CHECK(ssend_recv_test(static_cast<wchar_t>('A')));
        CHECK(ssend_recv_test(static_cast<char16_t>('A')));
        CHECK(ssend_recv_test(static_cast<char32_t>('A')));
        CHECK(ssend_recv_test(static_cast<float>(3.14)));
        CHECK(ssend_recv_test(static_cast<double>(3.14)));
        CHECK(ssend_recv_test(static_cast<long double>(3.14)));
        CHECK(ssend_recv_test(std::complex<float>(3.14, 2.72)));
        CHECK(ssend_recv_test(std::complex<double>(3.14, 2.72)));
        CHECK(ssend_recv_test(std::complex<long double>(3.14, 2.72)));
        CHECK(ssend_recv_test(true));
        enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };
        CHECK(ssend_recv_test(my_enum::val));
        CHECK(ssend_recv_test(std::pair<int, double>{1, 2.3}));
        CHECK(ssend_recv_test(std::tuple<int, double, bool>{1, 2.3, true}));
        CHECK(ssend_recv_test(std::array<int, 5>{1, 2, 3, 4, 5}));
#if __cplusplus >= 202002L
        {
          int array[]{1, 2, 3, 4, 5};
          std::span span{array};
          CHECK(ssend_recv_test(span));
        }
#endif
        CHECK(ssend_recv_test(std::string{"Hello World"}));
        CHECK(ssend_recv_test(std::wstring{L"Hello World"}));
        CHECK(ssend_recv_test(std::vector<int>{1, 2, 3, 4, 5}));
        CHECK(ssend_recv_test(std::list<int>{1, 2, 3, 4, 5}));
        CHECK(ssend_recv_test(std::set<int>{1, 2, 3, 4, 5}));
        CHECK(ssend_recv_iter_test(std::array<int, 5>{1, 2, 3, 4, 5}));
        CHECK(ssend_recv_iter_test(std::vector<int>{1, 2, 3, 4, 5}));
        CHECK(ssend_recv_iter_test(std::list<int>{1, 2, 3, 4, 5}));
        CHECK(ssend_recv_iter_test(std::set<int>{1, 2, 3, 4, 5}));
      }
    }

    WHEN("transmitting data using ready rsend and recv") {
      THEN("all data types and containers are received identically") {
        CHECK(rsend_recv_test(std::byte(77)));
        CHECK(rsend_recv_test(std::numeric_limits<char>::max() - 1));
        CHECK(rsend_recv_test(std::numeric_limits<signed char>::max() - 1));
        CHECK(rsend_recv_test(std::numeric_limits<unsigned char>::max() - 1));
        CHECK(rsend_recv_test(std::numeric_limits<signed short>::max() - 1));
        CHECK(rsend_recv_test(std::numeric_limits<unsigned short>::max() - 1));
        CHECK(rsend_recv_test(std::numeric_limits<signed int>::max() - 1));
        CHECK(rsend_recv_test(std::numeric_limits<unsigned int>::max() - 1));
        CHECK(rsend_recv_test(std::numeric_limits<signed long>::max() - 1));
        CHECK(rsend_recv_test(std::numeric_limits<unsigned long>::max() - 1));
        CHECK(rsend_recv_test(std::numeric_limits<signed long long>::max() - 1));
        CHECK(rsend_recv_test(std::numeric_limits<unsigned long long>::max() - 1));
        CHECK(rsend_recv_test(static_cast<wchar_t>('A')));
        CHECK(rsend_recv_test(static_cast<char16_t>('A')));
        CHECK(rsend_recv_test(static_cast<char32_t>('A')));
        CHECK(rsend_recv_test(static_cast<float>(3.14)));
        CHECK(rsend_recv_test(static_cast<double>(3.14)));
        CHECK(rsend_recv_test(static_cast<long double>(3.14)));
        CHECK(rsend_recv_test(std::complex<float>(3.14, 2.72)));
        CHECK(rsend_recv_test(std::complex<double>(3.14, 2.72)));
        CHECK(rsend_recv_test(std::complex<long double>(3.14, 2.72)));
        CHECK(rsend_recv_test(true));
        enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };
        CHECK(rsend_recv_test(my_enum::val));
        CHECK(rsend_recv_test(std::pair<int, double>{1, 2.3}));
        CHECK(rsend_recv_test(std::tuple<int, double, bool>{1, 2.3, true}));
        CHECK(rsend_recv_test(std::array<int, 5>{1, 2, 3, 4, 5}));
#if __cplusplus >= 202002L
        {
          int array[]{1, 2, 3, 4, 5};
          std::span span{array};
          CHECK(rsend_recv_test(span));
        }
#endif
        CHECK(rsend_recv_test(std::string{"Hello World"}));
        CHECK(rsend_recv_test(std::wstring{L"Hello World"}));
        CHECK(rsend_recv_test(std::vector<int>{1, 2, 3, 4, 5}));
        CHECK(rsend_recv_test(std::list<int>{1, 2, 3, 4, 5}));
        CHECK(rsend_recv_test(std::set<int>{1, 2, 3, 4, 5}));
        CHECK(rsend_recv_iter_test(std::array<int, 5>{1, 2, 3, 4, 5}));
        CHECK(rsend_recv_iter_test(std::vector<int>{1, 2, 3, 4, 5}));
        CHECK(rsend_recv_iter_test(std::list<int>{1, 2, 3, 4, 5}));
        CHECK(rsend_recv_iter_test(std::set<int>{1, 2, 3, 4, 5}));
      }
    }
  }
}
