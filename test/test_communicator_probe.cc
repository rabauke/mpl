#include <catch2/catch_test_macros.hpp>
#include <complex>
#include <cstddef>
#include <limits>
#include <list>
#include <set>
#include <string>
#include <type_traits>
#include <tuple>
#include <utility>
#include <vector>
#include <mpl/mpl.hpp>
#include "test_helper.hpp"


static_assert(
    std::is_same_v<decltype(std::declval<mpl::status_t>().get_count<int>()), mpl::ssize_t>);
static_assert(std::is_class_v<mpl::message_t>);


template<typename T>
bool probe_test(const T& data) {
  const mpl::communicator& comm_world = mpl::environment::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0)
    comm_world.send(data, 1);
  if (comm_world.rank() == 1) {
    const mpl::status_t s{comm_world.probe(0)};
    if (s.source() != 0)
      return false;
    if constexpr (has_size<T>::value) {
      if (s.get_count<typename T::value_type>() != static_cast<int>(data.size()))
        return false;
    } else {
      if (s.get_count<T>() != 1)
        return false;
    }
    T data_r{};
    comm_world.recv(data_r, 0);
    return data_r == data;
  }
  return true;
}


template<typename T>
bool probe_iter_test(const T& data) {
  const mpl::communicator& comm_world = mpl::environment::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0)
    comm_world.send(data, 1);
  if (comm_world.rank() == 1) {
    const mpl::status_t s{comm_world.probe(0)};
    if (s.source() != 0)
      return false;
    mpl::ssize_t count{s.get_count<typename T::value_type>()};
    if (count != static_cast<int>(data.size()))
      return false;
    T data_r{};
    if constexpr (has_resize<T>())
      data_r.resize(count);
    comm_world.recv(std::begin(data_r), std::end(data_r), 0);
    return data_r == data;
  }
  return true;
}


template<typename T>
bool iprobe_test(const T& data) {
  const mpl::communicator& comm_world = mpl::environment::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0)
    comm_world.send(data, 1);
  if (comm_world.rank() == 1) {
    while (true) {
      auto s{comm_world.iprobe(0)};
      if (s) {
        if (s->source() != 0)
          return false;
        if constexpr (has_size<T>::value) {
          if (s->template get_count<typename T::value_type>() != static_cast<int>(data.size()))
            return false;
        } else {
          if (s->template get_count<T>() != 1)
            return false;
        }
        T data_r{};
        mpl::irequest request{comm_world.irecv(data_r, 0)};
        request.wait();
        return data_r == data;
      }
    }
  }
  return true;
}


template<typename T>
bool iprobe_iter_test(const T& data) {
  const mpl::communicator& comm_world = mpl::environment::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0)
    comm_world.send(data, 1);
  if (comm_world.rank() == 1) {
    while (true) {
      auto s{comm_world.iprobe(0)};
      if (s) {
        if (s->source() != 0)
          return false;
        mpl::ssize_t count{s->template get_count<typename T::value_type>()};
        if (count != static_cast<int>(data.size()))
          return false;
        T data_r{};
        if constexpr (has_resize<T>())
          data_r.resize(count);
        mpl::irequest request{comm_world.irecv(std::begin(data_r), std::end(data_r), 0)};
        request.wait();
        return data_r == data;
      }
    }
  }
  return true;
}


SCENARIO("layouts represent counts larger than int", "[layout]") {
  (void)mpl::environment::comm_world();

  const size_t count{static_cast<size_t>(std::numeric_limits<int>::max()) + 1};
  const mpl::contiguous_layout<char> contiguous{count};
  REQUIRE(contiguous.byte_extent() == static_cast<ssize_t>(count));

  const mpl::vector_layout<char> vector{count};
  REQUIRE(vector.byte_extent() == static_cast<ssize_t>(count));

  const mpl::strided_vector_layout<char> strided{static_cast<mpl::ssize_t>(count), 1, 2};
  REQUIRE(strided.byte_extent() == static_cast<ssize_t>(2 * count - 1));

  const mpl::hindexed_layout<char> hindexed{{{2, 0}, {3, 5}}};
  REQUIRE(hindexed.byte_extent() == 8);

  const mpl::hindexed_block_layout<char> hindexed_block{2, {0, 4}};
  REQUIRE(hindexed_block.byte_extent() == 6);

  const std::vector<int> iterator_data{1, 2, 3, 4};
  const mpl::iterator_layout<int> iterator{iterator_data.begin(), iterator_data.end()};
  REQUIRE(iterator.byte_extent() ==
          static_cast<mpl::ssize_t>(iterator_data.size() * sizeof(int)));
}


SCENARIO("probing incoming message status and metadata", "[communicator][probe]") {
  const mpl::communicator& comm_world = mpl::environment::comm_world();

  GIVEN("a world communicator with sender rank 0 and receiver rank 1") {
    if (comm_world.size() < 2)
      return;

    WHEN("probing messages synchronously using probe") {
      THEN("message metadata is inspected and payload received properly") {
        REQUIRE(probe_test(std::byte(77)));
        REQUIRE(probe_test(std::numeric_limits<char>::max() - 1));
        REQUIRE(probe_test(std::numeric_limits<signed char>::max() - 1));
        REQUIRE(probe_test(std::numeric_limits<unsigned char>::max() - 1));
        REQUIRE(probe_test(std::numeric_limits<signed short>::max() - 1));
        REQUIRE(probe_test(std::numeric_limits<unsigned short>::max() - 1));
        REQUIRE(probe_test(std::numeric_limits<signed int>::max() - 1));
        REQUIRE(probe_test(std::numeric_limits<unsigned int>::max() - 1));
        REQUIRE(probe_test(std::numeric_limits<signed long>::max() - 1));
        REQUIRE(probe_test(std::numeric_limits<unsigned long>::max() - 1));
        REQUIRE(probe_test(std::numeric_limits<signed long long>::max() - 1));
        REQUIRE(probe_test(std::numeric_limits<unsigned long long>::max() - 1));
        REQUIRE(probe_test(static_cast<wchar_t>('A')));
        REQUIRE(probe_test(static_cast<char16_t>('A')));
        REQUIRE(probe_test(static_cast<char32_t>('A')));
        REQUIRE(probe_test(static_cast<float>(3.14)));
        REQUIRE(probe_test(static_cast<double>(3.14)));
        REQUIRE(probe_test(static_cast<long double>(3.14)));
        REQUIRE(probe_test(std::complex<float>(3.14, 2.72)));
        REQUIRE(probe_test(std::complex<double>(3.14, 2.72)));
        REQUIRE(probe_test(std::complex<long double>(3.14, 2.72)));
        REQUIRE(probe_test(std::pair<int, double>{1, 2.3}));
        REQUIRE(probe_test(std::tuple<int, double, bool>{1, 2.3, true}));
        REQUIRE(probe_test(std::vector<int>{1, 2, 3, 4, 5}));
        REQUIRE(probe_test(std::list<int>{1, 2, 3, 4, 5}));
        REQUIRE(probe_iter_test(std::vector<int>{1, 2, 3, 4, 5}));
        REQUIRE(probe_iter_test(std::list<int>{1, 2, 3, 4, 5}));
      }
    }

    WHEN("probing messages asynchronously using iprobe") {
      THEN("message metadata is polled and payload received properly") {
        REQUIRE(iprobe_test(std::byte(77)));
        REQUIRE(iprobe_test(std::numeric_limits<char>::max() - 1));
        REQUIRE(iprobe_test(std::numeric_limits<signed char>::max() - 1));
        REQUIRE(iprobe_test(std::numeric_limits<unsigned char>::max() - 1));
        REQUIRE(iprobe_test(std::numeric_limits<signed short>::max() - 1));
        REQUIRE(iprobe_test(std::numeric_limits<unsigned short>::max() - 1));
        REQUIRE(iprobe_test(std::numeric_limits<signed int>::max() - 1));
        REQUIRE(iprobe_test(std::numeric_limits<unsigned int>::max() - 1));
        REQUIRE(iprobe_test(std::numeric_limits<signed long>::max() - 1));
        REQUIRE(iprobe_test(std::numeric_limits<unsigned long>::max() - 1));
        REQUIRE(iprobe_test(std::numeric_limits<signed long long>::max() - 1));
        REQUIRE(iprobe_test(std::numeric_limits<unsigned long long>::max() - 1));
        REQUIRE(iprobe_test(static_cast<wchar_t>('A')));
        REQUIRE(iprobe_test(static_cast<char16_t>('A')));
        REQUIRE(iprobe_test(static_cast<char32_t>('A')));
        REQUIRE(iprobe_test(static_cast<float>(3.14)));
        REQUIRE(iprobe_test(static_cast<double>(3.14)));
        REQUIRE(iprobe_test(static_cast<long double>(3.14)));
        REQUIRE(iprobe_test(std::complex<float>(3.14, 2.72)));
        REQUIRE(iprobe_test(std::complex<double>(3.14, 2.72)));
        REQUIRE(iprobe_test(std::complex<long double>(3.14, 2.72)));
        REQUIRE(iprobe_test(std::pair<int, double>{1, 2.3}));
        REQUIRE(iprobe_test(std::tuple<int, double, bool>{1, 2.3, true}));
        REQUIRE(iprobe_test(std::vector<int>{1, 2, 3, 4, 5}));
        REQUIRE(iprobe_test(std::list<int>{1, 2, 3, 4, 5}));
        REQUIRE(iprobe_iter_test(std::vector<int>{1, 2, 3, 4, 5}));
        REQUIRE(iprobe_iter_test(std::list<int>{1, 2, 3, 4, 5}));
      }
    }
  }
}
