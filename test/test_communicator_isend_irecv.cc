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


struct large_count_test_container {
  using value_type = char;

  char value{};

  [[nodiscard]] std::size_t size() const {
    return static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1;
  }

  const char &operator[](std::size_t) const {
    return value;
  }
};

namespace mpl::detail {

  template<>
  class datatype_traits<large_count_test_container> {
  public:
    using data_type_category = contiguous_stl_container;
  };

}  // namespace mpl::detail


template<typename T>
bool isend_irecv_test(const T &data) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    auto r{comm_world.isend(data, 1)};
    r.wait();
  }
  if (comm_world.rank() == 1) {
    T data_r;
    auto r{comm_world.irecv(data_r, 0)};
    while (not r.test()) {
    }
    return data_r == data;
  }
  return true;
}


TEST_CASE("nonblocking send handles counts greater than int", "[communicator][isend_irecv]") {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  const large_count_test_container data{};
  REQUIRE(data.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()));

  mpl::irequest request{comm_world.isend(data, mpl::proc_null)};
  const mpl::status_t status{request.wait()};
  REQUIRE(status.get_count<char>() == 0);
}


template<typename T>
bool isend_irecv_iter_test(const T &data) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    auto r{comm_world.isend(std::begin(data), std::end(data), 1)};
    r.wait();
  }
  if (comm_world.rank() == 1) {
    T data_r;
    if constexpr (std::is_const_v<std::remove_reference_t<decltype(*std::begin(data_r))>>) {
      auto r{comm_world.irecv(data_r, 0)};
      while (not r.test()) {
      }
      return data_r == data;
    } else {
      if constexpr (has_resize<T>())
        data_r.resize(data.size());
      auto r{comm_world.irecv(std::begin(data_r), std::end(data_r), 0)};
      while (not r.test()) {
      }
      return data_r == data;
    }
  }
  return true;
}


template<typename T>
bool ibsend_irecv_test(const T &data) {
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
    auto r{comm_world.ibsend(data, 1)};
    r.wait();
  }
  if (comm_world.rank() == 1) {
    T data_r;
    auto r{comm_world.irecv(data_r, 0)};
    while (not r.test()) {
    }
    return data_r == data;
  }
  return true;
}


template<typename T>
bool ibsend_irecv_iter_test(const T &data) {
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
    auto r{comm_world.ibsend(std::begin(data), std::end(data), 1)};
    r.wait();
  }
  if (comm_world.rank() == 1) {
    T data_r;
    if constexpr (std::is_const_v<std::remove_reference_t<decltype(*std::begin(data_r))>>) {
      auto r{comm_world.irecv(data_r, 0)};
      while (not r.test()) {
      }
      return data_r == data;
    } else {
      if constexpr (has_resize<T>())
        data_r.resize(data.size());
      auto r{comm_world.irecv(std::begin(data_r), std::end(data_r), 0)};
      while (not r.test()) {
      }
      return data_r == data;
    }
  }
  return true;
}


template<typename T>
bool issend_irecv_test(const T &data) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    auto r{comm_world.issend(data, 1)};
    r.wait();
  }
  if (comm_world.rank() == 1) {
    T data_r;
    auto r{comm_world.irecv(data_r, 0)};
    while (not r.test()) {
    }
    return data_r == data;
  }
  return true;
}


template<typename T>
bool issend_irecv_iter_test(const T &data) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    auto r{comm_world.issend(std::begin(data), std::end(data), 1)};
    r.wait();
  }
  if (comm_world.rank() == 1) {
    T data_r;
    if constexpr (std::is_const_v<std::remove_reference_t<decltype(*std::begin(data_r))>>) {
      auto r{comm_world.irecv(data_r, 0)};
      while (not r.test()) {
      }
      return data_r == data;
    } else {
      if constexpr (has_resize<T>())
        data_r.resize(data.size());
      auto r{comm_world.irecv(std::begin(data_r), std::end(data_r), 0)};
      while (not r.test()) {
      }
      return data_r == data;
    }
  }
  return true;
}


template<typename T>
bool irsend_irecv_test(const T &data) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    comm_world.barrier();
    auto r{comm_world.irsend(data, 1)};
    r.wait();
  } else if (comm_world.rank() == 1) {
    // must ensure that MPI_Irecv is called before mpl::communicator::irsend
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


template<typename T>
bool irsend_irecv_iter_test(const T &data) {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    comm_world.barrier();
    auto r{comm_world.irsend(std::begin(data), std::end(data), 1)};
    r.wait();
  } else if (comm_world.rank() == 1) {
    // must ensure that MPI_Irecv is called before mpl::communicator::irsend
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


bool irequest_pool_wait_test() {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    int val1{42};
    int val2{84};
    int val3{126};
    mpl::irequest_pool pool;
    pool.push(comm_world.isend(val1, 1, mpl::tag_t(101)));
    pool.push(comm_world.isend(val2, 1, mpl::tag_t(102)));
    pool.push(comm_world.isend(val3, 1, mpl::tag_t(103)));
    pool.wait(2);
    pool.wait(0);
    pool.wait(1);
    return true;
  }
  if (comm_world.rank() == 1) {
    int val1{0};
    int val2{0};
    int val3{0};
    mpl::irequest_pool pool;
    pool.push(comm_world.irecv(val1, 0, mpl::tag_t(101)));
    pool.push(comm_world.irecv(val2, 0, mpl::tag_t(102)));
    pool.push(comm_world.irecv(val3, 0, mpl::tag_t(103)));
    auto status2{pool.wait(2)};
    auto status0{pool.wait(0)};
    auto status1{pool.wait(1)};
    bool ok{(val1 == 42 and val2 == 84 and val3 == 126)};
    ok = ok and
         (status0.tag() == mpl::tag_t(101) and pool.get_status(0).tag() == mpl::tag_t(101));
    ok = ok and
         (status1.tag() == mpl::tag_t(102) and pool.get_status(1).tag() == mpl::tag_t(102));
    ok = ok and
         (status2.tag() == mpl::tag_t(103) and pool.get_status(2).tag() == mpl::tag_t(103));
    ok = ok and (status0.source() == 0 and status1.source() == 0 and status2.source() == 0);
    return ok;
  }
  return true;
}


bool irequest_pool_test_test() {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    int val1{11};
    int val2{22};
    mpl::irequest_pool pool;
    pool.push(comm_world.isend(val1, 1, mpl::tag_t(201)));
    pool.push(comm_world.isend(val2, 1, mpl::tag_t(202)));
    while (not pool.test(1)) {
    }
    while (not pool.test(0)) {
    }
    return true;
  }
  if (comm_world.rank() == 1) {
    int val1{0};
    int val2{0};
    mpl::irequest_pool pool;
    pool.push(comm_world.irecv(val1, 0, mpl::tag_t(201)));
    pool.push(comm_world.irecv(val2, 0, mpl::tag_t(202)));
    while (not pool.test(1)) {
    }
    while (not pool.test(0)) {
    }
    bool ok{(val1 == 11 and val2 == 22)};
    ok = ok and (pool.get_status(0).tag() == mpl::tag_t(201));
    ok = ok and (pool.get_status(1).tag() == mpl::tag_t(202));
    return ok;
  }
  return true;
}


SCENARIO("non-blocking point-to-point communication and request pools",
         "[communicator][isend_irecv]") {
  const mpl::communicator &comm_world = mpl::environment::comm_world();

  GIVEN("a world communicator with sender rank 0 and receiver rank 1") {
    if (comm_world.size() < 2)
      return;

    WHEN("transmitting data using non-blocking isend and irecv") {
      THEN("all data types are received correctly") {
        REQUIRE(isend_irecv_test(std::byte(77)));
        REQUIRE(isend_irecv_test(std::numeric_limits<char>::max() - 1));
        REQUIRE(isend_irecv_test(std::numeric_limits<signed char>::max() - 1));
        REQUIRE(isend_irecv_test(std::numeric_limits<unsigned char>::max() - 1));
        REQUIRE(isend_irecv_test(std::numeric_limits<signed short>::max() - 1));
        REQUIRE(isend_irecv_test(std::numeric_limits<unsigned short>::max() - 1));
        REQUIRE(isend_irecv_test(std::numeric_limits<signed int>::max() - 1));
        REQUIRE(isend_irecv_test(std::numeric_limits<unsigned int>::max() - 1));
        REQUIRE(isend_irecv_test(std::numeric_limits<signed long>::max() - 1));
        REQUIRE(isend_irecv_test(std::numeric_limits<unsigned long>::max() - 1));
        REQUIRE(isend_irecv_test(std::numeric_limits<signed long long>::max() - 1));
        REQUIRE(isend_irecv_test(std::numeric_limits<unsigned long long>::max() - 1));
        REQUIRE(isend_irecv_test(static_cast<wchar_t>('A')));
        REQUIRE(isend_irecv_test(static_cast<char16_t>('A')));
        REQUIRE(isend_irecv_test(static_cast<char32_t>('A')));
        REQUIRE(isend_irecv_test(static_cast<float>(3.14)));
        REQUIRE(isend_irecv_test(static_cast<double>(3.14)));
        REQUIRE(isend_irecv_test(static_cast<long double>(3.14)));
        REQUIRE(isend_irecv_test(std::complex<float>(3.14, 2.72)));
        REQUIRE(isend_irecv_test(std::complex<double>(3.14, 2.72)));
        REQUIRE(isend_irecv_test(std::complex<long double>(3.14, 2.72)));
        REQUIRE(isend_irecv_test(true));
        enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };
        REQUIRE(isend_irecv_test(my_enum::val));
        REQUIRE(isend_irecv_test(std::pair<int, double>{1, 2.3}));
        REQUIRE(isend_irecv_test(std::tuple<int, double, bool>{1, 2.3, true}));
        REQUIRE(isend_irecv_test(std::array<int, 5>{1, 2, 3, 4, 5}));
        REQUIRE(isend_irecv_test(std::string{"Hello World"}));
        REQUIRE(isend_irecv_test(std::wstring{L"Hello World"}));
        REQUIRE(isend_irecv_test(std::vector<int>{1, 2, 3, 4, 5}));
        REQUIRE(isend_irecv_test(std::list<int>{1, 2, 3, 4, 5}));
        REQUIRE(isend_irecv_test(std::set<int>{1, 2, 3, 4, 5}));
        REQUIRE(isend_irecv_iter_test(std::array<int, 5>{1, 2, 3, 4, 5}));
        REQUIRE(isend_irecv_iter_test(std::vector<int>{1, 2, 3, 4, 5}));
        REQUIRE(isend_irecv_iter_test(std::list<int>{1, 2, 3, 4, 5}));
        REQUIRE(isend_irecv_iter_test(std::set<int>{1, 2, 3, 4, 5}));
      }
    }

    WHEN("transmitting data using non-blocking buffered ibsend and irecv") {
      THEN("all data types are received correctly") {
        REQUIRE(ibsend_irecv_test(std::byte(77)));
        REQUIRE(ibsend_irecv_test(std::numeric_limits<char>::max() - 1));
        REQUIRE(ibsend_irecv_test(std::numeric_limits<signed char>::max() - 1));
        REQUIRE(ibsend_irecv_test(std::numeric_limits<unsigned char>::max() - 1));
        REQUIRE(ibsend_irecv_test(std::numeric_limits<signed short>::max() - 1));
        REQUIRE(ibsend_irecv_test(std::numeric_limits<unsigned short>::max() - 1));
        REQUIRE(ibsend_irecv_test(std::numeric_limits<signed int>::max() - 1));
        REQUIRE(ibsend_irecv_test(std::numeric_limits<unsigned int>::max() - 1));
        REQUIRE(ibsend_irecv_test(std::numeric_limits<signed long>::max() - 1));
        REQUIRE(ibsend_irecv_test(std::numeric_limits<unsigned long>::max() - 1));
        REQUIRE(ibsend_irecv_test(std::numeric_limits<signed long long>::max() - 1));
        REQUIRE(ibsend_irecv_test(std::numeric_limits<unsigned long long>::max() - 1));
        REQUIRE(ibsend_irecv_test(static_cast<wchar_t>('A')));
        REQUIRE(ibsend_irecv_test(static_cast<char16_t>('A')));
        REQUIRE(ibsend_irecv_test(static_cast<char32_t>('A')));
        REQUIRE(ibsend_irecv_test(static_cast<float>(3.14)));
        REQUIRE(ibsend_irecv_test(static_cast<double>(3.14)));
        REQUIRE(ibsend_irecv_test(static_cast<long double>(3.14)));
        REQUIRE(ibsend_irecv_test(std::complex<float>(3.14, 2.72)));
        REQUIRE(ibsend_irecv_test(std::complex<double>(3.14, 2.72)));
        REQUIRE(ibsend_irecv_test(std::complex<long double>(3.14, 2.72)));
        REQUIRE(ibsend_irecv_test(true));
        enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };
        REQUIRE(ibsend_irecv_test(my_enum::val));
        REQUIRE(ibsend_irecv_test(std::pair<int, double>{1, 2.3}));
        REQUIRE(ibsend_irecv_test(std::tuple<int, double, bool>{1, 2.3, true}));
        REQUIRE(ibsend_irecv_test(std::array<int, 5>{1, 2, 3, 4, 5}));
        REQUIRE(ibsend_irecv_test(std::string{"Hello World"}));
        REQUIRE(ibsend_irecv_test(std::wstring{L"Hello World"}));
        REQUIRE(ibsend_irecv_test(std::vector<int>{1, 2, 3, 4, 5}));
        REQUIRE(ibsend_irecv_test(std::list<int>{1, 2, 3, 4, 5}));
        REQUIRE(ibsend_irecv_test(std::set<int>{1, 2, 3, 4, 5}));
        REQUIRE(ibsend_irecv_iter_test(std::array<int, 5>{1, 2, 3, 4, 5}));
        REQUIRE(ibsend_irecv_iter_test(std::vector<int>{1, 2, 3, 4, 5}));
        REQUIRE(ibsend_irecv_iter_test(std::list<int>{1, 2, 3, 4, 5}));
        REQUIRE(ibsend_irecv_iter_test(std::set<int>{1, 2, 3, 4, 5}));
      }
    }

    WHEN("transmitting data using non-blocking synchronous issend and irecv") {
      THEN("all data types are received correctly") {
        REQUIRE(issend_irecv_test(std::byte(77)));
        REQUIRE(issend_irecv_test(std::numeric_limits<char>::max() - 1));
        REQUIRE(issend_irecv_test(std::numeric_limits<signed char>::max() - 1));
        REQUIRE(issend_irecv_test(std::numeric_limits<unsigned char>::max() - 1));
        REQUIRE(issend_irecv_test(std::numeric_limits<signed short>::max() - 1));
        REQUIRE(issend_irecv_test(std::numeric_limits<unsigned short>::max() - 1));
        REQUIRE(issend_irecv_test(std::numeric_limits<signed int>::max() - 1));
        REQUIRE(issend_irecv_test(std::numeric_limits<unsigned int>::max() - 1));
        REQUIRE(issend_irecv_test(std::numeric_limits<signed long>::max() - 1));
        REQUIRE(issend_irecv_test(std::numeric_limits<unsigned long>::max() - 1));
        REQUIRE(issend_irecv_test(std::numeric_limits<signed long long>::max() - 1));
        REQUIRE(issend_irecv_test(std::numeric_limits<unsigned long long>::max() - 1));
        REQUIRE(issend_irecv_test(static_cast<wchar_t>('A')));
        REQUIRE(issend_irecv_test(static_cast<char16_t>('A')));
        REQUIRE(issend_irecv_test(static_cast<char32_t>('A')));
        REQUIRE(issend_irecv_test(static_cast<float>(3.14)));
        REQUIRE(issend_irecv_test(static_cast<double>(3.14)));
        REQUIRE(issend_irecv_test(static_cast<long double>(3.14)));
        REQUIRE(issend_irecv_test(std::complex<float>(3.14, 2.72)));
        REQUIRE(issend_irecv_test(std::complex<double>(3.14, 2.72)));
        REQUIRE(issend_irecv_test(std::complex<long double>(3.14, 2.72)));
        REQUIRE(issend_irecv_test(true));
        enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };
        REQUIRE(issend_irecv_test(my_enum::val));
        REQUIRE(issend_irecv_test(std::pair<int, double>{1, 2.3}));
        REQUIRE(issend_irecv_test(std::tuple<int, double, bool>{1, 2.3, true}));
        REQUIRE(issend_irecv_test(std::array<int, 5>{1, 2, 3, 4, 5}));
        REQUIRE(issend_irecv_test(std::string{"Hello World"}));
        REQUIRE(issend_irecv_test(std::wstring{L"Hello World"}));
        REQUIRE(issend_irecv_test(std::vector<int>{1, 2, 3, 4, 5}));
        REQUIRE(issend_irecv_test(std::list<int>{1, 2, 3, 4, 5}));
        REQUIRE(issend_irecv_test(std::set<int>{1, 2, 3, 4, 5}));
        REQUIRE(issend_irecv_iter_test(std::array<int, 5>{1, 2, 3, 4, 5}));
        REQUIRE(issend_irecv_iter_test(std::vector<int>{1, 2, 3, 4, 5}));
        REQUIRE(issend_irecv_iter_test(std::list<int>{1, 2, 3, 4, 5}));
        REQUIRE(issend_irecv_iter_test(std::set<int>{1, 2, 3, 4, 5}));
      }
    }

    WHEN("transmitting data using non-blocking ready irsend and irecv") {
      THEN("all data types are received correctly") {
        REQUIRE(irsend_irecv_test(std::byte(77)));
        REQUIRE(irsend_irecv_test(std::numeric_limits<char>::max() - 1));
        REQUIRE(irsend_irecv_test(std::numeric_limits<signed char>::max() - 1));
        REQUIRE(irsend_irecv_test(std::numeric_limits<unsigned char>::max() - 1));
        REQUIRE(irsend_irecv_test(std::numeric_limits<signed short>::max() - 1));
        REQUIRE(irsend_irecv_test(std::numeric_limits<unsigned short>::max() - 1));
        REQUIRE(irsend_irecv_test(std::numeric_limits<signed int>::max() - 1));
        REQUIRE(irsend_irecv_test(std::numeric_limits<unsigned int>::max() - 1));
        REQUIRE(irsend_irecv_test(std::numeric_limits<signed long>::max() - 1));
        REQUIRE(irsend_irecv_test(std::numeric_limits<unsigned long>::max() - 1));
        REQUIRE(irsend_irecv_test(std::numeric_limits<signed long long>::max() - 1));
        REQUIRE(irsend_irecv_test(std::numeric_limits<unsigned long long>::max() - 1));
        REQUIRE(irsend_irecv_test(static_cast<wchar_t>('A')));
        REQUIRE(irsend_irecv_test(static_cast<char16_t>('A')));
        REQUIRE(irsend_irecv_test(static_cast<char32_t>('A')));
        REQUIRE(irsend_irecv_test(static_cast<float>(3.14)));
        REQUIRE(irsend_irecv_test(static_cast<double>(3.14)));
        REQUIRE(irsend_irecv_test(static_cast<long double>(3.14)));
        REQUIRE(irsend_irecv_test(std::complex<float>(3.14, 2.72)));
        REQUIRE(irsend_irecv_test(std::complex<double>(3.14, 2.72)));
        REQUIRE(irsend_irecv_test(std::complex<long double>(3.14, 2.72)));
        REQUIRE(irsend_irecv_test(true));
        enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };
        REQUIRE(irsend_irecv_test(my_enum::val));
        REQUIRE(irsend_irecv_test(std::pair<int, double>{1, 2.3}));
        REQUIRE(irsend_irecv_test(std::tuple<int, double, bool>{1, 2.3, true}));
        REQUIRE(irsend_irecv_test(std::array<int, 5>{1, 2, 3, 4, 5}));
        REQUIRE(irsend_irecv_test(std::string{"Hello World"}));
        REQUIRE(irsend_irecv_test(std::wstring{L"Hello World"}));
        REQUIRE(irsend_irecv_test(std::vector<int>{1, 2, 3, 4, 5}));
        REQUIRE(irsend_irecv_test(std::list<int>{1, 2, 3, 4, 5}));
        REQUIRE(irsend_irecv_test(std::set<int>{1, 2, 3, 4, 5}));
        REQUIRE(irsend_irecv_iter_test(std::array<int, 5>{1, 2, 3, 4, 5}));
        REQUIRE(irsend_irecv_iter_test(std::vector<int>{1, 2, 3, 4, 5}));
        REQUIRE(irsend_irecv_iter_test(std::list<int>{1, 2, 3, 4, 5}));
        REQUIRE(irsend_irecv_iter_test(std::set<int>{1, 2, 3, 4, 5}));
      }
    }

    WHEN("managing multiple requests in an irequest_pool") {
      THEN("individual wait and test operations complete successfully") {
        REQUIRE(irequest_pool_wait_test());
        REQUIRE(irequest_pool_test_test());
      }
    }
  }
}
