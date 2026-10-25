#include <catch2/catch_test_macros.hpp>
#include <numeric>
#include <tuple>
#include <vector>
#include <mpl/mpl.hpp>
#include "test_helper.hpp"


namespace {

template<use_non_root_overload variant, typename T>
bool gatherv_test(const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  const int N{(comm_world.size() * comm_world.size() + comm_world.size()) / 2};
  std::vector<T> v_gather(N);
  std::vector<T> v_send(comm_world.rank() + 1);
  std::vector<T> v_expected(N);
  std::iota(begin(v_expected), end(v_expected), val);
  mpl::layouts<T> layouts;
  for (int i{0}, i_end{comm_world.size()}, offset{0}; i < i_end; ++i) {
    layouts.push_back(mpl::indexed_layout<T>({{i + 1, offset}}));
    offset += i + 1;
  }
  T t_val{val};
  for (int i{0}, i_end{(comm_world.rank() * comm_world.rank() + comm_world.rank()) / 2};
       i < i_end; ++i)
    ++t_val;
  std::iota(begin(v_send), end(v_send), t_val);
  const mpl::vector_layout<T> layout(comm_world.rank() + 1);
  if constexpr (variant == use_non_root_overload::yes) {
    if (comm_world.rank() == 0)
      comm_world.gatherv(0, v_send.data(), layout, v_gather.data(), layouts);
    else
      comm_world.gatherv(0, v_send.data(), layout);
  } else {
    comm_world.gatherv(0, v_send.data(), layout, v_gather.data(), layouts);
  }
  return comm_world.rank() == 0 ? v_gather == v_expected : true;
}


template<use_non_root_overload variant, typename T>
bool gatherv_contiguous_test(const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  const int N{(comm_world.size() * comm_world.size() + comm_world.size()) / 2};
  std::vector<T> v_gather(N);
  std::vector<T> v_send(comm_world.rank() + 1);
  std::vector<T> v_expected(N);
  std::iota(begin(v_expected), end(v_expected), val);
  mpl::contiguous_layouts<T> layouts;
  mpl::displacements displacements;
  for (int i{0}, i_end{comm_world.size()}, offset{0}; i < i_end; ++i) {
    layouts.push_back(mpl::contiguous_layout<T>(i + 1));
    displacements.push_back(sizeof(T) * offset);
    offset += i + 1;
  }
  T t_val{val};
  for (int i{0}, i_end{(comm_world.rank() * comm_world.rank() + comm_world.rank()) / 2};
       i < i_end; ++i)
    ++t_val;
  std::iota(begin(v_send), end(v_send), t_val);
  const mpl::contiguous_layout<T> layout(comm_world.rank() + 1);
  if constexpr (variant == use_non_root_overload::yes) {
    if (comm_world.rank() == 0)
      comm_world.gatherv(0, v_send.data(), layout, v_gather.data(), layouts, displacements);
    else
      comm_world.gatherv(0, v_send.data(), layout);
  } else {
    comm_world.gatherv(0, v_send.data(), layout, v_gather.data(), layouts, displacements);
  }
  return comm_world.rank() == 0 ? v_gather == v_expected : true;
}


template<use_non_root_overload variant, typename T>
bool igatherv_test(const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  const int N{(comm_world.size() * comm_world.size() + comm_world.size()) / 2};
  std::vector<T> v_gather(N);
  std::vector<T> v_send(comm_world.rank() + 1);
  std::vector<T> v_expected(N);
  std::iota(begin(v_expected), end(v_expected), val);
  mpl::layouts<T> layouts;
  for (int i{0}, i_end{comm_world.size()}, offset{0}; i < i_end; ++i) {
    layouts.push_back(mpl::indexed_layout<T>({{i + 1, offset}}));
    offset += i + 1;
  }
  T t_val{val};
  for (int i{0}, i_end{(comm_world.rank() * comm_world.rank() + comm_world.rank()) / 2};
       i < i_end; ++i)
    ++t_val;
  std::iota(begin(v_send), end(v_send), t_val);
  const mpl::vector_layout<T> layout(comm_world.rank() + 1);
  if constexpr (variant == use_non_root_overload::yes) {
    if (comm_world.rank() == 0) {
      auto r{comm_world.igatherv(0, v_send.data(), layout, v_gather.data(), layouts)};
      r.wait();
    } else {
      auto r{comm_world.igatherv(0, v_send.data(), layout)};
      r.wait();
    }
  } else {
    auto r{comm_world.igatherv(0, v_send.data(), layout, v_gather.data(), layouts)};
    r.wait();
  }
  return comm_world.rank() == 0 ? v_gather == v_expected : true;
}


template<use_non_root_overload variant, typename T>
bool igatherv_contiguous_test(const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  const int N{(comm_world.size() * comm_world.size() + comm_world.size()) / 2};
  std::vector<T> v_gather(N);
  std::vector<T> v_send(comm_world.rank() + 1);
  std::vector<T> v_expected(N);
  std::iota(begin(v_expected), end(v_expected), val);
  mpl::contiguous_layouts<T> layouts;
  mpl::displacements displacements;
  for (int i{0}, i_end{comm_world.size()}, offset{0}; i < i_end; ++i) {
    layouts.push_back(mpl::contiguous_layout<T>(i + 1));
    displacements.push_back(sizeof(T) * offset);
    offset += i + 1;
  }
  T t_val{val};
  for (int i{0}, i_end{(comm_world.rank() * comm_world.rank() + comm_world.rank()) / 2};
       i < i_end; ++i)
    ++t_val;
  std::iota(begin(v_send), end(v_send), t_val);
  const mpl::contiguous_layout<T> layout(comm_world.rank() + 1);
  if constexpr (variant == use_non_root_overload::yes) {
    if (comm_world.rank() == 0) {
      auto r{comm_world.igatherv(0, v_send.data(), layout, v_gather.data(), layouts,
                                 displacements)};
      r.wait();
    } else {
      auto r{comm_world.igatherv(0, v_send.data(), layout)};
      r.wait();
    }
  } else {
    auto r{
        comm_world.igatherv(0, v_send.data(), layout, v_gather.data(), layouts, displacements)};
    r.wait();
  }
  return comm_world.rank() == 0 ? v_gather == v_expected : true;
}

}  // namespace


SCENARIO("variable-length data gathering to root process with layout vectors",
         "[communicator][gatherv]") {
  GIVEN("communicator processes sending variable length data blocks") {
    WHEN("gathering variable lengths synchronously using gatherv") {
      THEN("root receives gathered variable data correctly across overload variants") {
        REQUIRE(gatherv_test<use_non_root_overload::no>(1.0));
        REQUIRE(gatherv_test<use_non_root_overload::no>(tuple{1, 2.0}));

        REQUIRE(gatherv_test<use_non_root_overload::yes>(1.0));
        REQUIRE(gatherv_test<use_non_root_overload::yes>(tuple{1, 2.0}));

        REQUIRE(gatherv_contiguous_test<use_non_root_overload::no>(1.0));
        REQUIRE(gatherv_contiguous_test<use_non_root_overload::no>(tuple{1, 2.0}));

        REQUIRE(gatherv_contiguous_test<use_non_root_overload::yes>(1.0));
        REQUIRE(gatherv_contiguous_test<use_non_root_overload::yes>(tuple{1, 2.0}));
      }
    }

    WHEN("gathering variable lengths asynchronously using igatherv") {
      THEN("root receives gathered variable data upon request completion") {
#if !defined MPICH || MPICH_NUMVERSION >= 40101000
        REQUIRE(igatherv_test<use_non_root_overload::no>(1.0));
        REQUIRE(igatherv_test<use_non_root_overload::no>(tuple{1, 2.0}));

        REQUIRE(igatherv_test<use_non_root_overload::yes>(1.0));
        REQUIRE(igatherv_test<use_non_root_overload::yes>(tuple{1, 2.0}));
#endif

        REQUIRE(igatherv_contiguous_test<use_non_root_overload::no>(1.0));
        REQUIRE(igatherv_contiguous_test<use_non_root_overload::no>(tuple{1, 2.0}));

        REQUIRE(igatherv_contiguous_test<use_non_root_overload::yes>(1.0));
        REQUIRE(igatherv_contiguous_test<use_non_root_overload::yes>(tuple{1, 2.0}));
      }
    }
  }
}
