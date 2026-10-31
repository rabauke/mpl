#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <vector>
#include <mpl/mpl.hpp>
#include "test_helper.hpp"


namespace {

bool cartesian_communicator_test() {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  const mpl::cartesian_communicator::dimensions dimensions{mpl::cartesian_communicator::periodic,
                                                     mpl::cartesian_communicator::non_periodic};
  const mpl::cartesian_communicator comm_c{comm_world,
                                     mpl::dims_create(comm_world.size(), dimensions)};
  if (comm_c.dimensionality() != 2)
    return false;
  const int rank{comm_c.rank()};
  auto coordinate{comm_c.coordinates()};
  if (comm_c.rank(coordinate) != rank)
    return false;
  const auto dims{comm_c.get_dimensions()};
  if (dims.size(0) * dims.size(1) != comm_c.size())
    return false;
  if (not(dims.periodicity(0) == mpl::cartesian_communicator::periodic and
          dims.periodicity(1) == mpl::cartesian_communicator::non_periodic))
    return false;
  const auto [source, destination]{comm_c.shift(0, 1)};
  ++coordinate[0];
  if (coordinate[0] >= dims.size(0))
    coordinate[0] = 0;
  const int destination_1{comm_c.rank(coordinate)};
  coordinate[0] -= 2;
  if (coordinate[0] < 0)
    coordinate[0] += dims.size(0);
  const int source_1{comm_c.rank(coordinate)};
  if (not(source == source_1 and destination == destination_1))
    return false;
  {
    const double x{1};
    std::vector<double> y(4, 0.);
    comm_c.neighbor_allgather(x, y.data());
    if ((y[0] != 0 and y[0] != 1) or (y[1] != 0 and y[1] != 1) or (y[2] != 0 and y[2] != 1) or
        (y[3] != 0 and y[3] != 1))
      return false;
  }
  {
    const std::vector<double> x(4, rank + 1.0);
    std::vector<double> y(4, 0.0);
    comm_c.neighbor_alltoall(x.data(), y.data());
    const auto [source, destination]{comm_c.shift(0, 1)};
    const auto ranks_1{comm_c.shift(1, 1)};
    if (source != mpl::proc_null and y[0] != source + 1.)
      return false;
    if (destination != mpl::proc_null and y[1] != destination + 1.)
      return false;
    if (ranks_1.source != mpl::proc_null and y[2] != ranks_1.source + 1.)
      return false;
    if (ranks_1.destination != mpl::proc_null and y[3] != ranks_1.destination + 1.)
      return false;
  }
  {
    const std::vector<double> x(4, rank + 1.0);
    std::vector<double> y(4, 0.0);
    mpl::layouts<double> ls;
    ls.push_back(mpl::indexed_layout<double>({{1, 0}}));
    ls.push_back(mpl::indexed_layout<double>({{1, 1}}));
    ls.push_back(mpl::indexed_layout<double>({{1, 2}}));
    ls.push_back(mpl::indexed_layout<double>({{1, 3}}));
    comm_c.neighbor_alltoallv(x.data(), ls, y.data(), ls);
    const auto ranks_0{comm_c.shift(0, 1)};
    const auto ranks_1{comm_c.shift(1, 1)};
    if (ranks_0.source != mpl::proc_null and y[0] != ranks_0.source + 1.)
      return false;
    if (ranks_0.destination != mpl::proc_null and y[1] != ranks_0.destination + 1.)
      return false;
    if (ranks_1.source != mpl::proc_null and y[2] != ranks_1.source + 1.)
      return false;
    if (ranks_1.destination != mpl::proc_null and y[3] != ranks_1.destination + 1.)
      return false;
  }
  return true;
}


template<typename T>
bool cartesian_communicator_neighbor_alltoall_test(const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  const mpl::cartesian_communicator::dimensions dimensions{mpl::cartesian_communicator::periodic};
  mpl::cartesian_communicator comm_c{comm_world,
                                     mpl::dims_create(comm_world.size(), dimensions)};
  T send_val{val};
  for (int i{0}; i < comm_c.rank(); ++i)
    ++send_val;
  std::vector<T> send_data(2, send_val);
  std::vector<T> recv_data(2);
  T expected_val{val};
  std::vector<T> expected_values;
  for (int i{0}; i < comm_c.size(); ++i) {
    expected_values.push_back(expected_val);
    ++expected_val;
  }
  std::vector<T> expected;
  expected.push_back(expected_values[(comm_c.rank() - 1 + comm_c.size()) % comm_c.size()]);
  expected.push_back(expected_values[(comm_c.rank() + 1) % comm_c.size()]);
  comm_c.neighbor_alltoall(send_data.data(), recv_data.data());
  return recv_data == expected;
}


template<typename T>
bool cartesian_communicator_neighbor_alltoall_layout_test(const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  const mpl::cartesian_communicator::dimensions dimensions{mpl::cartesian_communicator::periodic};
  mpl::cartesian_communicator comm_c{comm_world,
                                     mpl::dims_create(comm_world.size(), dimensions)};
  const int vector_size{3};
  T send_val{val};
  for (int i{0}; i < comm_c.rank(); ++i)
    ++send_val;
  std::vector<T> send_data(2 * vector_size, send_val);
  std::vector<T> recv_data(2 * vector_size);
  T expected_val{val};
  std::vector<T> expected_values;
  for (int i{0}; i < comm_c.size(); ++i) {
    expected_values.push_back(expected_val);
    ++expected_val;
  }
  std::vector<T> expected;
  for (int j{0}; j < vector_size; ++j)
    expected.push_back(expected_values[(comm_c.rank() - 1 + comm_c.size()) % comm_c.size()]);
  for (int j{0}; j < vector_size; ++j)
    expected.push_back(expected_values[(comm_c.rank() + 1) % comm_c.size()]);
  mpl::vector_layout<T> sendrecvl(vector_size);
  comm_c.neighbor_alltoall(send_data.data(), sendrecvl, recv_data.data(), sendrecvl);
  return recv_data == expected;
}


template<typename T>
bool cartesian_communicator_ineighbor_alltoall_test(const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  const mpl::cartesian_communicator::dimensions dimensions{mpl::cartesian_communicator::periodic};
  mpl::cartesian_communicator comm_c{comm_world,
                                     mpl::dims_create(comm_world.size(), dimensions)};
  T send_val{val};
  for (int i{0}; i < comm_c.rank(); ++i)
    ++send_val;
  std::vector<T> send_data(2, send_val);
  std::vector<T> recv_data(2);
  T expected_val{val};
  std::vector<T> expected_values;
  for (int i{0}; i < comm_c.size(); ++i) {
    expected_values.push_back(expected_val);
    ++expected_val;
  }
  std::vector<T> expected;
  expected.push_back(expected_values[(comm_c.rank() - 1 + comm_c.size()) % comm_c.size()]);
  expected.push_back(expected_values[(comm_c.rank() + 1) % comm_c.size()]);
  auto r{comm_c.ineighbor_alltoall(send_data.data(), recv_data.data())};
  r.wait();
  return recv_data == expected;
}


template<typename T>
bool cartesian_communicator_ineighbor_alltoall_layout_test(const T &val) {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  const mpl::cartesian_communicator::dimensions dimensions{mpl::cartesian_communicator::periodic};
  mpl::cartesian_communicator comm_c{comm_world,
                                     mpl::dims_create(comm_world.size(), dimensions)};
  const int vector_size{3};
  T send_val{val};
  for (int i{0}; i < comm_c.rank(); ++i)
    ++send_val;
  std::vector<T> send_data(2 * vector_size, send_val);
  std::vector<T> recv_data(2 * vector_size);
  T expected_val{val};
  std::vector<T> expected_values;
  for (int i{0}; i < comm_c.size(); ++i) {
    expected_values.push_back(expected_val);
    ++expected_val;
  }
  std::vector<T> expected;
  for (int j{0}; j < vector_size; ++j)
    expected.push_back(expected_values[(comm_c.rank() - 1 + comm_c.size()) % comm_c.size()]);
  for (int j{0}; j < vector_size; ++j)
    expected.push_back(expected_values[(comm_c.rank() + 1) % comm_c.size()]);
  mpl::vector_layout<T> sendrecvl(vector_size);
  auto r{comm_c.ineighbor_alltoall(send_data.data(), sendrecvl, recv_data.data(), sendrecvl)};
  r.wait();
  return recv_data == expected;
}

}  // namespace


SCENARIO("cartesian topology communicators manage multi-dimensional grids",
         "[cartesian_communicator]") {
  GIVEN("a 2D cartesian grid with periodic and non-periodic dimensions") {
    WHEN("querying topology dimensions, coordinates, shifts, and neighbor operations") {
      THEN("all topology queries and shifts return expected ranks") {
        REQUIRE(cartesian_communicator_test());
      }
    }
  }

  GIVEN("cartesian communicator helper types") {
    WHEN("manipulating cartesian vector objects") {
      mpl::cartesian_communicator::vector vector{1, 2, 3, 4, 5};
      THEN("vector dimensions and accumulated values match") {
        REQUIRE(vector.dimensions() == 5);
        vector.add(6);
        REQUIRE(vector.dimensions() == 6);
        REQUIRE(std::accumulate(vector.begin(), vector.end(), 0) == 21);
        REQUIRE(std::accumulate(vector.cbegin(), vector.cend(), 0) == 21);
      }
    }

    WHEN("manipulating included_tags collections") {
      const auto included = mpl::cartesian_communicator::included;
      const auto excluded = mpl::cartesian_communicator::excluded;
      mpl::cartesian_communicator::included_tags is_included{10};
      THEN("tags can be added and queried") {
        REQUIRE(is_included.size() == 10);
        is_included.add(included);
        is_included.add(excluded);
        REQUIRE(is_included.size() == 12);
        REQUIRE(std::find(is_included.begin(), is_included.end(), included) != is_included.end());
      }
    }

    WHEN("manipulating dimensions objects") {
      mpl::cartesian_communicator::dimensions dimensions{mpl::cartesian_communicator::periodic,
                                                         mpl::cartesian_communicator::non_periodic,
                                                         mpl::cartesian_communicator::non_periodic};
      THEN("periodicities and sizes are correctly stored and modified") {
        REQUIRE(dimensions.dimensionality() == 3);
        REQUIRE(dimensions.periodicity(0) == mpl::cartesian_communicator::periodic);
        REQUIRE(dimensions.periodicity(1) == mpl::cartesian_communicator::non_periodic);
        REQUIRE(dimensions.periodicity(2) == mpl::cartesian_communicator::non_periodic);
        dimensions[1] = {10, mpl::cartesian_communicator::periodic};
        REQUIRE(dimensions.periodicity(1) == mpl::cartesian_communicator::periodic);
        REQUIRE(dimensions.size(1) == 10);
        dimensions.add(11, mpl::cartesian_communicator::non_periodic);
        REQUIRE(dimensions.periodicity(3) == mpl::cartesian_communicator::non_periodic);
        REQUIRE(dimensions.size(3) == 11);
        REQUIRE(std::find(dimensions.begin(), dimensions.end(),
                          std::tuple{11, mpl::cartesian_communicator::non_periodic}) !=
                dimensions.end());
      }
    }
  }

  GIVEN("a 1D periodic cartesian grid") {
    WHEN("performing neighbor all-to-all exchanges with fundamental and compound types") {
      THEN("neighbor data is exchanged correctly across blocking and non-blocking calls") {
        REQUIRE(cartesian_communicator_neighbor_alltoall_test(1.0));
        REQUIRE(cartesian_communicator_neighbor_alltoall_test(tuple{1, 2.0}));

        REQUIRE(cartesian_communicator_neighbor_alltoall_layout_test(1.0));
        REQUIRE(cartesian_communicator_neighbor_alltoall_layout_test(tuple{1, 2.0}));

        REQUIRE(cartesian_communicator_ineighbor_alltoall_test(1.0));
        REQUIRE(cartesian_communicator_ineighbor_alltoall_test(tuple{1, 2.0}));

        REQUIRE(cartesian_communicator_ineighbor_alltoall_layout_test(1.0));
        REQUIRE(cartesian_communicator_ineighbor_alltoall_layout_test(tuple{1, 2.0}));
      }
    }
  }
}
