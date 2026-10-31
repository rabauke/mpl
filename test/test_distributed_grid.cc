// Tests for mpl::distributed_grid and mpl::local_grid.
// Designed for 2 to 4 processes. The global sizes 11 x 13 are not divisible by the process
// counts 2, 3 or 4, so the decomposition is uneven. The grid is non-periodic.
// The halo exchange only transfers face overlap regions (no diagonal neighbors), so corner
// overlap regions are not checked.
#include <catch2/catch_test_macros.hpp>
#include <mpl/mpl.hpp>


namespace {

  constexpr int nx{11};
  constexpr int ny{13};
  constexpr int overlap_x{2};
  constexpr int overlap_y{1};
  constexpr int sentinel{-1};

  int global_value(std::ptrdiff_t i, std::ptrdiff_t j) {
    return static_cast<int>(100 * i + j);
  }


  template<std::size_t dim, typename T, typename A>
  void update_overlap(const mpl::cartesian_communicator& comm_c,
                      mpl::distributed_grid<dim, T, A>& grid) {
    for (std::size_t i{0}; i < dim; ++i) {
      // send to left
      auto [source_l, destination_l] = comm_c.shift(i, -1);
      comm_c.sendrecv(grid.data(), grid.left_border_layout(i), destination_l, mpl::tag_t(),
                      grid.data(), grid.right_mirror_layout(i), source_l, mpl::tag_t());
      // send to right
      auto [source_r, destination_r] = comm_c.shift(i, +1);
      comm_c.sendrecv(grid.data(), grid.right_border_layout(i), destination_r, mpl::tag_t(),
                      grid.data(), grid.left_mirror_layout(i), source_r, mpl::tag_t());
    }
  }

}  // namespace


SCENARIO("distributed_grid decomposes a 2D grid over the processes",
         "[distributed_grid][decomposition]") {
  GIVEN(
      "a 2D Cartesian communicator and a grid with sizes not divisible by the process "
      "counts") {
    const mpl::communicator& comm_world{mpl::environment::comm_world()};
    mpl::cartesian_communicator::dimensions dimensions{
        mpl::cartesian_communicator::non_periodic, mpl::cartesian_communicator::non_periodic};
    mpl::cartesian_communicator comm_c{comm_world,
                                       mpl::dims_create(comm_world.size(), dimensions)};
    mpl::distributed_grid<2, int> grid{comm_c, {{nx, overlap_x}, {ny, overlap_y}}};
    const auto coordinates{comm_c.coordinates()};
    const auto c_dimensions{comm_c.get_dimensions()};
    const int p_x{static_cast<int>(c_dimensions.size(0))};
    const int p_y{static_cast<int>(c_dimensions.size(1))};

    WHEN("summing the local sizes over the processes of each row and column") {
      THEN("the sums equal the global grid sizes") {
        const int local_x{coordinates[1] == 0 ? static_cast<int>(grid.size(0)) : 0};
        const int local_y{coordinates[0] == 0 ? static_cast<int>(grid.size(1)) : 0};
        int sum_x{0};
        int sum_y{0};
        comm_c.allreduce(mpl::plus<int>(), local_x, sum_x);
        comm_c.allreduce(mpl::plus<int>(), local_y, sum_y);
        REQUIRE(sum_x == nx);
        REQUIRE(sum_y == ny);
      }
    }

    WHEN("comparing the local sizes with the balanced decomposition") {
      THEN(
          "each local size is the floor or the ceiling of the global size divided by the "
          "process count") {
        CHECK(grid.size(0) >= nx / p_x);
        CHECK(grid.size(0) <= (nx + p_x - 1) / p_x);
        CHECK(grid.size(1) >= ny / p_y);
        CHECK(grid.size(1) <= (ny + p_y - 1) / p_y);
      }
    }
  }
}

SCENARIO("distributed_grid overlap regions are updated from neighboring processes",
         "[distributed_grid][halo]") {
  GIVEN("a 2D grid with overlap regions filled with global values in its interior") {
    const mpl::communicator& comm_world{mpl::environment::comm_world()};
    mpl::cartesian_communicator::dimensions dimensions{
        mpl::cartesian_communicator::non_periodic, mpl::cartesian_communicator::non_periodic};
    mpl::cartesian_communicator comm_c{comm_world,
                                       mpl::dims_create(comm_world.size(), dimensions)};
    mpl::distributed_grid<2, int> grid{comm_c, {{nx, overlap_x}, {ny, overlap_y}}};
    for (auto j{grid.obegin(1)}; j < grid.oend(1); ++j)
      for (auto i{grid.obegin(0)}; i < grid.oend(0); ++i) {
        const bool interior{grid.begin(0) <= i and i < grid.end(0) and grid.begin(1) <= j and
                            j < grid.end(1)};
        grid(i, j) = interior ? global_value(grid.gindex(0, i), grid.gindex(1, j)) : sentinel;
      }

    WHEN("updating the overlap regions from the neighboring processes") {
      update_overlap(comm_c, grid);

      THEN(
          "face overlap regions hold the values of the neighbors and the global border "
          "remains untouched") {
        int mismatches{0};
        for (auto j{grid.obegin(1)}; j < grid.oend(1); ++j)
          for (auto i{grid.obegin(0)}; i < grid.oend(0); ++i) {
            const bool inside_x{grid.begin(0) <= i and i < grid.end(0)};
            const bool inside_y{grid.begin(1) <= j and j < grid.end(1)};
            if (not inside_x and not inside_y)
              continue;
            const auto g_i{grid.gindex(0, i)};
            const auto g_j{grid.gindex(1, j)};
            const bool in_domain{0 <= g_i and g_i < nx and 0 <= g_j and g_j < ny};
            const int expected{in_domain ? global_value(g_i, g_j) : sentinel};
            if (grid(i, j) != expected)
              ++mismatches;
          }
        REQUIRE(mismatches == 0);
      }
    }
  }
}

SCENARIO("local_grid scatters and gathers the global grid",
         "[local_grid][distributed_grid][scatter][gather]") {
  GIVEN("a 2D Cartesian communicator, a local grid and a distributed grid on each process") {
    const mpl::communicator& comm_world{mpl::environment::comm_world()};
    mpl::cartesian_communicator::dimensions dimensions{
        mpl::cartesian_communicator::non_periodic, mpl::cartesian_communicator::non_periodic};
    mpl::cartesian_communicator comm_c{comm_world,
                                       mpl::dims_create(comm_world.size(), dimensions)};
    constexpr int root{0};
    const bool is_root{comm_c.rank() == root};
    mpl::local_grid<2, int> local{comm_c, {nx, ny}};
    mpl::distributed_grid<2, int> grid{comm_c, {{nx, overlap_x}, {ny, overlap_y}}};
    mpl::local_grid<2, int> gathered{comm_c, {nx, ny}};
    for (int j{0}; j < ny; ++j)
      for (int i{0}; i < nx; ++i)
        local(i, j) = global_value(i, j);

    THEN("the local grid extents cover the global grid") {
      CHECK(local.size(0) == nx);
      CHECK(local.size(1) == ny);
      CHECK(local.begin(0) == 0);
      CHECK(local.begin(1) == 0);
      CHECK(local.end(0) == nx);
      CHECK(local.end(1) == ny);
      CHECK(local.sub_layouts().size() == static_cast<std::size_t>(comm_c.size()));
    }

    WHEN("scattering the local grid over the distributed grids") {
      if (is_root)
        comm_c.scatterv(root, local.data(), local.sub_layouts(), grid.data(),
                        grid.interior_layout());
      else
        comm_c.scatterv(root, grid.data(), grid.interior_layout());

      THEN("the interior of each distributed grid holds the matching global values") {
        int mismatches{0};
        for (auto j{grid.begin(1)}; j < grid.end(1); ++j)
          for (auto i{grid.begin(0)}; i < grid.end(0); ++i)
            if (grid(i, j) != global_value(grid.gindex(0, i), grid.gindex(1, j)))
              ++mismatches;
        REQUIRE(mismatches == 0);
      }

      AND_WHEN("gathering the distributed grids back into a local grid") {
        if (is_root)
          comm_c.gatherv(root, grid.data(), grid.interior_layout(), gathered.data(),
                         gathered.sub_layouts());
        else
          comm_c.gatherv(root, grid.data(), grid.interior_layout());

        THEN("the root process receives the original global grid") {
          if (is_root) {
            int mismatches{0};
            for (int j{0}; j < ny; ++j)
              for (int i{0}; i < nx; ++i)
                if (gathered(i, j) != global_value(i, j))
                  ++mismatches;
            REQUIRE(mismatches == 0);
          }
        }
      }
    }
  }
}
