#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <mpl/mpl.hpp>
#include <unistd.h>

static std::random_device rd;
static std::mt19937 mt(rd());

void fill_random(std::vector<double> &v) {
  std::uniform_real_distribution<double> dist(0, 1);
  for (auto &x : v)
    x=dist(mt);
}

// parallel sort algorithm for distributed memory computers
//
// algorith works as follows:
//   1) ecah process draws (size-1) random samples from its local data
//   2) all processes gather local random samples => size*(size-1) samples
//   3) size*(size-1) samples are sorted locally
//   4) pick (size-1) pivot elements from the globally sorted sample
//   5) partition local data with respect to the pivot elements into size bins
//   6) redistribute data such that data in bin i goes to process with rank i
//   7) sort redistributed data locally
//
// Note that the amount of data at each process changes during the algorithm.
// In worst case, a single process may hold finally all data.
//
template<typename T>
void parallel_sort(std::vector<T> &v) {
  auto comm_world{ mpl::environment::comm_world() };
  const int rank{ comm_world.rank() };
  const int size{ comm_world.size() };
  std::vector<T> local_pivots, pivots(size*(size-1));
  std::sample(begin(v), end(v), std::back_inserter(local_pivots), size-1, mt);
  comm_world.allgather(local_pivots.data(), mpl::vector_layout<T>(size-1),
                       pivots.data(), mpl::vector_layout<T>(size-1));
  std::sort(begin(pivots), end(pivots));
  local_pivots.resize(0);
  for (std::size_t i=1; i<size; ++i)
    local_pivots.push_back(pivots[i*(size-1)]);
  swap(local_pivots, pivots);
  std::vector<typename std::vector<T>::iterator> pivot_pos;
  pivot_pos.push_back(begin(v));
  for (T p : pivots)
    pivot_pos.push_back(std::partition(pivot_pos.back(), end(v), [p](T x) { return x<p; }));
  pivot_pos.push_back(end(v));
  std::vector<std::size_t> local_block_sizes, block_sizes(size*size);
  for (std::size_t i=0; i<pivot_pos.size()-1; ++i)
    local_block_sizes.push_back(std::distance(pivot_pos[i], pivot_pos[i+1]));
  comm_world.allgather(local_block_sizes.data(), mpl::vector_layout<std::size_t>(size),
                       block_sizes.data(), mpl::vector_layout<std::size_t>(size));
  mpl::layouts<T> send_layouts, recv_layouts;
  std::size_t send_pos{ 0 }, recv_pos{ 0 };
  for (int i=0; i<size; ++i) {
    send_layouts.push_back(mpl::indexed_layout<T>({{ block_sizes[rank*size+i], send_pos }}));
    send_pos+=block_sizes[rank*size+i];
    recv_layouts.push_back(mpl::indexed_layout<T>({{ block_sizes[rank+size*i], recv_pos }}));
    recv_pos+=block_sizes[rank+size*i];
  }
  std::vector<T> v2(recv_pos);
  comm_world.alltoallv(v.data(), send_layouts, v2.data(), recv_layouts);
  std::sort(begin(v2), end(v2));
  swap(v, v2);
}

int main() {
  auto comm_world{ mpl::environment::comm_world() };
  const int rank{ comm_world.rank() };
  const int size{ comm_world.size() };

  const std::size_t N{ 100000000/size };
  std::vector<double> v(N);
  fill_random(v);
  parallel_sort(v);

//  for (int i=0; i<size; ++i) {
//    if (rank==i) {
//      std::for_each(begin(v), end(v), [](double x) { std::cout << x << '\n'; });
//      std::cout << std::flush;
//    }
//    comm_world.barrier();
//    usleep(50000);
//  }
  return EXIT_SUCCESS;
}
