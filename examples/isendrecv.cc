#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>
#include <mpl/mpl.hpp>


int main() {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  const int rank{comm_world.rank()};
  const int size{comm_world.size()};
  const int right{(rank + 1) % size};
  const int left{(rank - 1 + size) % size};
  const mpl::tag_t tag{0};

  // exchange single values with the neighbors in a ring, send to the right neighbor and
  // receive from the left neighbor without blocking
  int send_value{rank}, recv_value{-1};
  mpl::irequest r_1{comm_world.isendrecv(send_value, right, tag, recv_value, left, tag)};
  // ... do some useful work here while the communication is in progress ...
  mpl::status_t s{r_1.wait()};
  std::cout << "rank " << rank << " received " << recv_value << " from rank " << s.source()
            << '\n';

  // exchange arrays, e.g., halo data in a domain decomposition
  const int n{8};
  std::vector<double> send_buf(n), recv_buf(n);
  std::iota(send_buf.begin(), send_buf.end(), 10.0 * rank);
  mpl::irequest r_2{comm_world.isendrecv(send_buf.begin(), send_buf.end(), right, tag,
                                         recv_buf.begin(), recv_buf.end(), left, tag)};
  r_2.wait();
  std::cout << "rank " << rank << " received first array element " << recv_buf.front() << '\n';

  // rotate data in place
  std::vector<int> data(n, rank);
  mpl::irequest r_3{
      comm_world.isendrecv_replace(data.begin(), data.end(), right, tag, left, tag)};
  r_3.wait();
  std::cout << "rank " << rank << " holds data of rank " << data.front() << " after rotation\n";
  return EXIT_SUCCESS;
}
