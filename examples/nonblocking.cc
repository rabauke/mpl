#include <cstdlib>
#include <iostream>
#include <array>
#include <vector>
#include <numeric>
#include <mpl/mpl.hpp>

template<typename I>
void print_range(const char *const str, I i1, I i2) {
  std::cout << str;
  while (i1 != i2) {
    std::cout << (*i1);
    ++i1;
    std::cout << ((i1 != i2) ? ' ' : '\n');
  }
}

int main() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  // run the program with two or more processes
  if (comm_world.size() < 2)
    return EXIT_FAILURE;
  const int n = 12;
  std::vector<int> v1(n), v2(n), v3(n), v4(n);
  mpl::contiguous_layout<int> l(n);
  // process 0 sends
  if (comm_world.rank() == 0) {
    // see MPI Standard for the semantics of standard send, buffered send,
    // synchronous send and ready send
    double x = 1.23456;
    mpl::irequest r(comm_world.isend(x, 1));  // send x to rank 1 via standard send
    r.wait();                                 // wait until send has finished
    ++x;
    {
      // create a buffer for buffered send,
      // memory will be freed on leaving the scope
      int size = {comm_world.bsend_size<decltype(x)>()};
      mpl::bsend_buffer<> buff(size);
      r = comm_world.ibsend(x, 1);  // send x to rank 1 via buffered send
      r.wait();                     // wait until send has finished
    }
    ++x;
    r = comm_world.issend(x, 1);  // send x to rank 1 via synchronous send
    r.wait();                     // wait until send has finished
    ++x;
    r = comm_world.irsend(x, 1);  // send x to rank 1 via ready send
    r.wait();                     // wait until send has finished
    std::iota(v1.begin(), v1.end(), 0);
    std::iota(v2.begin(), v2.end(), 1);
    std::iota(v3.begin(), v3.end(), 2);
    std::iota(v4.begin(), v4.end(), 3);
    {
      // create a buffer for buffered send,
      // memory will be freed on leaving the scope
      int size = {comm_world.bsend_size(l)};
      mpl::bsend_buffer<> buff(size);
      mpl::irequest_pool r;
      r.push(comm_world.isend(v1.data(), l, 1));   // send x to rank 1 via standard send
      r.push(comm_world.ibsend(v2.data(), l, 1));  // send x to rank 1 via buffered send
      r.push(comm_world.issend(v3.data(), l, 1));  // send x to rank 1 via synchronous send
      r.push(comm_world.irsend(v4.data(), l, 1));  // send x to rank 1 via ready send
      r.waitall();                                 // wait until all sends have finished
    }
    {
      mpl::irequest_pool r;
      r.push(comm_world.isend(v1.data(), l, 1));   // send v1 to rank 1 via standard send
      r.push(comm_world.ibsend(v2.data(), l, 1));  // send v2 to rank 1 via buffered send
      r.push(comm_world.issend(v3.data(), l, 1));  // send v3 to rank 1 via synchronous send
      r.push(comm_world.irsend(v4.data(), l, 1));  // send v4 to rank 1 via ready send
      std::array<mpl::irequest_pool::size_type, 4>
          finished;  // memory to store indices of finished send operations
      while (true) {
        auto i = r.waitsome(finished.begin());  // wait until one ore more sends have finished
        if (i == finished.begin())              // there have been no pending sends
          break;
        // print indices of finished sends
        std::cout << "send finished : ";
        std::for_each(finished.begin(), i,
                      [](mpl::irequest_pool::size_type j) { std::cout << j << ' '; });
        std::cout << "\n";
      }
    }
  }
  // process 1 receives
  if (comm_world.rank() == 1) {
    double x;
    mpl::irequest r(comm_world.irecv(x, 0));  // receive x from rank 0
    r.wait();                                 // wait until receive has finished
    std::cout << "x = " << x << '\n';
    r = comm_world.irecv(x, 0);  // receive x from rank 0
    r.wait();                    // wait until receive has finished
    std::cout << "x = " << x << '\n';
    r = comm_world.irecv(x, 0);  // receive x from rank 0
    r.wait();                    // wait until receive has finished
    std::cout << "x = " << x << '\n';
    r = comm_world.irecv(x, 0);  // receive x from rank 0
    r.wait();                    // wait until receive has finished
    std::cout << "x = " << x << '\n';
    {
      mpl::irequest_pool r;
      r.push(comm_world.irecv(v1.data(), l, 0));  // receive v1 from rank 0
      r.push(comm_world.irecv(v2.data(), l, 0));  // receive v2 from rank 0
      r.push(comm_world.irecv(v3.data(), l, 0));  // receive v3 from rank 0
      r.push(comm_world.irecv(v4.data(), l, 0));  // receive v4 from rank 0
      r.waitall();                                // wait until all receives have finished
      print_range("v = ", v1.begin(), v1.end());
      print_range("v = ", v2.begin(), v2.end());
      print_range("v = ", v3.begin(), v3.end());
      print_range("v = ", v4.begin(), v4.end());
    }
    {
      mpl::irequest_pool r;
      r.push(comm_world.irecv(v1.data(), l, 0));  // receive v1 from rank 0
      r.push(comm_world.irecv(v2.data(), l, 0));  // receive v2 from rank 0
      r.push(comm_world.irecv(v3.data(), l, 0));  // receive v3 from rank 0
      r.push(comm_world.irecv(v4.data(), l, 0));  // receive v4 from rank 0
      while (true) {
        std::array<mpl::irequest_pool::size_type, 4>
            finished;  // memory to store indices of finished recv operations
        auto i =
            r.waitsome(finished.begin());  // wait until one ore more receives have finished
        if (i == finished.begin())         // there have been no pending receives
          break;
        // print indices of finished receives
        std::cout << "recv finished : ";
        std::for_each(finished.begin(), i,
                      [](mpl::irequest_pool::size_type j) { std::cout << j << ' '; });
        std::cout << '\n';
      }
      print_range("v = ", v1.begin(), v1.end());
      print_range("v = ", v2.begin(), v2.end());
      print_range("v = ", v3.begin(), v3.end());
      print_range("v = ", v4.begin(), v4.end());
    }
  }
  return EXIT_SUCCESS;
}
