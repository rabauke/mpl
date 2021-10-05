#include <cstdlib>
#include <iostream>
#include <array>
#include <vector>
#include <numeric>
#include <mpl/mpl.hpp>

template<typename I>
void print_range(const char *const str, I i_1, I i_2) {
  std::cout << str;
  while (i_1 != i_2) {
    std::cout << (*i_1);
    ++i_1;
    std::cout << ((i_1 != i_2) ? ' ' : '\n');
  }
}

int main() {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  // run the program with two or more processes
  if (comm_world.size() < 2)
    return EXIT_FAILURE;
  const int n{12};
  std::vector<int> v_1(n), v_2(n), v_3(n), v_4(n);
  mpl::contiguous_layout<int> l(n);
  // process 0 sends
  if (comm_world.rank() == 0) {
    // see MPI Standard for the semantics of standard send, buffered send,
    // synchronous send and ready send
    double x{1.23456};
    mpl::irequest r(comm_world.isend(x, 1));  // send x to rank 1 via standard send
    r.wait();                                 // wait until send has finished
    ++x;
    {
      // create a buffer for buffered send,
      // memory will be freed on leaving the scope
      const int size{comm_world.bsend_size<decltype(x)>()};
      mpl::bsend_buffer buff(size);
      r = comm_world.ibsend(x, 1);  // send x to rank 1 via buffered send
      r.wait();                     // wait until send has finished
    }
    ++x;
    r = comm_world.issend(x, 1);  // send x to rank 1 via synchronous send
    r.wait();                     // wait until send has finished
    ++x;
    r = comm_world.irsend(x, 1);  // send x to rank 1 via ready send
    r.wait();                     // wait until send has finished
    std::iota(v_1.begin(), v_1.end(), 0);
    std::iota(v_2.begin(), v_2.end(), 1);
    std::iota(v_3.begin(), v_3.end(), 2);
    std::iota(v_4.begin(), v_4.end(), 3);
    {
      // create a buffer for buffered send,
      // memory will be freed on leaving the scope
      const int size{comm_world.bsend_size(l)};
      mpl::bsend_buffer buff(size);
      mpl::irequest_pool r;
      r.push(comm_world.isend(v_1.data(), l, 1));   // send x to rank 1 via standard send
      r.push(comm_world.ibsend(v_2.data(), l, 1));  // send x to rank 1 via buffered send
      r.push(comm_world.issend(v_3.data(), l, 1));  // send x to rank 1 via synchronous send
      r.push(comm_world.irsend(v_4.data(), l, 1));  // send x to rank 1 via ready send
      r.waitall();                                  // wait until all sends have finished
    }
    {
      mpl::irequest_pool r;
      r.push(comm_world.isend(v_1.data(), l, 1));   // send v1 to rank 1 via standard send
      r.push(comm_world.ibsend(v_2.data(), l, 1));  // send v2 to rank 1 via buffered send
      r.push(comm_world.issend(v_3.data(), l, 1));  // send v3 to rank 1 via synchronous send
      r.push(comm_world.irsend(v_4.data(), l, 1));  // send v4 to rank 1 via ready send
      std::array<mpl::irequest_pool::size_type, 4>
          finished;  // memory to store indices of finished send operations
      while (true) {
        auto i{r.waitsome(finished.begin())};  // wait until one or more sends have finished
        if (i == finished.begin())             // there have been no pending sends
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
      r.push(comm_world.irecv(v_1.data(), l, 0));  // receive v1 from rank 0
      r.push(comm_world.irecv(v_2.data(), l, 0));  // receive v2 from rank 0
      r.push(comm_world.irecv(v_3.data(), l, 0));  // receive v3 from rank 0
      r.push(comm_world.irecv(v_4.data(), l, 0));  // receive v4 from rank 0
      r.waitall();                                 // wait until all receives have finished
      print_range("v = ", v_1.begin(), v_1.end());
      print_range("v = ", v_2.begin(), v_2.end());
      print_range("v = ", v_3.begin(), v_3.end());
      print_range("v = ", v_4.begin(), v_4.end());
    }
    {
      mpl::irequest_pool r;
      r.push(comm_world.irecv(v_1.data(), l, 0));  // receive v1 from rank 0
      r.push(comm_world.irecv(v_2.data(), l, 0));  // receive v2 from rank 0
      r.push(comm_world.irecv(v_3.data(), l, 0));  // receive v3 from rank 0
      r.push(comm_world.irecv(v_4.data(), l, 0));  // receive v4 from rank 0
      while (true) {
        std::array<mpl::irequest_pool::size_type, 4>
            finished;  // memory to store indices of finished recv operations
        auto i{r.waitsome(finished.begin())};  // wait until one or more receives have finished
        if (i == finished.begin())             // there have been no pending receives
          break;
        // print indices of finished receives
        std::cout << "recv finished : ";
        std::for_each(finished.begin(), i,
                      [](mpl::irequest_pool::size_type j) { std::cout << j << ' '; });
        std::cout << '\n';
      }
      print_range("v = ", v_1.begin(), v_1.end());
      print_range("v = ", v_2.begin(), v_2.end());
      print_range("v = ", v_3.begin(), v_3.end());
      print_range("v = ", v_4.begin(), v_4.end());
    }
  }
  return EXIT_SUCCESS;
}
