#include <cstdlib>
#include <iostream>
#include <vector>
#include <list>
#include <numeric>
#include <algorithm>
#include <mpl/mpl.hpp>

template<typename I>
void print_range(const char * const str, I i1, I i2) {
  std::cout << str;
  while (i1!=i2) {
    std::cout << (*i1);
    ++i1;
    std::cout << ((i1!=i2) ? ' ' : '\n');
  }
}

int main() {
  const mpl::communicator &comm_world=mpl::environment::comm_world();
  if (comm_world.size()<2) 
    return EXIT_FAILURE;
  const int n=12;
  std::vector<int> v1(n), v2(n), v3(n), v4(n);
  mpl::contiguous_layout<int> l(n);
  if (comm_world.rank()==0) {
    double x=1.23456;
    { mpl::irequest r(comm_world.isend(x, 1));  r.wait(); }
    ++x;
    { 
      int size={ comm_world.bsend_size<decltype(x)>() };
      mpl::bsend_buffer<> buff(size);
      mpl::irequest r(comm_world.ibsend(x, 1));  r.wait(); 
    }
    ++x;
    { mpl::irequest r(comm_world.issend(x, 1));  r.wait(); }
    ++x;
    { mpl::irequest r(comm_world.irsend(x, 1));  r.wait(); }
    auto add_one=[](int x){ return x+1; };
    std::iota(v1.begin(), v1.end(), 0);
    std::transform(v1.begin(), v1.end(), v2.begin(), add_one);
    std::transform(v2.begin(), v2.end(), v3.begin(), add_one);
    std::transform(v3.begin(), v3.end(), v4.begin(), add_one);
    {
      int size={ comm_world.bsend_size(l) };
      mpl::bsend_buffer<> buff(size);
      mpl::irequest_pool r;
      r.push(comm_world.isend(v1.data(), l, 1));
      r.push(comm_world.ibsend(v2.data(), l, 1));
      r.push(comm_world.issend(v3.data(), l, 1));
      r.push(comm_world.irsend(v4.data(), l, 1));
      r.waitall();
    }
    {
      mpl::irequest_pool r;
      r.push(comm_world.isend(v1.data(), l, 1));
      r.push(comm_world.ibsend(v2.data(), l, 1));
      r.push(comm_world.issend(v3.data(), l, 1));
      r.push(comm_world.irsend(v4.data(), l, 1));
      while (true) {
	std::list<int> finished;
	r.waitsome(std::back_inserter(finished));
	if (finished.empty())
	  break;
      } 
    }
  }
  if (comm_world.rank()==1) {
    double x;
    { mpl::irequest r(comm_world.irecv(x, 0));  r.wait(); }
    std::cout << "x = " << x << '\n';
    { mpl::irequest r(comm_world.irecv(x, 0));  r.wait(); }
    std::cout << "x = " << x << '\n';
    { mpl::irequest r(comm_world.irecv(x, 0));  r.wait(); }
    std::cout << "x = " << x << '\n';
    { mpl::irequest r(comm_world.irecv(x, 0));  r.wait(); }
    std::cout << "x = " << x << '\n';
    {
      mpl::irequest_pool r;
      r.push(comm_world.irecv(v1.data(), l, 0));
      r.push(comm_world.irecv(v2.data(), l, 0));
      r.push(comm_world.irecv(v3.data(), l, 0));
      r.push(comm_world.irecv(v4.data(), l, 0));
      r.waitall();
      print_range("v = ", v1.begin(), v1.end());
      print_range("v = ", v2.begin(), v2.end());
      print_range("v = ", v3.begin(), v3.end());
      print_range("v = ", v4.begin(), v4.end());
    }
    {
      mpl::irequest_pool r;
      r.push(comm_world.irecv(v1.data(), l, 0));
      r.push(comm_world.irecv(v2.data(), l, 0));
      r.push(comm_world.irecv(v3.data(), l, 0));
      r.push(comm_world.irecv(v4.data(), l, 0));
      while (true) {
	std::array<int, n> finished;
	// r.waitsome(std::back_inserter(finished));
	// if (finished.empty())
	//   break;
	std::array<int, n>::iterator i=r.waitsome(finished.begin());
	if (i==finished.begin())
	  break;
	std::cout << "finished : ";
	std::for_each(finished.begin(), i, [](int i){  std::cout << i << ' '; });
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
