#include <cstdlib>
#include <ctime>
#include <iostream>
#include <functional>
#include <vector>
#include <mpl/mpl.hpp>

// calculate least common multiple
template<typename T>
class lcm : public std::function<T (T, T)> {
  // helper: calculate greatest common divisor
  T gcd(T a, T b) {
    T zero=T(), t;
    if (a<zero) a=-a;
    if (b<zero) b=-b;
    while (b>zero) {
      t=a%b;  a=b;  b=t;
    }
    return a;
  }
public:
  T operator()(T a, T b) {
    T zero=T();
    T t((a/gcd(a, b))*b);
    if (t<zero)
      return -t;
    return t;
  }
};

int main() {
  const mpl::communicator &comm_world=mpl::environment::comm_world();
  const int n=12;
  std::vector<int> v;
  std::srand(std::time(0)*comm_world.rank());
  for (int i=0; i<n; ++i) 
    v.push_back(std::rand()%16+1);
  mpl::contiguous_layout<int> layout(n);
  std::vector<int> result(n);
  comm_world.reduce(v.data(), result.data(), layout, lcm<int>(), 0);
  if (comm_world.rank()==0) {
    std::cout << "Results:\n";
    for (int i=0; i<n; ++i) 
      std::cout << result[i] << '\t';
    std::cout << "\n\nArguments:\n";
    for (int r=0; r<comm_world.size(); ++r) {
      if (r>0)
	comm_world.recv(v.data(), layout, r);
      for (int i=0; i<n; ++i) 
	std::cout << v[i] << '\t';
      std::cout << '\n';
    }
  } else {
    comm_world.send(v.data(), layout, 0);
  }
  return EXIT_SUCCESS;
}
