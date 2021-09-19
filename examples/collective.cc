#include <cstdlib>
#include <iostream>
#include <numeric>
#include <mpl/mpl.hpp>


struct my_array {
  static constexpr int n_0{3};
  static constexpr int n_1{4};
  double data[n_0][n_1]{};
  // overload operators for some syntactical sugar
  const double &operator()(int i_0, int i_1) const { return data[i_0][i_1]; }
  double &operator()(int i_0, int i_1) { return data[i_0][i_1]; }
};

// use reflection macro to make the struct compatible with mpl
MPL_REFLECTION(my_array, data);

// overload plus operator
my_array operator+(const my_array &a, const my_array &b) {
  my_array res;
  for (int i_1{0}; i_1 < my_array::n_1; ++i_1)
    for (int i_0{0}; i_0 < my_array::n_0; ++i_0)
      res(i_0, i_1) = a(i_0, i_1) + b(i_0, i_1);
  return res;
}


int main() {
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  int root{0};

  // synchronize processes via barrier
  comm_world.barrier();
  std::cout << mpl::environment::processor_name() << " has passed barrier\n";
  comm_world.barrier();
  double x{0};
  if (comm_world.rank() == root)
    x = 10;

  // broadcast x to all from root rank
  comm_world.bcast(root, x);
  std::cout << "x = " << x << '\n';

  // collect data from all ranks via gather to root rank
  x = comm_world.rank() + 1;
  if (comm_world.rank() == root) {
    std::vector<double> v(comm_world.size());  // receive buffer
    comm_world.gather(root, x, v.data());
    std::cout << "v = ";
    for (const auto &item : v)
      std::cout << item << ' ';
    std::cout << '\n';
  } else
    comm_world.gather(root, x);

  // send data to all ranks via scatter from root rank
  double y{0};
  if (comm_world.rank() == root) {
    std::vector<double> v(comm_world.size());  // send buffer
    std::iota(v.begin(), v.end(), 1);          // populate send buffer
    comm_world.scatter(root, v.data(), y);
  } else
    comm_world.scatter(root, y);
  std::cout << "y = " << y << '\n';

  // reduce/sum all values of x on all nodes and send global result to root
  if (comm_world.rank() == root) {
    comm_world.reduce(mpl::plus<double>(), root, x, y);
    std::cout << "sum after reduce " << y << '\n';
  } else
    comm_world.reduce(mpl::plus<double>(), root, x);

  // reduce/multiply all values of x on all nodes and send global result to all
  comm_world.allreduce(mpl::multiplies<double>(), x, y);
  std::cout << "sum after allreduce " << y << '\n';

  // reduce a C-style array using a contiguous layout
  {
    const int n_0{3}, n_1{4};
    using array_type = double[n_0][n_1];
    array_type array;
    for (int i_1{0}; i_1 < n_1; ++i_1)
      for (int i_0{0}; i_0 < n_0; ++i_0)
        array[i_0][i_1] = i_0 + 100 * i_1;
    mpl::contiguous_layout<double> l(n_1 * n_0);
    comm_world.allreduce(mpl::plus<double>(), &array[0][0], l);
    if (comm_world.rank() == 0) {
      std::cout << "array after allreduce\n";
      for (int i_1{0}; i_1 < n_1; ++i_1) {
        for (int i_0{0}; i_0 < n_0; ++i_0)
          std::cout << array[i_0][i_1] << '\t';
        std::cout << '\n';
      }
    }
  }

  // reduce a wrapped array
  {
    my_array array;
    for (int i_1{0}; i_1 < my_array::n_1; ++i_1)
      for (int i_0{0}; i_0 < my_array::n_0; ++i_0)
        array(i_0, i_1) = i_0 + 100 * i_1;
    comm_world.allreduce(mpl::plus<my_array>(), array);
    if (comm_world.rank() == 0) {
      std::cout << "array after allreduce\n";
      for (int i_1{0}; i_1 < my_array::n_1; ++i_1) {
        for (int i_0{0}; i_0 < my_array::n_0; ++i_0)
          std::cout << array(i_0, i_1) << '\t';
        std::cout << '\n';
      }
    }
  }

  return EXIT_SUCCESS;
}
