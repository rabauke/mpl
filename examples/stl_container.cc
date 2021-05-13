#include <cstdlib>
#include <iostream>
#include <iterator>
#include <vector>
#include <list>
#include <map>
#include <string>
#include <utility>
#include <cmath>
#include <mpl/mpl.hpp>

// print elements of a pair
template<typename ch, typename tr, typename T1, typename T2>
std::basic_ostream<ch, tr> &operator<<(std::basic_ostream<ch, tr> &out,
                                       const std::pair<T1, T2> &p) {
  return out << '(' << p.first << ',' << p.second << ')';
}

// print all elements of a container
template<typename ch, typename tr, typename C>
std::basic_ostream<ch, tr> &print_container(std::basic_ostream<ch, tr> &out, const C &c) {
  out << '(';
  for (auto i{std::begin(c)}; i != std::end(c); ++i) {
    out << (*i);
    if (std::next(i) != std::end(c))
      out << ',';
  }
  return out << ')';
}

// print all elements of a vector
template<typename ch, typename tr, typename T, typename A>
std::basic_ostream<ch, tr> &operator<<(std::basic_ostream<ch, tr> &out,
                                       const std::vector<T, A> &v) {
  return print_container(out, v);
}

// print all elements of a list
template<typename ch, typename tr, typename T, typename A>
std::basic_ostream<ch, tr> &operator<<(std::basic_ostream<ch, tr> &out,
                                       const std::list<T, A> &l) {
  return print_container(out, l);
}

// print all elements of a map
template<typename ch, typename tr, typename K, typename C, typename A>
std::basic_ostream<ch, tr> &operator<<(std::basic_ostream<ch, tr> &out,
                                       const std::map<K, C, A> &m) {
  return print_container(out, m);
}

// print all elements of a map
template<typename ch, typename tr, typename T>
std::basic_ostream<ch, tr> &operator<<(std::basic_ostream<ch, tr> &out,
                                       const std::valarray<T> &v) {
  return print_container(out, v);
}

// send an stl container
template<typename T>
void send(const mpl::communicator &comm, const T &x) {
  comm.send(x, 1);
}

// send an stl container
template<typename T>
void isend(const mpl::communicator &comm, const T &x) {
  mpl::irequest r{comm.isend(x, 1)};
  r.wait();
}

// receive an stl container
template<typename T>
void recv(const mpl::communicator &comm) {
  using value_type = mpl::detail::remove_const_from_members_t<typename T::value_type>;
  T x;
  auto s = comm.recv(x, 0);
  std::cout << "x = " << x << " with " << s.template get_count<value_type>() << " elements\n";
}

// receive an stl container
template<typename T>
void irecv(const mpl::communicator &comm) {
  using value_type = mpl::detail::remove_const_from_members_t<typename T::value_type>;
  T x;
  mpl::irequest r{comm.irecv(x, 0)};
  mpl::status s{r.wait()};
  std::cout << "x = " << x << " with " << s.template get_count<value_type>() << " elements\n";
}

int main() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  // run the program with two or more processes
  if (comm_world.size() < 2)
    comm_world.abort(EXIT_FAILURE);
  // process 0 sends
  if (comm_world.rank() == 0) {
    std::string t1{"Hello World!"};
    send(comm_world, t1);
    isend(comm_world, t1);
    std::vector<int> t2{0, 1, 2, 3, 4, 5, 6, 77, 42};
    send(comm_world, t2);
    isend(comm_world, t2);
    std::vector<std::tuple<int, double>> t3{{0, 0.0}, {1, 0.1}, {2, 0.2}, {3, 0.3}, {4, 0.4}};
    send(comm_world, t3);
    isend(comm_world, t3);
    std::vector<bool> t4{false, true, false, true, true};
    send(comm_world, t4);
    isend(comm_world, t4);
    std::valarray<double> t5{1, 2, 3, 4, 42, 4 * std::atan(1.0)};
    send(comm_world, t5);
    isend(comm_world, t5);
  }
  // process 1 receives
  if (comm_world.rank() == 1) {
    recv<std::string>(comm_world);
    irecv<std::string>(comm_world);
    recv<std::list<int>>(comm_world);
    irecv<std::list<int>>(comm_world);
    recv<std::map<int, double>>(comm_world);
    irecv<std::map<int, double>>(comm_world);
    recv<std::vector<bool>>(comm_world);
    irecv<std::vector<bool>>(comm_world);
    recv<std::valarray<double>>(comm_world);
    irecv<std::valarray<double>>(comm_world);
  }
  return EXIT_SUCCESS;
}
