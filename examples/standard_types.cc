#include <cstdlib>
#include <complex>
#include <iostream>
#include <tuple>
#include <array>
#include <utility>
#include <mpl/mpl.hpp>

// print elements of a pair
template<typename ch, typename tr, typename T1, typename T2>
std::basic_ostream<ch, tr> &operator<<(std::basic_ostream<ch, tr> &out,
                                       const std::pair<T1, T2> &p) {
  return out << '(' << p.first << ',' << p.second << ')';
}

// helper function for printing all elements of a tuple/ an array
template<typename ch, typename tr, typename tuple, std::size_t... is>
void print_tuple_impl(std::basic_ostream<ch, tr> &out, const tuple &t,
                      std::index_sequence<is...> is_) {
  auto print_element = [&](auto i, auto a) -> void {
    out << a << (i + 1 < is_.size() ? "," : "");
  };
  std::initializer_list<int> _{(print_element(is, std::get<is>(t)), 0)...};
}

// print all elements of a tuple
template<typename ch, typename tr, typename... args>
std::basic_ostream<ch, tr> &operator<<(std::basic_ostream<ch, tr> &out,
                                       const std::tuple<args...> &t) {
  out << '(';
  print_tuple_impl(out, t, std::index_sequence_for<args...>{});
  return out << ')';
}

// print all elements of an array
template<typename ch, typename tr, typename ty, std::size_t s>
std::basic_ostream<ch, tr> &operator<<(std::basic_ostream<ch, tr> &out,
                                       const std::array<ty, s> &t) {
  out << '(';
  print_tuple_impl(out, t, std::make_integer_sequence<std::size_t, s>{});
  return out << ')';
}

// print a byte
template<typename ch, typename tr>
std::basic_ostream<ch, tr> &operator<<(std::basic_ostream<ch, tr> &out, const std::byte &t) {
  return out << std::to_integer<int>(t);
}

// send some item of a standard type
template<typename T>
void send(const mpl::communicator &comm, const T &x) {
  comm.send(x, 1);
}

// receive some item of a standard type
template<typename T>
void recv(const mpl::communicator &comm) {
  T x;
  comm.recv(x, 0);
  std::cout << "x = " << std::boolalpha << x << '\n';
}

int main() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  // run the program with two or more processes
  if (comm_world.size() < 2)
    comm_world.abort(EXIT_FAILURE);
  // process 0 sends
  if (comm_world.rank() == 0) {
    char t1 = 'A';
    send(comm_world, t1);
    signed char t2 = 'B';
    send(comm_world, t2);
    unsigned char t3 = 'C';
    send(comm_world, t3);
    signed short int t4 = -1;
    send(comm_world, t4);
    unsigned short int t5 = 1;
    send(comm_world, t5);
    signed int t6 = -10;
    send(comm_world, t6);
    unsigned int t7 = 10;
    send(comm_world, t7);
    signed long int t8 = -100;
    send(comm_world, t8);
    unsigned long int t9 = 100;
    send(comm_world, t9);
    signed long long int t10 = -1000;
    send(comm_world, t10);
    unsigned long long int t11 = 1000;
    send(comm_world, t11);
    bool t12 = true;
    send(comm_world, t12);
    float t13 = 1.2345f;
    send(comm_world, t13);
    double t14 = 2.3456;
    send(comm_world, t14);
    long double t15 = 3.4567;
    send(comm_world, t15);
    std::complex<float> t16(1.2f, -1.2f);
    send(comm_world, t16);
    std::complex<double> t17(2.3, -2.3);
    send(comm_world, t17);
    std::complex<long double> t18(3.4, -3.4);
    send(comm_world, t18);
    std::pair<int, double> t19(-2, 0.1234);
    send(comm_world, t19);
    std::tuple<int, std::complex<double>> t20(-2, 0.1234);
    send(comm_world, t20);
    std::array<int, 4> t21{1, 2, 3, 4};
    send(comm_world, t21);
    std::byte t22{255};
    send(comm_world, t22);
  }
  // process 1 receives
  if (comm_world.rank() == 1) {
    recv<char>(comm_world);
    recv<signed char>(comm_world);
    recv<unsigned char>(comm_world);
    recv<signed short int>(comm_world);
    recv<unsigned short int>(comm_world);
    recv<signed int>(comm_world);
    recv<unsigned int>(comm_world);
    recv<signed long int>(comm_world);
    recv<unsigned long int>(comm_world);
    recv<signed long long int>(comm_world);
    recv<unsigned long long int>(comm_world);
    recv<bool>(comm_world);
    recv<float>(comm_world);
    recv<double>(comm_world);
    recv<long double>(comm_world);
    recv<std::complex<float>>(comm_world);
    recv<std::complex<double>>(comm_world);
    recv<std::complex<long double>>(comm_world);
    recv<std::pair<int, double>>(comm_world);
    recv<std::tuple<int, std::complex<double>>>(comm_world);
    recv<std::array<int, 4>>(comm_world);
    recv<std::byte>(comm_world);
  }
  return EXIT_SUCCESS;
}
