#include <cstdlib>
#include <complex>
#include <iostream>
#include <mpl/mpl.hpp>

template<typename T1, typename T2>
std::ostream & operator<<(std::ostream &out, const std::pair<T1, T2> &p) {
  out << p.first << ' ' << p.second;
  return out;
}

template<typename F, typename T, std::size_t n>
class apply_n {
  F &f;
public:
  apply_n(F &f) : f(f) {
  }
  void operator()(const T &x) {
    apply_n<F, T, n-1> next(f);
    next(x);
    f(std::get<n-1>(x));
  }
};

template<typename F, typename T>
struct apply_n<F, T, 1> {
  F &f;
public:
  apply_n(F &f) : f(f) {
  }
  void operator()(const T &x) {
    f(std::get<0>(x));
  }
};

template<typename F, typename... Args>
void apply(const std::tuple<Args...> &t, F &f) {
  apply_n<F, std::tuple<Args...>, std::tuple_size<std::tuple<Args...> >::value> app(f);
  app(t);
}

class print_element {
  std::ostream &out;
public:
  print_element(std::ostream &out) : out(out) {
  }
  template<typename T>
  void operator()(const T &x) const {
    out << x << ' ';
  }
};

template<typename... Ts>
std::ostream & operator<<(std::ostream &out, const std::tuple<Ts...> &t) {
  print_element f(out);
  apply<print_element>(t, f);
  return out;
}

template<typename T>
void send(const mpl::communicator &comm, const T &x) {
  comm.send(x, 1);
}

template<typename T>
void recv(const mpl::communicator &comm) {
  T x;
  comm.recv(x, 0);
  std::cout << "x = " << std::boolalpha << x << '\n';
}

int main() {
  const mpl::communicator &comm_world=mpl::environment::comm_world();
  if (comm_world.size()<2)
    comm_world.abort(EXIT_FAILURE);
  if (comm_world.rank()==0) {
    char t1='A';                               send(comm_world, t1);
    signed char t2='B';                        send(comm_world, t2);
    unsigned char t3='C';                      send(comm_world, t3);
    signed short int t4=-1;                    send(comm_world, t4);
    unsigned short int t5=1;                   send(comm_world, t5);
    signed int t6=-10;                         send(comm_world, t6);
    unsigned int t7=10;                        send(comm_world, t7);
    signed long int t8=-100;                   send(comm_world, t8);
    unsigned long int t9=100;                  send(comm_world, t9);
    signed long long int t10=-1000;            send(comm_world, t10);
    unsigned long long int t11=1000;           send(comm_world, t11);
    bool t12=true;                             send(comm_world, t12);
    float t13=1.2345;                          send(comm_world, t13);
    double t14=2.3456;                         send(comm_world, t14);
    long double t15=3.4567;                    send(comm_world, t15);
    std::complex<float> t16(1.2, -1.2);        send(comm_world, t16);
    std::complex<double> t17(2.3, -2.3);       send(comm_world, t17);
    std::complex<long double> t18(3.4, -3.4);  send(comm_world, t18);
    std::pair<int, double> t19(-2, 0.1234);    send(comm_world, t19);
    std::tuple<int, std::complex<double> > t20(-2, 0.1234);   send(comm_world, t20);
  } 
  if (comm_world.rank()==1) {
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
    recv<std::complex<float> >(comm_world);
    recv<std::complex<double> >(comm_world);
    recv<std::complex<long double> >(comm_world);
    recv<std::pair<int, double> >(comm_world);
    recv<std::tuple<int, std::complex<double> > >(comm_world);
  }
  return EXIT_SUCCESS;
}
