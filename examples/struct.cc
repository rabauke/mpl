#include <cstdlib>
#include <vector>
#include <iostream>
#include <numeric>
#include <mpl/mpl.hpp>

// some structures
struct structure {
  double d{0};
  int i[9]{0, 0, 0, 0, 0, 0, 0, 0, 0};

  structure() = default;
};

// print elements of structure
template<typename ch, typename tr>
std::basic_ostream<ch, tr> &operator<<(std::basic_ostream<ch, tr> &out, const structure &s) {
  out << '(' << s.d << ",[" << s.i[0];
  for (int i{1}; i < 9; ++i)
    out << ',' << s.i[i];
  return out << "])";
}

struct structure2 {
  double d{0};
  structure str;

  structure2() = default;
};

// print elements of structure2
template<typename ch, typename tr>
std::basic_ostream<ch, tr> &operator<<(std::basic_ostream<ch, tr> &out, const structure2 &s) {
  return out << '(' << s.d << "," << s.str << ")";
}


// specialize trait template class struct_builder
// for the structures defined above
namespace mpl {

  template<>
  class struct_builder<structure> : public base_struct_builder<structure> {
    struct_layout<structure> layout;

  public:
    struct_builder() {
      structure str;
      layout.register_struct(str);
      // register each element of struct structure
      layout.register_element(str.d);
      layout.register_element(str.i);
      // finalize
      define_struct(layout);
    }
  };

}  // namespace mpl

// MPL_REFLECTION is a convenient macro which creates the required
// specialization of the struct_builder template automatically.  Just
// pass the class name and the public members as arguments to the
// macro.  MPL_REFLECTION is limited to 120 class members.
MPL_REFLECTION(structure2, d, str)

int main() {
  const mpl::communicator &comm_world = mpl::environment::comm_world();
  // run the program with two or more processes
  if (comm_world.size() < 2)
    comm_world.abort(EXIT_FAILURE);
  // send / receive a single structure
  structure str;
  if (comm_world.rank() == 0) {
    str.d = 1;
    std::iota(str.i, str.i + 9, 1);
    comm_world.send(str, 1);
  }
  if (comm_world.rank() == 1) {
    comm_world.recv(str, 0);
    std::cout << str << '\n';
  }
  // send / receive a single structure containing another structure
  structure2 str2;
  if (comm_world.rank() == 0) {
    str2.d = 1;
    str2.str.d = 1;
    std::iota(str2.str.i, str2.str.i + 9, 1);
    comm_world.send(str2, 1);
  }
  if (comm_world.rank() == 1) {
    comm_world.recv(str2, 0);
    std::cout << str2 << '\n';
  }
  // send / receive a vector of structures
  const int field_size = 8;
  std::vector<structure> str_field(field_size);
  mpl::contiguous_layout<structure> str_field_layout(field_size);
  if (comm_world.rank() == 0) {
    // populate vector of structures
    for (int k{0}; k < field_size; ++k) {
      str_field[k].d = k + 1;
      std::iota(str_field[k].i, str_field[k].i + 9, 1 + k);
    }
    // send vector of structures
    comm_world.send(str_field.data(), str_field_layout, 1);
  }
  if (comm_world.rank() == 1) {
    // receive vector of structures
    comm_world.recv(str_field.data(), str_field_layout, 0);
    for (int k = 0; k < field_size; ++k)
      std::cout << str_field[k] << '\n';
  }

  return EXIT_SUCCESS;
}
