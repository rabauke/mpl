#include <cstdlib>
#include <vector>
#include <iostream>
#include <mpl/mpl.hpp>

struct structure {
  double d;
  char i[9];
  structure() : d(0) {
    for (int j=0; j<9; ++j) 
      i[j]=0;
  }
};

struct structure2 {
  double d;
  structure str;
  structure2() : d(0), str() { }
};

namespace mpl {

  template<>
  class struct_builder<structure> : public base_struct_builder<structure> {
    struct_layout<structure> layout;
  public:
    struct_builder() {
      structure str;
      layout.register_struct(str);
      // register each element
      layout.register_element(str.d);
      layout.register_element(str.i);
      define_struct(layout);
    }
  };
  
  template<>
  class struct_builder<structure2> : public base_struct_builder<structure2> {
    struct_layout<structure2> layout;
  public:
    struct_builder() {
      structure2 str2;
      layout.register_struct(str2);
      // register each element
      layout.register_element(str2.d);
      layout.register_element(str2.str);
      define_struct(layout);
    }
  };
  
}

int main() {
  const mpl::communicator &comm_world=mpl::environment::comm_world();
  if (comm_world.size()<2)
    comm_world.abort(EXIT_FAILURE);
  // --- send / receive a single structure
  structure str;
  if (comm_world.rank()==0) {
    str.d=1;
    for (int j=0; j<9; ++j) 
      str.i[j]=j+1;
    comm_world.send(str, 1, 0);
  }
  if (comm_world.rank()==1) {
    comm_world.recv(str, 0, 0);
    std::cout << "d = " << str.d << '\n';
    for (int j=0; j<9; ++j) 
      std::cout << "i[" << j << "] = " << static_cast<int>(str.i[j]) << '\n';
  }
  // --- send / receive a single structure containg another structure
  structure2 str2;
  if (comm_world.rank()==0) {
    str2.d=1;
    str2.str.d=1;
    for (int j=0; j<9; ++j) 
      str2.str.i[j]=j+1;
    comm_world.send(str2, 1, 0);
  }
  if (comm_world.rank()==1) {
    comm_world.recv(str2, 0, 0);
    std::cout << "d = " << str2.d << '\n';
    std::cout << "str.d = " << str2.str.d << '\n';
    for (int j=0; j<9; ++j) 
      std::cout << "str.i[" << j << "] = " << static_cast<int>(str2.str.i[j]) << '\n';
  }
  // --- send / receive a field of single structures
  int field_size=8;
  std::vector<structure> str_field(field_size);
  mpl::contiguous_layout<structure> str_field_layout(field_size);
  if (comm_world.rank()==0) {
    for (int k=0; k<field_size; ++k) {
      str_field[k].d=k+1;
      for (int j=0; j<9; ++j) 
  	str_field[k].i[j]=j+10*k;
    }
    comm_world.send(&str_field[0], str_field_layout, 1, 0);
  }
  if (comm_world.rank()==1) {
    comm_world.recv(&str_field[0], str_field_layout, 0, 0);
    for (int k=0; k<field_size; ++k) {
      std::cout << "d = " << str_field[k].d << '\n';
      for (int j=0; j<9; ++j) 
  	std::cout << "i[" << j << "] = " << static_cast<int>(str_field[k].i[j]) << '\n';
    }
  }
  
  return EXIT_SUCCESS;
}
