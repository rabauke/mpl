# Data types
 \anchor data_types

With MPL processes can send and receive messages containing data of 
different data types. MPL requires some knowledge about the internal 
representation of this data. MPL supports

* all standard integer types, signed as well as unsigned, such as `int`,
  `unsigned int` etc.,
    
* the character types `char`, `signed char`, `unsigned char` as well as 
  the wide character types `wchar_t`, `char8_t` (if compiler supports 
  C++-20 features), `char16_t` and `char32_t`,

* the floating point types `float`, `double` and `long double`,

* the complex types `std::complex<float>`, `std::complex<double>` and
  `std::complex<long double>`,
  
* the Boolean type `bool`, 

* the type `std::byte` and

* enumeration types.

MPL would not be very limited if it would only support these elementary
data types. However, MPL comes also with some support for user-defined
data types. To be able to exchange data of custom types via a message
passing library. The message passing library must have some knowledge
about the internal representation of user-defined data types. Because
C++ has very limited type introspection capabilities, this knowledge
cannot be obtained automatically by the message passing library. Usually
information about the internal structure of user-defined types 
(structures and classes) has to be exposed explicitly to the message
passing library. Therefore, MPL supports message exchange of data where
information about the internal representation can be obtained
automatically and introduces a mechanism to expose the internal
representation of custom types to MPL if this is not possible.

The data types, where MPL can infer their internal representation, C 
arrays of constant size and the template classes
`std::array`, `std::pair` and `std::tuple` of the C++ Standard Template
Library. The only limitation is, that the C arrays as well as the 
mentioned STL template classes hold data elements of types that can be 
sent or received by MPL, e.g., the elementary types mentioned above. 
This rule can be applied recursively, which allows one to build quite 
complex data structures. This means, for example, one can send and 
receive data of type std::pair<int, double>, because `int` and
`double` can be sent or received. But also 
`std::array<std::pair<int, double>, 8>`, which represents 8 pairs of 
`int` and `double`, can be used in a message.

User-defined data structures usually come as structures or classes.
Provided that these classes hold only non-static non-const data members
of types, which MPL is able to send or receive, it is possible to expose
these data members to MPL via template specialization of the class
`struct_builder` such that messages containing objects of these classes
can be exchanged. Template specialization of the class struct_builder is
illustrated in the example program [`example_struct.cc`](struct_8cc-example.html). 
The specialized template has to derived from `base_struct_builder` and 
the internal data representation of the user-defined class is exposed 
to MPL in the constructor. 
