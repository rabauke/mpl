Reduction operations
====================

Overview
--------

MPL supports various communication operations that perform a reduction operation over all processes within a communicator.  The reduction operation is passed as an argument to the communicator class method that realizes to communication operation.  The reduction operation can be given by a user-defined two-arguments functor class, a lambda function, e.g.,

.. code:: c++

   comm.reduce([](auto a, auto b) { return a + b; }, ...);

or by any of the template classes as documented below.


Class documentation
-------------------

Maximum
^^^^^^^

Perform the reduction operation

.. math::

   y = \max (x_1, x_2)

.. doxygenstruct:: mpl::max


Minimum
^^^^^^^

Perform the reduction operation

.. math::

   y = \min (x_1, x_2)

.. doxygenstruct:: mpl::min


Addition
^^^^^^^^

Perform the reduction operation

.. math::

   y = x_1 + x_2

.. doxygenstruct:: mpl::plus


Multiplication
^^^^^^^^^^^^^^

Perform the reduction operation

.. math::

   y = x_1 \cdot x_2

.. doxygenstruct:: mpl::multiplies


Logical conjunction
^^^^^^^^^^^^^^^^^^^

Perform the reduction operation

.. math::

   y = x_1 \land x_2

.. doxygenstruct:: mpl::logical_and


Logical disjunction
^^^^^^^^^^^^^^^^^^^

Perform the reduction operation

.. math::

   y = x_1 \lor x_2

.. doxygenstruct:: mpl::logical_or


Exclusive disjunction
^^^^^^^^^^^^^^^^^^^^^

Perform the reduction operation

.. math::

   y = x_1 \oplus x_2

.. doxygenstruct:: mpl::logical_xor


Bitwise and
^^^^^^^^^^^

Perform for integer arguments the bitwise reduction operation

.. math::

   y = x_1 \land x_2

.. doxygenstruct:: mpl::bit_and


Bitwise or
^^^^^^^^^^

Perform for integer arguments the bitwise reduction operation

.. math::

   y = x_1 \lor x_2

.. doxygenstruct:: mpl::bit_or


Bitwise exclusive-or
^^^^^^^^^^^^^^^^^^^^

Perform for integer arguments the bitwise reduction operation

.. math::

   y = x_1 \oplus x_2

.. doxygenstruct:: mpl::bit_xor


Operator traits
^^^^^^^^^^^^^^^

The application of some reduction operations can be performed more efficiently by exploiting the commutativity properties of the employed reduction operation.  Partial template specializations of the class ``mpl::op_traits`` provide information about the commutativity properties of the reduction operation.  Users may provide further user-defined specializations of ``mpl::op_traits`` for user-defined operators.

.. doxygenstruct:: mpl::op_traits
.. doxygenstruct:: mpl::op_traits< max< T > >
.. doxygenstruct:: mpl::op_traits< min< T > >
.. doxygenstruct:: mpl::op_traits< plus< T > >
.. doxygenstruct:: mpl::op_traits< multiplies< T > >
.. doxygenstruct:: mpl::op_traits< logical_and< T > >
.. doxygenstruct:: mpl::op_traits< logical_or< T > >
.. doxygenstruct:: mpl::op_traits< logical_xor< T > >
.. doxygenstruct:: mpl::op_traits< bit_and< T > >
.. doxygenstruct:: mpl::op_traits< bit_or< T > >
.. doxygenstruct:: mpl::op_traits< bit_xor< T > >
