Data layouts
============

Overview
--------

MPL message exchange methods come in several overloaded variants with signatures of different complexity.  The most simple overloads allow to send or receive one object only, e.g., a single integer.  As sending and receiving single data items would be too limiting for a message passing library, MPL introduces the concept of data layouts. Data layouts specify the memory layout of a set of objects to be sent or received (similar to derived data types in MPI). The layout may be continuous, a strided vector etc.  The layouts on the sending and on the receiving sides need not to be identical but compatible, e.g., represent the same number of elements with the same types on both communication ends.  See section :doc:`examples/layouts` for some usage examples of layouts.

The MPL layout classes wrap MPI generalized data types into a flexible RAII interface and inherit their semantics.  See the MPI Standard for details.


Class documentation
-------------------

Data layout base class
^^^^^^^^^^^^^^^^^^^^^^

.. doxygenclass:: mpl::layout
   :allow-dot-graphs:


Null layout
^^^^^^^^^^^

.. doxygenclass:: mpl::null_layout
   :allow-dot-graphs:


Empty layout
^^^^^^^^^^^^

.. doxygenclass:: mpl::empty_layout
   :allow-dot-graphs:


Contiguous layout
^^^^^^^^^^^^^^^^^

.. doxygenclass:: mpl::contiguous_layout
   :allow-dot-graphs:


Vector layout
^^^^^^^^^^^^^

.. doxygenclass:: mpl::vector_layout
   :allow-dot-graphs:


Strided vector layout
^^^^^^^^^^^^^^^^^^^^^

.. doxygenclass:: mpl::strided_vector_layout
   :allow-dot-graphs:


Indexed layout
^^^^^^^^^^^^^^

.. doxygenclass:: mpl::indexed_layout
   :allow-dot-graphs:


Heterogeneous indexed layout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenclass:: mpl::hindexed_layout
   :allow-dot-graphs:


Indexed block layout
^^^^^^^^^^^^^^^^^^^^

.. doxygenclass:: mpl::indexed_block_layout


Heterogeneous indexed block layout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenclass:: mpl::hindexed_block_layout


Iterator layout
^^^^^^^^^^^^^^^

.. doxygenclass:: mpl::iterator_layout


Subarray layout
^^^^^^^^^^^^^^^

.. doxygenclass:: mpl::subarray_layout


Helper functions and classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenenum:: mpl::array_orders


Heterogeneous layout
^^^^^^^^^^^^^^^^^^^^

.. doxygenclass:: mpl::heterogeneous_layout


Helper functions and classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenclass:: mpl::absolute_data
.. doxygenclass:: mpl::absolute_data< const T * >
.. doxygenclass:: mpl::absolute_data< T * >
.. doxygenfunction:: mpl::make_absolute(const T *x, const layout<T> &l)
.. doxygenfunction:: mpl::make_absolute(T *x, const layout<T> &l)


Layouts
^^^^^^^

.. doxygenclass:: mpl::layouts


Contiguous layouts
^^^^^^^^^^^^^^^^^^

.. doxygenclass:: mpl::contiguous_layouts
