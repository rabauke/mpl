Info objects
============

Overview
--------

Info objects store key-value pairs of string type.  The semantics of these key-value pairs is defined by the MPI standard and by the employed MPI implementation.  See the MPI standard and the documentation of your MPI implementation for details. Info objects are used by some MPL to pass the key-value pairs to the underlying MPI implementation to improve performance or resource utilization.


Class documentation
-------------------

.. doxygenclass:: mpl::info

.. doxygenclass:: mpl::infos
