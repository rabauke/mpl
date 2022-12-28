Auxiliary functions and classes
===============================

Integer types
-------------

.. doxygentypedef:: mpl::size_t
.. doxygentypedef:: mpl::ssize_t


Ranks
-----

.. doxygenclass:: mpl::ranks


Types for probing messages
--------------------------

The following types are used in the context of probing messages. See section :doc:`examples/probe` for an example.

.. doxygentypedef:: mpl::message_t
.. doxygenclass:: mpl::status_t
.. doxygenstruct:: mpl::mprobe_status


Memory displacements
--------------------

The ``mpl::displacements`` class is used in the context of various collective communication operations that send and/or receive an amount of data that varies over the set of participating processes.

.. doxygenclass:: mpl::displacements


Requests
--------

Test status enum
^^^^^^^^^^^^^^^^

.. doxygenenum:: mpl::test_result


Non-blocking communication requests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenclass:: mpl::irequest
.. doxygenclass:: mpl::irequest_pool


Persistent communication requests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenclass:: mpl::prequest
.. doxygenclass:: mpl::prequest_pool


Command-line arguments
----------------------

.. doxygenclass:: mpl::command_line

.. doxygenclass:: mpl::command_lines
