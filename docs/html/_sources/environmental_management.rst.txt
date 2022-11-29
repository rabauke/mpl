Environmental management
========================

MPL provides various functions for querying characteristics of the computational environment.


Current processor name
----------------------

.. doxygenfunction:: mpl::environment::processor_name


Predefined communicators
------------------------

.. doxygenfunction:: mpl::environment::comm_world
.. doxygenfunction:: mpl::environment::comm_self


Threading support
-----------------

.. doxygenfunction:: mpl::environment::is_thread_main
.. doxygenfunction:: mpl::environment::threading_mode
.. doxygenenum:: mpl::threading_modes


Time
----

.. doxygenfunction:: mpl::environment::wtime_is_global
.. doxygenfunction:: mpl::environment::wtime
.. doxygenfunction:: mpl::environment::wtick


Management of buffers for buffered send operations
--------------------------------------------------

.. doxygenfunction:: mpl::environment::buffer_attach
.. doxygenfunction:: mpl::environment::buffer_detach
.. doxygenclass:: mpl::bsend_buffer
