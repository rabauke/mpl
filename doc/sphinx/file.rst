File
====

Overview
--------

Parallel file operations are implemented via the ``mpl::file`` class.  It offers various read and write modalities (collective and non-collective, blocking and non-blocking etc.) by closely following the MPI standard.  See the MPI standard for a detailed description of the semantics of the various i/o operations.


Class documentation
-------------------

File class
^^^^^^^^^^

.. doxygenclass:: mpl::file


File access mode operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: mpl::operator|(file::access_mode, file::access_mode)
.. doxygenfunction:: mpl::operator|=(file::access_mode &, file::access_mode)
.. doxygenfunction:: mpl::operator&(file::access_mode, file::access_mode)
.. doxygenfunction:: mpl::operator&=(file::access_mode &, file::access_mode)


Error handling
^^^^^^^^^^^^^^

Methods of ``mpl::file`` class may throw an exception of the type ``mpl::io_failure`` in the case of run-time i/o failures.  Thus, file operations should be wrapped into a ``try`` block and possible exceptions should be caught in a matching ``catch`` clause as demonstrated in the following example:

.. literalinclude:: file_error_handling.cc
   :language: c++
