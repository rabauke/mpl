Error handling
==============

MPL performs some consistency checks when the macro ``MPL_DEBUG`` has been defined.  In case of an error, an exception may be thrown with an object having one of the following types.

.. doxygenclass:: mpl::error
.. doxygenclass:: mpl::invalid_rank
.. doxygenclass:: mpl::invalid_tag
.. doxygenclass:: mpl::invalid_size
.. doxygenclass:: mpl::invalid_count
.. doxygenclass:: mpl::invalid_layout
.. doxygenclass:: mpl::invalid_dim
.. doxygenclass:: mpl::invalid_datatype_bound
.. doxygenclass:: mpl::invalid_argument
.. doxygenclass:: mpl::mpi_error
.. doxygenclass:: mpl::io_failure
.. doxygenclass:: mpl::request_error

MPI errors
----------

When a checked MPI call returns an error, MPL throws ``mpl::mpi_error`` (or a subclass), carrying the MPI error code (``error_code()``), class (``error_class()``) and message (``what()``).  MPL argument exceptions such as ``mpl::invalid_rank`` derive directly from ``mpl::error`` and carry no MPI code.  Classification is by MPI error class: recognized I/O classes become ``mpl::io_failure`` even outside a file operation, while other classes returned by file operations remain ``mpl::mpi_error``.

When multiple-completion calls report per-request errors, ``mpl::request_error`` lists failed and successful indices reported by that call.  These lists are not a complete partition of pool state: unlisted slots are not necessarily pending, and null or inactive persistent slots can produce empty successful statuses.  Statuses of reported completions are stored in the pool; per-request error fields are meaningful on the prescribed ``MPI_ERR_IN_STATUS`` path, not an unsuccessful ``testall`` poll.

MPL installs ``MPI_ERRORS_RETURN`` on communicators it creates or wraps.  With external MPI initialization, ``MPI_COMM_WORLD`` and ``MPI_COMM_SELF`` are changed when the MPL environment is first accessed, not necessarily by earlier datatype or info operations.  Borrowed ``mpi_communicator`` and ``mpi_inter_communicator`` adapters change the original handle's handler immediately, do not restore it, and do not free the handle.  Other MPI calls, including file and objectless operations, may use different handlers.  A fatal handler can abort before MPI returns, so MPL cannot convert that failure into an exception; install an appropriate returning handler when needed.

.. note::
   Error reporting does not guarantee that execution can continue after an MPI error.  An exception on one process in a collective operation may leave other processes blocked.  Destructors do not throw; cleanup and finalization errors are ignored.
