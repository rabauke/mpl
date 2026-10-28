.. _Design:

MPL-MPI interoperability
========================

MPL is written on the basis of the MPI standard, but hides MPI-specific
types and handles as internal implementation details. Applications may
provide custom MPI initialization and deinitialization as described below.


Custom initialization and deinitialization of the MPI environment
-----------------------------------------------------------------

MPL entirely hides the initialization and deinitialization of the MPI
environment. With MPL, there is not need to call ``MPI_Init`` or
``MPI_Finalize`` (or some equivalent function) manually.  This is a
direct consequence of how MPL manages some internal resources. The
deallocation of some resources (by calling the appropriate MPI
functions) must be postponed until programm shutdown, i.e., *after*
exiting from ``main``.

Hiding the initialization and deinitialization of the MPI environment
is convenient but can become a limiting factor when mixing MPI and MPL
or when come custom initialization of the MPI environment is needed.

In order to write your custom initialization and deinitialization code
wrapp the calls to ``MPI_Init`` and ``MPI_Finalize`` into the
constructor and the destructor of a custom class, e.g.:

.. code-block::

	class my_mpi_environment {
	public:
	  my_mpi_environment(int *argc, char ***argv) {
	    MPI_Init(argc, argv);
	  }

	  ~my_mpi_environment() {
	    MPI_Finalize();
	  }
	};

Then create a static object of this class on function scope (e.g., in
the scope of ``main``) and call this function before any MPL function.
When MPL tries to initialize the MPI environment, it is checked if it
has already been initialized before.  In this case, MPL will also not
deinitialize at program shutdown.  In stead the MPI environment will
be finalized by the provided custom destructor of the class sketched
above.
