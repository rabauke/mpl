.. _Design:

Library design principles
=========================

MPL is designed on the basis of the following

Resource management
-------------------

In a typical MPI program, a number of resources, e.g., communicators,
custom data types etc., must be management.  Allocation and deallocation
of such resources must be done manually by explicitly calling the
respective allocation and deallocation function.  This is error-prone,
may lead to resource leaks and requires a lot of boilerplate code.

Therefore, MPL applies the principle of "resource acquisition is
initialization" (RAII) and wraps all resources in custom class types.
Resources are allocated in a constructor and automatically dealocated
when a resource object goes out of scope by the destructor.  In contrast
to MPI handles, all resource classes have a value-semantics. This means,
when a resource object is copies in to another one, then a new resource
is created and the two resource objects manage different independent
resources.


Custom data types
-----------------

Custom data types are one of the most versatile features of MPI.  With
custom data types, it is possible to write very well structured
code by hiding details of the complex communication pattern in
well-designed custom data types.  Therefore, MPL makes it easy to
create and use custom data types.  These are called layouts in MPL.
