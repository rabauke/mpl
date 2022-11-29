Iterators
=========

Sends and receives containers (``std::vector``, ``std::list``, etc.) given by their iterators. The types of the array elements must be suited for communication. On the receiving side, there must be sufficient preallocated memory, i.e., sending and receiving containers must be of the same size. Furthermore, tracking the address of a dereferenced iterator must result a non-const pointer on the receiving side.

.. literalinclude:: ../../../examples/iterators.cc
   :language: c++
