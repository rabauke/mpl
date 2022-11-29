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
