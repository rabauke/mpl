#if !(defined MPL_OPERATOR_HPP)

#define MPL_OPERATOR_HPP

#include <mpi.h>

namespace mpl {

  namespace detail {
    
    template<typename F>
    class op : public F {
    public:
      typedef F functor;
      typedef typename F::first_argument_type first_argument_type;
      typedef typename F::second_argument_type second_argument_type;
      typedef typename F::result_type result_type;
      static F f;
      static void apply(void *invec, void *inoutvec, int *len, 
			MPI_Datatype *datatype) {
	first_argument_type *i1=reinterpret_cast<first_argument_type *>(invec);
	second_argument_type *i2=reinterpret_cast<second_argument_type *>(inoutvec);
	for (int i=0, i_end=*len; i<i_end; ++i, ++i1, ++i2)
	  *i2=f(*i1, *i2);
      }
      MPI_Op mpi_op;
      op() {
	MPI_Op_create(op::apply, false, &mpi_op);
      }
      ~op() {
	MPI_Op_free(&mpi_op);
      }
    };

    template<typename F>
    F op<F>::f;
  
  }

}

#endif
