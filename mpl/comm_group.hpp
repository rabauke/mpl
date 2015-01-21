#if !(defined MPL_COMM_GROUP_HPP)

#define MPL_COMM_GROUP_HPP

#include <mpi.h>
#include <type_traits>

namespace mpl {

  class group;
  class communicator;

  namespace environment {

    namespace detail {

      class env;
      
    }

  }
  
  //--------------------------------------------------------------------
  
  class group {
    MPI_Group gr;
  public:
    typedef enum { ident=MPI_IDENT, similar=MPI_SIMILAR, unequal=MPI_UNEQUAL } equality_type;
    group() {
      gr=MPI_GROUP_EMPTY;
    }
    group(const communicator &comm);  // define later
    ~group() {
      int result;
      MPI_Group_compare(gr, MPI_GROUP_EMPTY, &result);
      if (result!=MPI_IDENT)
	MPI_Group_free(&gr);
    }
    void operator=(const group &)=delete;
    int size() const {
      int result;
      MPI_Group_size(gr, &result);
      return result;
    }
    int rank() const {
      int result;
      MPI_Group_rank(gr, &result);
      return result;
    }
    bool operator==(const group &other) const {
      int result;
      MPI_Group_compare(gr, other.gr, &result);
      return result==MPI_IDENT;
    }
    bool operator!=(const group &other) const {
      int result;
      MPI_Group_compare(gr, other.gr, &result);
      return result!=MPI_IDENT;
    }
    equality_type compare(const group &other) const {
      int result;
      MPI_Group_compare(gr, other.gr, &result);
      return static_cast<equality_type>(result);
    }
    friend class communicator;
  };

  //--------------------------------------------------------------------

  class communicator {
    MPI_Comm comm;
  public:
    typedef enum { ident=MPI_IDENT, congruent=MPI_CONGRUENT, similar=MPI_SIMILAR, unequal=MPI_UNEQUAL } equality_type;
  private:
    communicator(MPI_Comm comm) : comm(comm) {
    }
  public:
    communicator(const communicator &other) {
      MPI_Comm_dup(other.comm, &comm);
    }
    ~communicator() {
      int result1, result2;
      MPI_Comm_compare(comm, MPI_COMM_WORLD, &result1);
      MPI_Comm_compare(comm, MPI_COMM_SELF, &result2);
      if (result1!=MPI_IDENT and result2!=MPI_IDENT)
	MPI_Comm_free(&comm);
    }
    void operator=(const communicator &)=delete;
    int size() const {
      int result;
      MPI_Comm_size(comm, &result);
      return result;
    }
    int rank() const {
      int result;
      MPI_Comm_rank(comm, &result);
      return result;
    }
    bool operator==(const communicator &other) const {
      int result;
      MPI_Comm_compare(comm, other.comm, &result);
      return result==MPI_IDENT;
    }
    bool operator!=(const communicator &other) const {
      int result;
      MPI_Comm_compare(comm, other.comm, &result);
      return result!=MPI_IDENT;
    }
    equality_type compare(const communicator &other) const {
      int result;
      MPI_Comm_compare(comm, other.comm, &result);
      return static_cast<equality_type>(result);
    }
    friend class group;
    friend class environment::detail::env;

    void abort(int err) const {
      MPI_Abort(comm, err);
    }

    // === blocking point to point =====================================

    // --- send ---
    template<typename T>
    void send(const T &data, int dest, int tag=0) const {
      MPI_Send(const_cast<T *>(&data), 1, 
	       datatype_traits<T>::get_datatype(), 
	       dest, tag, comm);
    }
    template<typename T>
    void send(const T *data, const layout<T> &l, int dest, int tag=0) const {
      MPI_Send(const_cast<T *>(data), 1, 
      	       datatype_traits<layout<T> >::get_datatype(l), 
      	       dest, tag, comm);
    }
    // --- bsend_size ---
    template<typename T>
    int bsend_size() const {
      int size;
      MPI_Pack_size(1, 
		    datatype_traits<T>::get_datatype(), 
		    comm, &size);
      return size+MPI_BSEND_OVERHEAD;
    }
    template<typename T>
    int bsend_size(const layout<T> &l) const {
      int size;
      MPI_Pack_size(1, 
		    datatype_traits<layout<T> >::get_datatype(l), 
		    comm, &size);
      return size+MPI_BSEND_OVERHEAD;
    }
    // --- bsend ---
    template<typename T>
    void bsend(const T &data, int dest, int tag=0) const {
      MPI_Bsend(const_cast<T *>(&data), 1, 
		datatype_traits<T>::get_datatype(), 
		dest, tag, comm);
    }
    template<typename T>
    void bsend(const T *data, const layout<T> &l, int dest, int tag=0) const {
      MPI_Bsend(const_cast<T *>(data), 1, 
		datatype_traits<layout<T> >::get_datatype(l), 
		dest, tag, comm);
    }
    // --- ssend ---
    template<typename T>
    void ssend(const T &data, int dest, int tag=0) const {
      MPI_Ssend(const_cast<T *>(&data), 1, 
		datatype_traits<T>::get_datatype(), 
		dest, tag, comm);
    }
    template<typename T>
    void ssend(const T *data, const layout<T> &l, int dest, int tag=0) const {
      MPI_Ssend(const_cast<T *>(data), 1, 
		datatype_traits<layout<T> >::get_datatype(l), 
		dest, tag, comm);
    }
    // --- rsend ---
    template<typename T>
    void rsend(const T &data, int dest, int tag=0) const {
      MPI_Rsend(const_cast<T *>(&data), 1, 
		datatype_traits<T>::get_datatype(), 
		dest, tag, comm);
    }
    template<typename T>
    void rsend(const T *data, const layout<T> &l, int dest, int tag=0) const {
      MPI_Rsend(const_cast<T *>(data), 1, 
		datatype_traits<layout<T> >::get_datatype(l), 
		dest, tag, comm);
    }
    // --- recv ---
    template<typename T>
    status recv(T &data, int source, int tag=0) const {
      status s;
      MPI_Recv(&data, 1, 
	       datatype_traits<T>::get_datatype(), 
	       source, tag, comm, reinterpret_cast<MPI_Status *>(&s));
      return s;
    }
    template<typename T>
    status recv(T *data, const layout<T> &l, int source, int tag=0) const {
      status s;
      MPI_Recv(data, 1, 
      	       datatype_traits<layout<T> >::get_datatype(l), 
      	       source, tag, comm, reinterpret_cast<MPI_Status *>(&s));
      return s;
    }
    // --- probe ---
    template<typename T>
    status probe(int source, int tag=0) const {
      status s;
      MPI_Probe(source, tag, comm, reinterpret_cast<MPI_Status *>(&s));
      return s;
    }
    // --- sendrecv ---
    template<typename T>
    status sendrecv(const T &senddata, int dest, int sendtag,
		    T &recvdata, int source, int recvtag) const {
      status s;
      MPI_Sendrecv(const_cast<T *>(&senddata), 1, 
		   datatype_traits<T>::get_datatype(), dest, sendtag,
		   &recvdata, 1,
		   datatype_traits<T>::get_datatype(), source, recvtag,
		   comm, reinterpret_cast<MPI_Status *>(&s));
      return s;
    }
    template<typename T>
    status sendrecv(const T *senddata, const layout<T> &sendl, int dest, int sendtag,
		    T *recvdata, const layout<T> &recvl, int source, int recvtag) const {
      status s;
      MPI_Sendrecv(const_cast<T *>(senddata), 1, 
		   datatype_traits<layout<T> >::get_datatype(sendl), dest, sendtag,
		   recvdata, 1,
		   datatype_traits<layout<T> >::get_datatype(recvl), source, recvtag,
		   comm, reinterpret_cast<MPI_Status *>(&s));
      return s;
    }
    // --- sendrecv_replace ---
    template<typename T>
    status sendrecv_replace(T &data, 
			    int dest, int sendtag, int source, int recvtag) const {
      status s;
      MPI_Sendrecv_replace(&data, 1, 
			   datatype_traits<T>::get_datatype(), 
			   dest, sendtag, source, recvtag,
			   comm, reinterpret_cast<MPI_Status *>(&s));
      return s;
    }
    template<typename T>
    status sendrecv_replace(T *data, const layout<T> &l, 
			    int dest, int sendtag, int source, int recvtag) const {
      status s;
      MPI_Sendrecv_replace(data, 1, 
			   datatype_traits<layout<T> >::get_datatype(l), 
			   dest, sendtag, source, recvtag,
			   comm, reinterpret_cast<MPI_Status *>(&s));
      return s;
    }

    // === nonblocking point to point ==================================

    // --- isend ---
    template<typename T>
    detail::irequest isend(const T &data, int dest, int tag=0) const {
      MPI_Request req;
      MPI_Isend(const_cast<T *>(&data), 1, 
		datatype_traits<T>::get_datatype(), 
		dest, tag, comm, &req);
      return detail::irequest(req);
    }
    template<typename T>
    detail::irequest isend(const T *data, const layout<T> &l, int dest, int tag=0) const {
      MPI_Request req;
      MPI_Isend(const_cast<T *>(data), 1, 
		datatype_traits<layout<T> >::get_datatype(l), 
		dest, tag, comm, &req);
      return detail::irequest(req);
    }
    // --- ibsend ---
    template<typename T>
    detail::irequest ibsend(const T &data, int dest, int tag=0) const {
      MPI_Request req;
      MPI_Ibsend(const_cast<T *>(&data), 1, 
		 datatype_traits<T>::get_datatype(), 
		 dest, tag, comm, &req);
      return detail::irequest(req);
    }
    template<typename T>
    detail::irequest ibsend(const T *data, const layout<T> &l, int dest, int tag=0) const {
      MPI_Request req;
      MPI_Ibsend(const_cast<T *>(data), 1, 
		 datatype_traits<layout<T> >::get_datatype(l), 
		 dest, tag, comm, &req);
      return detail::irequest(req);
    }
    // --- issend ---
    template<typename T>
    detail::irequest issend(const T &data, int dest, int tag=0) const {
      MPI_Request req;
      MPI_Issend(const_cast<T *>(&data), 1, 
		 datatype_traits<T>::get_datatype(), 
		 dest, tag, comm, &req);
      return detail::irequest(req);
    }
    template<typename T>
    detail::irequest issend(const T *data, const layout<T> &l, int dest, int tag=0) const {
      MPI_Request req;
      MPI_Issend(const_cast<T *>(data), 1, 
		 datatype_traits<layout<T> >::get_datatype(l), 
		 dest, tag, comm, &req);
      return detail::irequest(req);
    }
    // --- irsend ---
    template<typename T>
    detail::irequest irsend(const T &data, int dest, int tag=0) const {
      MPI_Request req;
      MPI_Irsend(const_cast<T *>(&data), 1, 
		 datatype_traits<T>::get_datatype(), 
		 dest, tag, comm, &req);
      return detail::irequest(req);
    }
    template<typename T>
    detail::irequest irsend(const T *data, const layout<T> &l, int dest, int tag=0) const {
      MPI_Request req;
      MPI_Irsend(const_cast<T *>(data), 1, 
		 datatype_traits<layout<T> >::get_datatype(l), 
		 dest, tag, comm, &req);
      return detail::irequest(req);
    }
    // --- irecv ---
    template<typename T>
    detail::irequest irecv(T &data, int source, int tag=0) const {
      MPI_Request req;
      MPI_Irecv(&data, 1, 
		datatype_traits<T>::get_datatype(), 
		source, tag, comm, &req);
      return detail::irequest(req);
    }
    template<typename T>
    detail::irequest irecv(T *data, const layout<T> &l, int source, int tag=0) const {
      MPI_Request req;
      MPI_Irecv(data, 1, 
		datatype_traits<layout<T> >::get_datatype(l), 
		source, tag, comm, &req);
      return detail::irequest(req);
    }
    // --- iprobe ---
    template<typename T>
    std::pair<bool, status> iprobe(int source, int tag=0) const {
      int result; 
      status s;
      MPI_Iprobe(source, tag, comm, &result, reinterpret_cast<MPI_Status *>(&s));
      return std::make_pair(static_cast<bool>(result), s);
    }

    // === persistent point to point ===================================

    // --- send_init ---
    template<typename T>
    detail::prequest send_init(const T &data, int dest, int tag=0) const {
      MPI_Request req;
      MPI_send_init(const_cast<T *>(&data), 1, 
		    datatype_traits<T>::get_datatype(), 
		    dest, tag, comm, &req);
      return detail::prequest(req);
    }
    template<typename T>
    detail::prequest send_init(const T *data, const layout<T> &l, int dest, int tag=0) const {
      MPI_Request req;
      MPI_Send_init(const_cast<T *>(data), 1, 
		    datatype_traits<layout<T> >::get_datatype(l), 
		    dest, tag, comm, &req);
      return detail::prequest(req);
    }
    // --- bsend_init ---
    template<typename T>
    detail::prequest bsend_init(const T &data, int dest, int tag=0) const {
      MPI_Request req;
      MPI_Bsend_init(const_cast<T *>(&data), 1, 
		     datatype_traits<T>::get_datatype(), 
		     dest, tag, comm, &req);
      return detail::prequest(req);
    }
    template<typename T>
    detail::prequest bsend_init(const T *data, const layout<T> &l, int dest, int tag=0) const {
      MPI_Request req;
      MPI_Bsend_init(const_cast<T *>(data), 1, 
		     datatype_traits<layout<T> >::get_datatype(l), 
		     dest, tag, comm, &req);
      return detail::prequest(req);
    }
    // --- ssend_init ---
    template<typename T>
    detail::prequest ssend_init(const T &data, int dest, int tag=0) const {
      MPI_Request req;
      MPI_Ssend_init(const_cast<T *>(&data), 1, 
		     datatype_traits<T>::get_datatype(), 
		     dest, tag, comm, &req);
      return detail::prequest(req);
    }
    template<typename T>
    detail::prequest ssend_init(const T *data, const layout<T> &l, int dest, int tag=0) const {
      MPI_Request req;
      MPI_Ssend_init(const_cast<T *>(data), 1, 
		     datatype_traits<layout<T> >::get_datatype(l), 
		     dest, tag, comm, &req);
      return detail::prequest(req);
    }
    // --- rsend_init ---
    template<typename T>
    detail::prequest rsend_init(const T &data, int dest, int tag=0) const {
      MPI_Request req;
      MPI_Rsend_init(const_cast<T *>(&data), 1, 
		     datatype_traits<T>::get_datatype(), 
		     dest, tag, comm, &req);
      return detail::prequest(req);
    }
    template<typename T>
    detail::prequest rsend_init(const T *data, const layout<T> &l, int dest, int tag=0) const {
      MPI_Request req;
      MPI_Rsend_init(const_cast<T *>(data), 1, 
		     datatype_traits<layout<T> >::get_datatype(l), 
		     dest, tag, comm, &req);
      return detail::prequest(req);
    }
    // --- recv_init ---
    template<typename T>
    detail::prequest recv_init(T &data, int source, int tag=0) const {
      MPI_Request req;
      MPI_Recv_init(&data, 1, 
		    datatype_traits<T>::get_datatype(), 
		    source, tag, comm, &req);
      return detail::prequest(req);
    }
    template<typename T>
    detail::prequest recv_init(T *data, const layout<T> &l, int source, int tag=0) const {
      MPI_Request req;
      MPI_Recv_init(data, 1, 
		    datatype_traits<layout<T> >::get_datatype(l), 
		    source, tag, comm, &req);
      return detail::prequest(req);
    }

    // === collective ==================================================

    // --- barrier ---
    void barrier() const {
      MPI_Barrier(comm);
    }
    // --- bcast ---
    template<typename T>
    void bcast(T &data, int root) const {
      MPI_Bcast(&data, 1, datatype_traits<T>::get_datatype(), root, comm);
    }
    template<typename T>
    void bcast(T *data, const layout<T> &l, int root) const {
      MPI_Bcast(data, 1, datatype_traits<layout<T> >::get_datatype(l), root, comm);
    }
    // --- gather ---
    template<typename T>
    void gather(const T &senddata, T *recvdata, int root) const {
      MPI_Gather(const_cast<T *>(&senddata), 1, datatype_traits<T>::get_datatype(),
		 recvdata, 1, datatype_traits<T>::get_datatype(),
		 root, comm);
    }
    template<typename T>
    void gather(const T *senddata, const layout<T> &sendl, 
		T *recvdata, const layout<T> &recvl, int root) const {
      MPI_Gather(const_cast<T *>(senddata), 1, datatype_traits<layout<T> >::get_datatype(sendl),
		 recvdata, 1, datatype_traits<layout<T> >::get_datatype(recvl),
		 root, comm);
    }
    template<typename T>
    void gather(T *data, const layout<T> &l, int root) const {
      MPI_Gather(data, 1, datatype_traits<layout<T> >::get_datatype(l),
		 MPI_IN_PLACE, 0, NULL,
		 root, comm);
    }
    // --- allgather ---
    template<typename T>
    void allgather(const T &senddata, T *recvdata) const {
      MPI_Allgather(const_cast<T *>(&senddata), 1, datatype_traits<T>::get_datatype(),
		    recvdata, 1, datatype_traits<T>::get_datatype(),
		    comm);
    }
    template<typename T>
    void allgather(const T *senddata, const layout<T> &sendl, 
		   T *recvdata, const layout<T> &recvl) const {
      MPI_Allgather(const_cast<T *>(senddata), 1, datatype_traits<T>::get_datatype(),
		    recvdata, 1, datatype_traits<T>::get_datatype(),
		    comm);
    }
    template<typename T>
    void allgather(T *data, const layout<T> &l) const {
      MPI_Allgather(data, 1, datatype_traits<layout<T> >::get_datatype(l),
		    MPI_IN_PLACE, 0, NULL,
		    comm);
    }
    // --- scatter ---
    template<typename T>
    void scatter(const T *senddata, T &recvdata, int root) const {
      MPI_Scatter(const_cast<T *>(senddata), 1, datatype_traits<T>::get_datatype(),
		  &recvdata, 1, datatype_traits<T>::get_datatype(),
		  root, comm);
    }
    template<typename T>
    void scatter(const T *senddata, const layout<T> &sendl, 
		 T *recvdata, const layout<T> &recvl, int root) const {
      MPI_Scatter(const_cast<T *>(senddata), 1, datatype_traits<layout<T> >::get_datatype(sendl),
		  recvdata, 1, datatype_traits<layout<T> >::get_datatype(recvl),
		  root, comm);
    }
    template<typename T>
    void scatter(T *data, const layout<T> &l, int root) const {
      MPI_Scatter(data, 1, datatype_traits<layout<T> >::get_datatype(l),
		  MPI_IN_PLACE, 0, NULL,
		  root, comm);
    }
    // --- alltoall ---
    template<typename T>
    void alltoall(const T *senddata, T *recvdata) const {
      MPI_Alltoall(const_cast<T *>(senddata), 1, datatype_traits<T>::get_datatype(),
		   recvdata, 1, datatype_traits<T>::get_datatype(),
		   comm);
    }
    template<typename T>
    void alltoall(const T *senddata, const layout<T> &sendl, 
		  T *recvdata, const layout<T> &recvl) const {
      MPI_Alltoall(const_cast<T *>(senddata), 1, datatype_traits<T>::get_datatype(),
		   recvdata, 1, datatype_traits<T>::get_datatype(),
		   comm);
    }
    template<typename T>
    void alltoall(T *data, const layout<T> &l) const {
      MPI_Alltoall(data, 1, datatype_traits<layout<T> >::get_datatype(l),
		   MPI_IN_PLACE, 0, NULL,
		   comm);
    }
    // --- reduce ---
    template<typename T, typename F>
    void reduce(const T &senddata, T &recvdata, F f, int root) const {
      static_assert(std::is_same<typename F::first_argument_type, typename F::second_argument_type>::value and
		    std::is_same<typename F::second_argument_type, typename F::result_type>::value and
		    std::is_same<typename F::result_type, T>::value, "argument type mismatch");
      static detail::op<F> functor;
      functor.f=f;      
      MPI_Reduce(const_cast<T *>(&senddata), &recvdata, 1, 
		 datatype_traits<T>::get_datatype(), functor.mpi_op, root, comm);
    }
    template<typename T, typename F>
    void reduce(T &data, F f, int root) const {
      static_assert(std::is_same<typename F::first_argument_type, typename F::second_argument_type>::value and
		    std::is_same<typename F::second_argument_type, typename F::result_type>::value and
		    std::is_same<typename F::result_type, T>::value, "argument type mismatch");
      static detail::op<F> functor;
      functor.f=f;      
      MPI_Reduce(&data, MPI_IN_PLACE, 1, 
		 datatype_traits<T>::get_datatype(), functor.mpi_op, root, comm);
    }
    template<typename T, typename F>
    void reduce(const T *senddata, T *recvdata, const contiguous_layout<T> &l, F f, int root) const {
      static_assert(std::is_same<typename F::first_argument_type, typename F::second_argument_type>::value and
		    std::is_same<typename F::second_argument_type, typename F::result_type>::value and
		    std::is_same<typename F::result_type, T>::value, "argument type mismatch");
      static detail::op<F> functor;
      functor.f=f;      
      MPI_Reduce(const_cast<T *>(senddata), recvdata, l.size(), 
		 datatype_traits<T>::get_datatype(), functor.mpi_op, root, comm);
    }
    template<typename T, typename F>
    void reduce(T *data, const contiguous_layout<T> &l, F f, int root) const {
      static_assert(std::is_same<typename F::first_argument_type, typename F::second_argument_type>::value and
		    std::is_same<typename F::second_argument_type, typename F::result_type>::value and
		    std::is_same<typename F::result_type, T>::value, "argument type mismatch");
      static detail::op<F> functor;
      functor.f=f;      
      MPI_Reduce(const_cast<T *>(data), MPI_IN_PLACE, l.size(), 
		 datatype_traits<T>::get_datatype(), functor.mpi_op, root, comm);
    }
    // --- allreduce ---
    template<typename T, typename F>
    void allreduce(const T &senddata, T &recvdata, F f) const {
      static_assert(std::is_same<typename F::first_argument_type, typename F::second_argument_type>::value and
		    std::is_same<typename F::second_argument_type, typename F::result_type>::value and
		    std::is_same<typename F::result_type, T>::value, "argument type mismatch");
      static detail::op<F> functor;
      functor.f=f;      
      MPI_Allreduce(const_cast<T *>(&senddata), &recvdata, 1, 
		    datatype_traits<T>::get_datatype(), functor.mpi_op, comm);
    }
    template<typename T, typename F>
    void allreduce(T &data, F f) const {
      static_assert(std::is_same<typename F::first_argument_type, typename F::second_argument_type>::value and
		    std::is_same<typename F::second_argument_type, typename F::result_type>::value and
		    std::is_same<typename F::result_type, T>::value, "argument type mismatch");
      static detail::op<F> functor;
      functor.f=f;      
      MPI_Allreduce(&data, MPI_IN_PLACE, 1, 
		    datatype_traits<T>::get_datatype(), functor.mpi_op, comm);
    }
    template<typename T, typename F>
    void allreduce(const T *senddata, T *recvdata, const contiguous_layout<T> &l, F f) const {
      static_assert(std::is_same<typename F::first_argument_type, typename F::second_argument_type>::value and
		    std::is_same<typename F::second_argument_type, typename F::result_type>::value and
		    std::is_same<typename F::result_type, T>::value, "argument type mismatch");
      static detail::op<F> functor;
      functor.f=f;      
      MPI_Allreduce(const_cast<T *>(senddata), recvdata, l.size(), 
		    datatype_traits<T>::get_datatype(), functor.mpi_op, comm);
    }
    template<typename T, typename F>
    void allreduce(T *data, const contiguous_layout<T> &l, F f) const {
      static_assert(std::is_same<typename F::first_argument_type, typename F::second_argument_type>::value and
		    std::is_same<typename F::second_argument_type, typename F::result_type>::value and
		    std::is_same<typename F::result_type, T>::value, "argument type mismatch");
      static detail::op<F> functor;
      functor.f=f;      
      MPI_Allreduce(const_cast<T *>(data), MPI_IN_PLACE, l.size(), 
		    datatype_traits<T>::get_datatype(), functor.mpi_op, comm);
    }
    //--- scan ---
    template<typename T, typename F>
    void scan(const T &senddata, T &recvdata, F f) const {
      static_assert(std::is_same<typename F::first_argument_type, typename F::second_argument_type>::value and
		    std::is_same<typename F::second_argument_type, typename F::result_type>::value and
		    std::is_same<typename F::result_type, T>::value, "argument type mismatch");
      static detail::op<F> functor;
      functor.f=f;
      MPI_Scan(const_cast<T *>(&senddata), &recvdata, 1, 
	       datatype_traits<T>::get_datatype(), functor.mpi_op, comm);
    }
    template<typename T, typename F>
    void scan(T &data, F f) const {
      static_assert(std::is_same<typename F::first_argument_type, typename F::second_argument_type>::value and
		    std::is_same<typename F::second_argument_type, typename F::result_type>::value and
		    std::is_same<typename F::result_type, T>::value, "argument type mismatch");
      static detail::op<F> functor;
      functor.f=f;      
      MPI_Scan(&data, MPI_IN_PLACE, 1, 
	       datatype_traits<T>::get_datatype(), functor.mpi_op, comm);
    }
    template<typename T, typename F>
    void scan(const T *senddata, T *recvdata, const contiguous_layout<T> &l, F f) const {
      static_assert(std::is_same<typename F::first_argument_type, typename F::second_argument_type>::value and
		    std::is_same<typename F::second_argument_type, typename F::result_type>::value and
		    std::is_same<typename F::result_type, T>::value, "argument type mismatch");
      static detail::op<F> functor;
      functor.f=f;      
      MPI_Scan(const_cast<T *>(senddata), recvdata, l.size(), 
	       datatype_traits<T>::get_datatype(), functor.mpi_op, comm);
    }
    template<typename T, typename F>
    void scan(T *data, const contiguous_layout<T> &l, F f) const {
      static_assert(std::is_same<typename F::first_argument_type, typename F::second_argument_type>::value and
		    std::is_same<typename F::second_argument_type, typename F::result_type>::value and
		    std::is_same<typename F::result_type, T>::value, "argument type mismatch");
      static detail::op<F> functor;
      functor.f=f;      
      MPI_Scan(const_cast<T *>(data), MPI_IN_PLACE, l.size(), 
	       datatype_traits<T>::get_datatype(), functor.mpi_op, comm);
    }
    //--- exscan ---
    template<typename T, typename F>
    void exscan(const T &senddata, T &recvdata, F f) const {
      static_assert(std::is_same<typename F::first_argument_type, typename F::second_argument_type>::value and
		    std::is_same<typename F::second_argument_type, typename F::result_type>::value and
		    std::is_same<typename F::result_type, T>::value, "argument type mismatch");
      static detail::op<F> functor;
      functor.f=f;      
      MPI_Exscan(const_cast<T *>(&senddata), &recvdata, 1, 
		 datatype_traits<T>::get_datatype(), functor.mpi_op, comm);
    }
    template<typename T, typename F>
    void exscan(T &data, F f) const {
      static_assert(std::is_same<typename F::first_argument_type, typename F::second_argument_type>::value and
		    std::is_same<typename F::second_argument_type, typename F::result_type>::value and
		    std::is_same<typename F::result_type, T>::value, "argument type mismatch");
      static detail::op<F> functor;
      functor.f=f;      
      MPI_Exscan(&data, MPI_IN_PLACE, 1, 
		 datatype_traits<T>::get_datatype(), functor.mpi_op, comm);
    }
    template<typename T, typename F>
    void exscan(const T *senddata, T *recvdata, const contiguous_layout<T> &l, F f) const {
      static_assert(std::is_same<typename F::first_argument_type, typename F::second_argument_type>::value and
		    std::is_same<typename F::second_argument_type, typename F::result_type>::value and
		    std::is_same<typename F::result_type, T>::value, "argument type mismatch");
      static detail::op<F> functor;
      functor.f=f;      
      MPI_Exscan(const_cast<T *>(senddata), recvdata, l.size(), 
		 datatype_traits<T>::get_datatype(), functor.mpi_op, comm);
    }
    template<typename T, typename F>
    void exscan(T *data, const contiguous_layout<T> &l, F f) const {
      static_assert(std::is_same<typename F::first_argument_type, typename F::second_argument_type>::value and
		    std::is_same<typename F::second_argument_type, typename F::result_type>::value and
		    std::is_same<typename F::result_type, T>::value, "argument type mismatch");
      static detail::op<F> functor;
      functor.f=f;      
      MPI_Exscan(const_cast<T *>(data), MPI_IN_PLACE, l.size(), 
		 datatype_traits<T>::get_datatype(), functor.mpi_op, comm);
    }
    
  };

  //--------------------------------------------------------------------

  inline group::group(const communicator &comm) {
    MPI_Comm_group(comm.comm, &gr);
  }

}

#endif
