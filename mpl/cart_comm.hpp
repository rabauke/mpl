#if !(defined MPL_CART_COMM_HPP)

#define MPL_CART_COMM_HPP

#include <mpi.h>
#include <algorithm>
#include <vector>
#include <tuple>

namespace mpl {

  struct shift_ranks {
    int source, dest;
  };
  
  //--------------------------------------------------------------------

  class cart_communicator : public communicator {
  public:
    class parameter {
      std::vector<int> dims_, periodic_;
    public:
      typedef std::vector<int>::size_type size_type;
      parameter(std::initializer_list<std::pair<int, bool>> list) {
	for (const std::pair<int, bool> &i : list) 
	  add(i.first, i.second);
      }
      void add(int dim, bool p) {
	dims_.push_back(dim);
	periodic_.push_back(p);
      }
      int dims(size_type i) const {
	return dims_[i];
      }
      bool periodic(size_type i) const {
	return periodic_[i];
      }
      friend class cart_communicator;
      friend parameter dims_create(int, parameter);
    };
    cart_communicator(const communicator &old_comm,
		      const parameter &par,
		      bool reorder=true) {
      MPI_Cart_create(old_comm.comm, par.dims_.size(), par.dims_.data(), par.periodic_.data(), reorder, &comm);
    }
    cart_communicator(const cart_communicator &old_comm,
		      const std::vector<int> &remain_dims) {
      MPI_Cart_sub(old_comm.comm, remain_dims.data(), &comm);
    }
    int dim() const {
      int ndims;
      MPI_Cartdim_get(comm, &ndims);
      return ndims;
    }
    int rank(const std::vector<int> &coords) const {
      int rank_;
      MPI_Cart_rank(comm, coords.data(), &rank_);
      return rank_;
    }
    std::vector<int> coords(int rank) const {
      std::vector<int> coords_(dim());
      MPI_Cart_coords(comm, rank, coords_.size(), coords_.data());
      return coords_;
    }
    std::vector<int> coords() const {
      int ndims(dim());
      std::vector<int> dims_(ndims), periodic_(ndims), coords_(ndims);
      MPI_Cart_get(comm, ndims, dims_.data(), periodic_.data(), coords_.data());
      return coords_;
    }
    std::vector<int> dims() const {
      int ndims(dim());
      std::vector<int> dims_(ndims), periodic_(ndims), coords_(ndims);
      MPI_Cart_get(comm, ndims, dims_.data(), periodic_.data(), coords_.data());
      return dims_;
    }
    std::vector<int> periodic() const {
      int ndims(dim());
      std::vector<int> dims_(ndims), periodic_(ndims), coords_(ndims);
      MPI_Cart_get(comm, ndims, dims_.data(), periodic_.data(), coords_.data());
      return periodic_;
    }
    shift_ranks shift(int direction, int disp) const {
      int rank_source, rank_dest;
      MPI_Cart_shift(comm, direction, disp, &rank_source, &rank_dest);
      return {rank_source, rank_dest};
    }
    // === neighbour collective ========================================
    // === neighbour allgather ===
    // === get a signle value from each neighbour and store in contiguous memory 
    // --- blocking neighbour allgather ---
    template<typename T>
    void neighbour_allgather(const T &senddata, T *recvdata) const {
#if defined MPL_DEBUG
#endif
      MPI_Neighbor_allgather(*senddata, 1, datatype_traits<T>::get_datatype(),
			     recvdata, 1, datatype_traits<T>::get_datatype(),
			     comm);
    }
    template<typename T>
    void neighbour_allgather(const T *senddata, const layout<T> &sendl, 
			     T *recvdata, const layout<T> &recvl) const {
      MPI_Neighbour_allgather(senddata, 1, datatype_traits<layout<T> >::get_datatype(sendl),
			      recvdata, 1, datatype_traits<layout<T> >::get_datatype(recvl),
			      comm);
    }
    // --- nonblocking neighbour allgather ---
    template<typename T>
    detail::irequest ineighbour_allgather(const T &senddata, T *recvdata) const {
#if defined MPL_DEBUG
#endif
      MPI_Request req;
      MPI_Ineighbour_allgather(&senddata, 1, datatype_traits<T>::get_datatype(),
			       recvdata, 1, datatype_traits<T>::get_datatype(),
			       comm, &req);
      return detail::irequest(req);
    }
    template<typename T>
    detail::irequest ineighbour_allgather(const T *senddata, const layout<T> &sendl, 
				T *recvdata, const layout<T> &recvl) const {
      MPI_Request req;
      MPI_Ineighbour_allgather(senddata, 1, datatype_traits<layout<T> >::get_datatype(sendl),
			       recvdata, 1, datatype_traits<layout<T> >::get_datatype(recvl),
			       comm, &req);
      return detail::irequest(req);
    }
    // === get varying amount of data from each neighbour and stores in noncontiguous memory 
    // --- blocking neighbour allgather ---
    template<typename T>
    void neighbour_allgatherv(const T *senddata, int sendcount,
			      T *recvdata, const counts &recvcounts, const displacements &displs) const {
#if defined MPL_DEBUG
#endif
      MPI_Neigbour_allgatherv(senddata, sendcount, datatype_traits<T>::get_datatype(),
			      recvdata, recvcounts(), displs(), datatype_traits<T>::get_datatype(), 
			      comm);
    }
    template<typename T>
    void neighbur_allgatherv(const T *senddata, const layout<T> &sendl, int sendcount, 
			     T *recvdata, const layout<T> &recvl, const counts &recvcounts, const displacements &displs) const {
#if defined MPL_DEBUG
#endif
      MPI_Neighbour_allgatherv(senddata, sendcount, datatype_traits<layout<T> >::get_datatype(sendl),
			       recvdata, recvcounts(), displs(), datatype_traits<layout<T> >::get_datatype(recvl),
			       comm);
    }
    // --- nonblocking neighbour allgather ---
    template<typename T>
    detail::irequest ineighbour_allgatherv(const T *senddata, int sendcount,
					   T *recvdata, const counts &recvcounts, const displacements &displs) const {
#if defined MPL_DEBUG
#endif
      MPI_Request req;
      MPI_Ineighbour_allgatherv(senddata, sendcount, datatype_traits<T>::get_datatype(),
				recvdata, recvcounts(), displs(), datatype_traits<T>::get_datatype(), 
				comm, &req);
      return detail::irequest(req);
    }
    template<typename T>
    detail::irequest ineughbour_allgatherv(const T *senddata, const layout<T> &sendl, int sendcount, 
					   T *recvdata, const layout<T> &recvl, const counts &recvcounts, const displacements &displs) const {
#if defined MPL_DEBUG
#endif
      MPI_Request req;
      MPI_Ineighbour_allgatherv(senddata, sendcount, datatype_traits<layout<T> >::get_datatype(sendl),
				recvdata, recvcounts(), displs(), datatype_traits<layout<T> >::get_datatype(recvl),
				comm, &req);
      return detail::irequest(req);
    }
    // === neighbour all-to-all ===
    // === each rank sends a signle value to each neighbour
    // --- blocking neighbour all-to-all ---
    template<typename T>
#if defined MPL_DEBUG
#endif
    void neighbour_alltoall(const T *senddata, T *recvdata) const {
      MPI_Neighbour_alltoall(senddata, 1, datatype_traits<T>::get_datatype(),
			     recvdata, 1, datatype_traits<T>::get_datatype(),
			     comm);
    }
    template<typename T>
    void neighbour_alltoall(const T *senddata, const layout<T> &sendl, 
			    T *recvdata, const layout<T> &recvl) const {
#if defined MPL_DEBUG
#endif
      MPI_Neighbour_alltoall(senddata, 1, datatype_traits<T>::get_datatype(),
			     recvdata, 1, datatype_traits<T>::get_datatype(),
			     comm);
    }
    // --- nonblocking neighbour all-to-all ---
    template<typename T>
    detail::irequest ineighbour_alltoall(const T *senddata, T *recvdata) const {
#if defined MPL_DEBUG
#endif
      MPI_Request req;
      MPI_Ineighbour_alltoall(senddata, 1, datatype_traits<T>::get_datatype(),
			      recvdata, 1, datatype_traits<T>::get_datatype(),
			      comm, &req);
      return detail::irequest(req);
    }
    template<typename T>
    detail::irequest ineighbour_alltoall(const T *senddata, const layout<T> &sendl, 
					 T *recvdata, const layout<T> &recvl) const {
#if defined MPL_DEBUG
#endif
      MPI_Request req;
      MPI_Ineighbour_alltoall(senddata, 1, datatype_traits<T>::get_datatype(),
			      recvdata, 1, datatype_traits<T>::get_datatype(),
			      comm, &req);
      return detail::irequest(req);
    }
    // === each rank sends a varying number of values to each neighbour
    // --- blocking neighbour all-to-all ---
    template<typename T>
    void neighbour_alltoallv(const T *senddata, const counts &sendcounts, const displacements &senddispls,
			     T *recvdata, const counts &recvcounts, const displacements &recvdispls) const {
#if defined MPL_DEBUG
#endif
      MPI_Neighbour_altoallv(senddata, sendcounts(), senddispls(), datatype_traits<T>::get_datatype(),
			     recvdata, recvcounts(), recvdispls(), datatype_traits<T>::get_datatype(), 
			     comm);
    }
    template<typename T>
    void neighbour_alltoallv(const T *senddata, const layout<T> &sendl, const counts &sendcounts, const displacements &senddispls,
			     T *recvdata, const layout<T> &recvl, const counts &recvcounts, const displacements &recvdispls) const {
      MPI_Neighbour_altoallv(senddata, sendcounts(), senddispls(), datatype_traits<layout<T> >::get_datatype(sendl),
			     recvdata, recvcounts(), recvdispls(), datatype_traits<layout<T> >::get_datatype(recvl),
			     comm);
    }
    // --- non-blocking neighbour all-to-all ---
    template<typename T>
    detail::irequest ineighbour_alltoallv(const T *senddata, const counts &sendcounts, const displacements &senddispls,
					  T *recvdata, const counts &recvcounts, const displacements &recvdispls) const {
#if defined MPL_DEBUG
#endif
      MPI_Request req;
      MPI_Ineighbour_altoallv(senddata, sendcounts(), senddispls(), datatype_traits<T>::get_datatype(),
			      recvdata, recvcounts(), recvdispls(), datatype_traits<T>::get_datatype(), 
			      comm, &req);
      return detail::irequest(req);
    }
    template<typename T>
    detail::irequest ineighbour_alltoallv(const T *senddata, const layout<T> &sendl, const counts &sendcounts, const displacements &senddispls,
					  T *recvdata, const layout<T> &recvl, const counts &recvcounts, const displacements &recvdispls) const {
#if defined MPL_DEBUG
#endif
      MPI_Request req;
      MPI_Ineighbour_altoallv(senddata, sendcounts(), senddispls(), datatype_traits<layout<T> >::get_datatype(sendl),
			      recvdata, recvcounts(), recvdispls(), datatype_traits<layout<T> >::get_datatype(recvl),
			      comm, &req);
      return detail::irequest(req);
    }
    // === each rank sends a varying number of values to each neighbor with possibly different layouts
    // --- blocking neighbour all-to-all ---
    template<typename T>
    void neighbour_alltoallw(const T *senddata, const layouts<T> &sendl, const counts &sendcounts, const displacements &senddispls, 
			     T *recvdata, const layouts<T> &recvl, const counts &recvcounts, const displacements &recvdispls) const {
#if defined MPL_DEBUG
#endif
      MPI_Neighbour_alltoallw(senddata, sendcounts(), senddispls(), reinterpret_cast<const MPI_Datatype *>(sendl()), 
			      recvdata, recvcounts(), recvdispls(), reinterpret_cast<const MPI_Datatype *>(recvl()), 
			      comm);
    }
    // --- non-blocking neighbour all-to-all ---
    template<typename T>
    detail::irequest ineighbour_alltoallw(const T *senddata, const layouts<T> &sendl, const counts &sendcounts, const displacements &senddispls, 
					  T *recvdata, const layouts<T> &recvl, const counts &recvcounts, const displacements &recvdispls) const {
#if defined MPL_DEBUG
#endif
      MPI_Request req;
      MPI_Ineighbour_alltoallw(senddata, sendcounts(), senddispls(), reinterpret_cast<const MPI_Datatype *>(sendl()), 
			       recvdata, recvcounts(), recvdispls(), reinterpret_cast<const MPI_Datatype *>(recvl()), 
			       comm, &req);
      return detail::irequest(req);
    }
  };

  //--------------------------------------------------------------------

  cart_communicator::parameter dims_create(int size, cart_communicator::parameter par) {
    MPI_Dims_create(size, par.dims_.size(), par.dims_.data());
    return par;
  }

}

#endif
