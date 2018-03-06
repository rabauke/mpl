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
    class sizes {
      std::vector<int> dims_, periodic_;
    public:
      typedef std::vector<int>::size_type size_type;

      sizes(std::initializer_list<std::pair<int, bool>> list) {
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
        return periodic_[i]!=0;
      }

      friend class cart_communicator;

      friend sizes dims_create(int, sizes);
    };

    cart_communicator()=default;

    cart_communicator(const communicator &old_comm,
                      const sizes &par,
                      bool reorder=true) {
      MPI_Cart_create(old_comm.comm, par.dims_.size(), par.dims_.data(), par.periodic_.data(), reorder, &comm);
    }

    cart_communicator(const cart_communicator &old_comm,
                      const std::vector<int> &remain_dims) {
      MPI_Cart_sub(old_comm.comm, remain_dims.data(), &comm);
    }

    cart_communicator(cart_communicator &&other) noexcept {
      comm=other.comm;
      other.comm=MPI_COMM_SELF;
    }

    void operator=(const cart_communicator &)= delete;

    cart_communicator &operator=(cart_communicator &&other) {
      if (this!=&other) {
        int result1, result2;
        MPI_Comm_compare(comm, MPI_COMM_WORLD, &result1);
        MPI_Comm_compare(comm, MPI_COMM_SELF, &result2);
        if (result1!=MPI_IDENT and result2!=MPI_IDENT)
          MPI_Comm_free(&comm);
        comm=other.comm;
        other.comm=MPI_COMM_SELF;
      }
      return *this;
    }

    int dim() const {
      int ndims;
      MPI_Cartdim_get(comm, &ndims);
      return ndims;
    }

    using communicator::rank;

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
      return { rank_source, rank_dest };
    }

    // === neighbour collective ========================================
    // === neighbour allgather ===
    // === get a signle value from each neighbour and store in contiguous memory
    // --- blocking neighbour allgather ---
    template<typename T>
    void neighbour_allgather(const T &senddata, T *recvdata) const {
      MPI_Neighbor_allgather(*senddata, 1, datatype_traits<T>::get_datatype(),
                             recvdata, 1, datatype_traits<T>::get_datatype(),
                             comm);
    }

    template<typename T>
    void neighbour_allgather(const T *senddata, const layout <T> &sendl,
                             T *recvdata, const layout <T> &recvl) const {
      MPI_Neighbor_allgather(senddata, 1, datatype_traits<layout<T>>::get_datatype(sendl),
                             recvdata, 1, datatype_traits<layout<T>>::get_datatype(recvl),
                             comm);
    }

    // --- nonblocking neighbour allgather ---
    template<typename T>
    irequest ineighbour_allgather(const T &senddata, T *recvdata) const {
      MPI_Request req;
      MPI_Ineighbor_allgather(&senddata, 1, datatype_traits<T>::get_datatype(),
                              recvdata, 1, datatype_traits<T>::get_datatype(),
                              comm, &req);
      return irequest(req);
    }

    template<typename T>
    irequest ineighbour_allgather(const T *senddata, const layout <T> &sendl,
                                  T *recvdata, const layout <T> &recvl) const {
      MPI_Request req;
      MPI_Ineighbor_allgather(senddata, 1, datatype_traits<layout<T>>::get_datatype(sendl),
                              recvdata, 1, datatype_traits<layout<T>>::get_datatype(recvl),
                              comm, &req);
      return irequest(req);
    }

    // === get varying amount of data from each neighbour and stores in noncontiguous memory
    // --- blocking neighbour allgather ---
    template<typename T>
    void neighbour_allgatherv(const T *senddata, const layout <T> &sendl,
                              T *recvdata, const layouts <T> &recvls, const displacements &recvdispls) const {
      int N(recvdispls.size());
      displacements senddispls(N);
      layouts<T> sendls(N, sendl);
      alltoallv(senddata, sendls, senddispls,
                recvdata, recvls, recvdispls);
    }

    // --- nonblocking neighbour allgather ---
    template<typename T>
    irequest ineighbour_allgatherv(const T *senddata, const layout <T> &sendl,
                                   T *recvdata, const layouts <T> &recvls, const displacements &recvdispls) const {
      int N(recvdispls.size());
      displacements senddispls(N);
      layouts<T> sendls(N, sendl);
      return ialltoallv(senddata, sendls, senddispls,
                        recvdata, recvls, recvdispls);
    }

    // === neighbour all-to-all ===
    // === each rank sends a signle value to each neighbour
    // --- blocking neighbour all-to-all ---
    template<typename T>
    void neighbour_alltoall(const T *senddata, T *recvdata) const {
      MPI_Neighbor_alltoall(senddata, 1, datatype_traits<T>::get_datatype(),
                            recvdata, 1, datatype_traits<T>::get_datatype(),
                            comm);
    }

    template<typename T>
    void neighbour_alltoall(const T *senddata, const layout <T> &sendl,
                            T *recvdata, const layout <T> &recvl) const {
      MPI_Neighbor_alltoall(senddata, 1, datatype_traits<T>::get_datatype(),
                            recvdata, 1, datatype_traits<T>::get_datatype(),
                            comm);
    }

    // --- nonblocking neighbour all-to-all ---
    template<typename T>
    irequest ineighbour_alltoall(const T *senddata, T *recvdata) const {
      MPI_Request req;
      MPI_Ineighbor_alltoall(senddata, 1, datatype_traits<T>::get_datatype(),
                             recvdata, 1, datatype_traits<T>::get_datatype(),
                             comm, &req);
      return irequest(req);
    }

    template<typename T>
    irequest ineighbour_alltoall(const T *senddata, const layout <T> &sendl,
                                 T *recvdata, const layout <T> &recvl) const {
      MPI_Request req;
      MPI_Ineighbor_alltoall(senddata, 1, datatype_traits<T>::get_datatype(),
                             recvdata, 1, datatype_traits<T>::get_datatype(),
                             comm, &req);
      return irequest(req);
    }

    // === each rank sends a varying number of values to each neighbor with possibly different layouts
    // --- blocking neighbour all-to-all ---
    template<typename T>
    void neighbour_alltoallv(const T *senddata, const layouts <T> &sendl, const displacements &senddispls,
                             T *recvdata, const layouts <T> &recvl, const displacements &recvdispls) const {
      std::vector<int> counts(recvl.size(), 1);
      MPI_Neighbor_alltoallw(senddata, counts.data(), senddispls(), reinterpret_cast<const MPI_Datatype *>(sendl()),
                             recvdata, counts.data(), recvdispls(), reinterpret_cast<const MPI_Datatype *>(recvl()),
                             comm);
    }

    // --- non-blocking neighbour all-to-all ---
    template<typename T>
    irequest ineighbour_alltoallv(const T *senddata, const layouts <T> &sendl, const displacements &senddispls,
                                  T *recvdata, const layouts <T> &recvl, const displacements &recvdispls) const {
      std::vector<int> counts(recvl.size(), 1);
      MPI_Request req;
      MPI_Ineighbor_alltoallw(senddata, counts.data(), senddispls(), reinterpret_cast<const MPI_Datatype *>(sendl()),
                              recvdata, counts.data(), recvdispls(), reinterpret_cast<const MPI_Datatype *>(recvl()),
                              comm, &req);
      return irequest(req);
    }
  };

  //--------------------------------------------------------------------

  inline cart_communicator::sizes dims_create(int size, cart_communicator::sizes par) {
    if (MPI_Dims_create(size, par.dims_.size(), par.dims_.data())!=MPI_SUCCESS)
      throw invalid_dim();
    return par;
  }

}

#endif
