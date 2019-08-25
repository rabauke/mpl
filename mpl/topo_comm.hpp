#if !(defined MPL_TOPO_COMM_HPP)

#define MPL_TOPO_COMM_HPP

#include <mpi.h>
#include <vector>

namespace mpl {

  namespace detail {

    class topo_communicator : public mpl::communicator {
    public:
      topo_communicator() = default;

      void operator=(const topo_communicator &) = delete;

      // === neighbor collective =========================================
      // === neighbor allgather ===
      // === get a signle value from each neighbor and store in contiguous memory
      // --- blocking neighbor allgather ---
      template<typename T>
      void neighbor_allgather(const T &senddata, T *recvdata) const {
        MPI_Neighbor_allgather(&senddata, 1, datatype_traits<T>::get_datatype(), recvdata, 1,
                               datatype_traits<T>::get_datatype(), comm);
      }

      template<typename T>
      void neighbor_allgather(const T *senddata, const layout<T> &sendl, T *recvdata,
                              const layout<T> &recvl) const {
        MPI_Neighbor_allgather(senddata, 1, datatype_traits<layout<T>>::get_datatype(sendl),
                               recvdata, 1, datatype_traits<layout<T>>::get_datatype(recvl),
                               comm);
      }

      // --- nonblocking neighbor allgather ---
      template<typename T>
      irequest ineighbor_allgather(const T &senddata, T *recvdata) const {
        MPI_Request req;
        MPI_Ineighbor_allgather(&senddata, 1, datatype_traits<T>::get_datatype(), recvdata, 1,
                                datatype_traits<T>::get_datatype(), comm, &req);
        return irequest(req);
      }

      template<typename T>
      irequest ineighbor_allgather(const T *senddata, const layout<T> &sendl, T *recvdata,
                                   const layout<T> &recvl) const {
        MPI_Request req;
        MPI_Ineighbor_allgather(senddata, 1, datatype_traits<layout<T>>::get_datatype(sendl),
                                recvdata, 1, datatype_traits<layout<T>>::get_datatype(recvl),
                                comm, &req);
        return irequest(req);
      }

      // === get varying amount of data from each neighbor and stores in noncontiguous memory
      // --- blocking neighbor allgather ---
      template<typename T>
      void neighbor_allgatherv(const T *senddata, const layout<T> &sendl, T *recvdata,
                               const layouts<T> &recvls,
                               const displacements &recvdispls) const {
        int N(recvdispls.size());
        displacements senddispls(N);
        layouts<T> sendls(N, sendl);
        neighbor_alltoallv(senddata, sendls, senddispls, recvdata, recvls, recvdispls);
      }

      // --- nonblocking neighbor allgather ---
      template<typename T>
      irequest ineighbor_allgatherv(const T *senddata, const layout<T> &sendl, T *recvdata,
                                    const layouts<T> &recvls,
                                    const displacements &recvdispls) const {
        int N(recvdispls.size());
        displacements senddispls(N);
        layouts<T> sendls(N, sendl);
        return ineighbor_alltoallv(senddata, sendls, senddispls, recvdata, recvls, recvdispls);
      }

      // === neighbor all-to-all ===
      // === each rank sends a signle value to each neighbor
      // --- blocking neighbor all-to-all ---
      template<typename T>
      void neighbor_alltoall(const T *senddata, T *recvdata) const {
        MPI_Neighbor_alltoall(senddata, 1, datatype_traits<T>::get_datatype(), recvdata, 1,
                              datatype_traits<T>::get_datatype(), comm);
      }

      template<typename T>
      void neighbor_alltoall(const T *senddata, const layout<T> &sendl, T *recvdata,
                             const layout<T> &recvl) const {
        MPI_Neighbor_alltoall(senddata, 1, datatype_traits<T>::get_datatype(), recvdata, 1,
                              datatype_traits<T>::get_datatype(), comm);
      }

      // --- nonblocking neighbor all-to-all ---
      template<typename T>
      irequest ineighbor_alltoall(const T *senddata, T *recvdata) const {
        MPI_Request req;
        MPI_Ineighbor_alltoall(senddata, 1, datatype_traits<T>::get_datatype(), recvdata, 1,
                               datatype_traits<T>::get_datatype(), comm, &req);
        return irequest(req);
      }

      template<typename T>
      irequest ineighbor_alltoall(const T *senddata, const layout<T> &sendl, T *recvdata,
                                  const layout<T> &recvl) const {
        MPI_Request req;
        MPI_Ineighbor_alltoall(senddata, 1, datatype_traits<T>::get_datatype(), recvdata, 1,
                               datatype_traits<T>::get_datatype(), comm, &req);
        return irequest(req);
      }

      // === each rank sends a varying number of values to each neighbor with possibly different
      // layouts
      // --- blocking neighbor all-to-all ---
      template<typename T>
      void neighbor_alltoallv(const T *senddata, const layouts<T> &sendl,
                              const displacements &senddispls, T *recvdata,
                              const layouts<T> &recvl, const displacements &recvdispls) const {
        std::vector<int> counts(recvl.size(), 1);
        MPI_Neighbor_alltoallw(senddata, counts.data(), senddispls(),
                               reinterpret_cast<const MPI_Datatype *>(sendl()), recvdata,
                               counts.data(), recvdispls(),
                               reinterpret_cast<const MPI_Datatype *>(recvl()), comm);
      }

      template<typename T>
      void neighbor_alltoallv(const T *senddata, const layouts<T> &sendl, T *recvdata,
                              const layouts<T> &recvl) const {
        displacements sendrecvdispls(size());
        neighbor_alltoallv(senddata, sendl, sendrecvdispls, recvdata, recvl, sendrecvdispls);
      }

      // --- non-blocking neighbor all-to-all ---
      template<typename T>
      irequest ineighbor_alltoallv(const T *senddata, const layouts<T> &sendl,
                                   const displacements &senddispls, T *recvdata,
                                   const layouts<T> &recvl,
                                   const displacements &recvdispls) const {
        std::vector<int> counts(recvl.size(), 1);
        MPI_Request req;
        MPI_Ineighbor_alltoallw(senddata, counts.data(), senddispls(),
                                reinterpret_cast<const MPI_Datatype *>(sendl()), recvdata,
                                counts.data(), recvdispls(),
                                reinterpret_cast<const MPI_Datatype *>(recvl()), comm, &req);
        return irequest(req);
      }

      template<typename T>
      irequest ineighbor_alltoallv(const T *senddata, const layouts<T> &sendl, T *recvdata,
                                   const layouts<T> &recvl) const {
        displacements sendrecvdispls(size());
        return ineighbor_alltoallv(senddata, sendl, sendrecvdispls, recvdata, recvl,
                                   sendrecvdispls);
      }
    };

  }  // namespace detail

}  // namespace mpl

#endif
