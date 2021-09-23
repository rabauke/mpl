#if !(defined MPL_TOPO_COMMUNICATOR_HPP)

#define MPL_TOPO_COMMUNICATOR_HPP

#include <mpi.h>
#include <vector>

namespace mpl {

  namespace impl {

    /// \brief Base class for communicators with a topology.
    class topo_communicator : public mpl::communicator {
    protected:
      /// \brief Default constructor.
      /// \note Objects of this class should not be instantiated by MPL users, just a base
      /// class.
      topo_communicator() = default;

      /// \brief Copy constructor.
      /// \param other the other communicator to copy from
      /// \note Objects of this class should not be instantiated by MPL users, just a base
      /// class.
      topo_communicator(const topo_communicator &other) = default;

      /// \brief Move constructor.
      /// \param other the other communicator to move from
      /// \note Objects of this class should not be instantiated by MPL users, just a base
      /// class.
      topo_communicator(topo_communicator &&other) = default;

    public:
      /// \brief Deleted copy assignment operator.
      void operator=(const topo_communicator &) = delete;

      // === neighbor collective =========================================
      // === neighbor allgather ===
      // === get a single value from each neighbor and store in contiguous memory
      // --- blocking neighbor allgather ---
      /// \brief Gather messages from all neighboring processes.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \ref data_types "data types" section
      /// \param senddata data to send to all neighbours
      /// \param recvdata pointer to continuous storage for incoming messages
      /// \note This is a collective operation and must be called (possibly by utilizing anther
      /// overload) by all  processes in the communicator.
      template<typename T>
      void neighbor_allgather(const T &senddata, T *recvdata) const {
        MPI_Neighbor_allgather(&senddata, 1, detail::datatype_traits<T>::get_datatype(),
                               recvdata, 1, detail::datatype_traits<T>::get_datatype(), comm_);
      }

      /// \brief Gather messages from all neighboring processes.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \ref data_types "data types" section
      /// \param senddata data to send to all neighbours
      /// \param sendl memory layout of the data to send
      /// \param recvdata pointer to continuous storage for incoming messages
      /// \param recvl memory layout of the data to receive
      /// \note This is a collective operation and must be called (possibly by utilizing anther
      /// overload) by all processes in the communicator.
      template<typename T>
      void neighbor_allgather(const T *senddata, const layout<T> &sendl, T *recvdata,
                              const layout<T> &recvl) const {
        MPI_Neighbor_allgather(
            senddata, 1, detail::datatype_traits<layout<T>>::get_datatype(sendl), recvdata, 1,
            detail::datatype_traits<layout<T>>::get_datatype(recvl), comm_);
      }

      // --- nonblocking neighbor allgather ---
      /// \brief Gather messages from all neighboring processes in a non-blocking manner.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \ref data_types "data types" section
      /// \param senddata data to send to all neighbours
      /// \param recvdata pointer to continuous storage for incoming messages
      /// \return request representing the ongoing message transfer
      /// \note This is a collective operation and must be called (possibly by utilizing anther
      /// overload) by all processes in the communicator.
      template<typename T>
      irequest ineighbor_allgather(const T &senddata, T *recvdata) const {
        MPI_Request req;
        MPI_Ineighbor_allgather(&senddata, 1, detail::datatype_traits<T>::get_datatype(),
                                recvdata, 1, detail::datatype_traits<T>::get_datatype(), comm_,
                                &req);
        return impl::irequest(req);
      }

      /// \brief Gather messages from all neighboring processes in a non-blocking manner.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \ref data_types "data types" section
      /// \param senddata data to send to all neighbours
      /// \param sendl memory layout of the data to send
      /// \param recvdata pointer to continuous storage for incoming messages
      /// \param recvl memory layout of the data to receive
      /// \return request representing the ongoing message transfer
      /// \note This is a collective operation and must be called (possibly by utilizing anther
      /// overload) by all processes in the communicator.
      template<typename T>
      irequest ineighbor_allgather(const T *senddata, const layout<T> &sendl, T *recvdata,
                                   const layout<T> &recvl) const {
        MPI_Request req;
        MPI_Ineighbor_allgather(
            senddata, 1, detail::datatype_traits<layout<T>>::get_datatype(sendl), recvdata, 1,
            detail::datatype_traits<layout<T>>::get_datatype(recvl), comm_, &req);
        return impl::irequest(req);
      }

      // === get varying amount of data from each neighbor and stores in non-contiguous memory
      // --- blocking neighbor allgather ---
      /// \brief Gather messages with a variable amount of data from all neighbouring processes.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \ref data_types "data types" section
      /// \param senddata data to send
      /// \param sendl memory layout of the data to send
      /// \param recvdata pointer to continuous storage for incoming messages
      /// \param recvls memory layouts of the data to receive
      /// \param recvdispls displacements of the data to receive
      /// \note This is a collective operation and must be called (possibly by utilizing anther
      /// overload) by all processes in the communicator.
      template<typename T>
      void neighbor_allgatherv(const T *senddata, const layout<T> &sendl, T *recvdata,
                               const layouts<T> &recvls,
                               const displacements &recvdispls) const {
        int N(recvdispls.size());
        displacements senddispls(N);
        layouts<T> sendls(N, sendl);
        neighbor_alltoallv(senddata, sendls, senddispls, recvdata, recvls, recvdispls);
      }

      /// \brief Gather messages with a variable amount of data from all neighbouring processes.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \ref data_types "data types" section
      /// \param senddata data to send
      /// \param sendl memory layout of the data to send
      /// \param recvdata pointer to continuous storage for incoming messages
      /// \param recvls memory layouts of the data to receive
      /// \note This is a collective operation and must be called (possibly by utilizing anther
      /// overload) by all processes in the communicator.
      template<typename T>
      void neighbor_allgatherv(const T *senddata, const layout<T> &sendl, T *recvdata,
                               const layouts<T> &recvls) const {
        neighbor_allgatherv(senddata, sendl, recvdata, recvls, displacements(size()));
      }

      // --- nonblocking neighbor allgather ---
      /// \brief Gather messages with a variable amount of data from all neighbouring processes
      /// in a non-blocking manner.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \ref data_types "data types" section
      /// \param senddata data to send
      /// \param sendl memory layout of the data to send
      /// \param recvdata pointer to continuous storage for incoming messages
      /// \param recvls memory layouts of the data to receive
      /// \param recvdispls displacements of the data to receive
      /// \return request representing the ongoing message transfer
      /// \note This is a collective operation and must be called (possibly by utilizing anther
      /// overload) by all processes in the communicator.
      template<typename T>
      irequest ineighbor_allgatherv(const T *senddata, const layout<T> &sendl, T *recvdata,
                                    const layouts<T> &recvls,
                                    const displacements &recvdispls) const {
        int N(recvdispls.size());
        displacements senddispls(N);
        layouts<T> sendls(N, sendl);
        return ineighbor_alltoallv(senddata, sendls, senddispls, recvdata, recvls, recvdispls);
      }

      /// \brief Gather messages with a variable amount of data from all neighbouring processes
      /// in a non-blocking manner.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \ref data_types "data types" section
      /// \param senddata data to send
      /// \param sendl memory layout of the data to send
      /// \param recvdata pointer to continuous storage for incoming messages
      /// \param recvls memory layouts of the data to receive
      /// \return request representing the ongoing message transfer
      /// \note This is a collective operation and must be called (possibly by utilizing anther
      /// overload) by all processes in the communicator.
      template<typename T>
      irequest ineighbor_allgatherv(const T *senddata, const layout<T> &sendl, T *recvdata,
                                    const layouts<T> &recvls) const {
        return ineighbor_allgatherv(senddata, sendl, recvdata, recvls, displacements(size()));
      }

      // === neighbor all-to-all ===
      // === each rank sends a single value to each neighbor
      // --- blocking neighbor all-to-all ---
      template<typename T>
      void neighbor_alltoall(const T *senddata, T *recvdata) const {
        MPI_Neighbor_alltoall(senddata, 1, detail::datatype_traits<T>::get_datatype(), recvdata,
                              1, detail::datatype_traits<T>::get_datatype(), comm_);
      }

      template<typename T>
      void neighbor_alltoall(const T *senddata, const layout<T> &sendl, T *recvdata,
                             const layout<T> &recvl) const {
        MPI_Neighbor_alltoall(senddata, 1, detail::datatype_traits<T>::get_datatype(), recvdata,
                              1, detail::datatype_traits<T>::get_datatype(), comm_);
      }

      // --- nonblocking neighbor all-to-all ---
      template<typename T>
      irequest ineighbor_alltoall(const T *senddata, T *recvdata) const {
        MPI_Request req;
        MPI_Ineighbor_alltoall(senddata, 1, detail::datatype_traits<T>::get_datatype(),
                               recvdata, 1, detail::datatype_traits<T>::get_datatype(), comm_,
                               &req);
        return impl::irequest(req);
      }

      template<typename T>
      irequest ineighbor_alltoall(const T *senddata, const layout<T> &sendl, T *recvdata,
                                  const layout<T> &recvl) const {
        MPI_Request req;
        MPI_Ineighbor_alltoall(senddata, 1, detail::datatype_traits<T>::get_datatype(),
                               recvdata, 1, detail::datatype_traits<T>::get_datatype(), comm_,
                               &req);
        return impl::irequest(req);
      }

      // === each rank sends a varying number of values to each neighbor with possibly different
      // layouts
      // --- blocking neighbor all-to-all ---
      template<typename T>
      void neighbor_alltoallv(const T *senddata, const layouts<T> &sendl,
                              const displacements &senddispls, T *recvdata,
                              const layouts<T> &recvl, const displacements &recvdispls) const {
        std::vector<int> counts(recvl.size(), 1);
        static_assert(
            sizeof(decltype(*sendl())) == sizeof(MPI_Datatype),
            "compiler adds some unexpected padding, reinterpret cast will yield wrong results");
        MPI_Neighbor_alltoallw(senddata, counts.data(), senddispls(),
                               reinterpret_cast<const MPI_Datatype *>(sendl()), recvdata,
                               counts.data(), recvdispls(),
                               reinterpret_cast<const MPI_Datatype *>(recvl()), comm_);
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
        static_assert(
            sizeof(decltype(*sendl())) == sizeof(MPI_Datatype),
            "compiler adds some unexpected padding, reinterpret cast will yield wrong results");
        MPI_Ineighbor_alltoallw(senddata, counts.data(), senddispls(),
                                reinterpret_cast<const MPI_Datatype *>(sendl()), recvdata,
                                counts.data(), recvdispls(),
                                reinterpret_cast<const MPI_Datatype *>(recvl()), comm_, &req);
        return impl::irequest(req);
      }

      template<typename T>
      irequest ineighbor_alltoallv(const T *senddata, const layouts<T> &sendl, T *recvdata,
                                   const layouts<T> &recvl) const {
        displacements sendrecvdispls(size());
        return ineighbor_alltoallv(senddata, sendl, sendrecvdispls, recvdata, recvl,
                                   sendrecvdispls);
      }
    };

  }  // namespace impl

}  // namespace mpl

#endif
