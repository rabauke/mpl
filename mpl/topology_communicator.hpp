#if !(defined MPL_TOPOLOGY_COMMUNICATOR_HPP)

#define MPL_TOPOLOGY_COMMUNICATOR_HPP

#include <mpi.h>
#include <vector>

namespace mpl::impl {

  /// Base class for communicators with a topology.
  class topology_communicator : public mpl::communicator {
  protected:
    /// Default constructor.
    /// \note Objects of this class should not be instantiated by MPL users, just a base
    /// class.
    topology_communicator() = default;

    /// Copy constructor.
    /// \param other the other communicator to copy from
    /// \note Objects of this class should not be instantiated by MPL users, just a base
    /// class.
    topology_communicator(const topology_communicator &other) = default;

    /// Move constructor.
    /// \param other the other communicator to move from
    /// \note Objects of this class should not be instantiated by MPL users, just a base
    /// class.
    topology_communicator(topology_communicator &&other) = default;

  public:
    /// Deleted copy assignment operator.
    void operator=(const topology_communicator &) = delete;

    // === neighbor collective =========================================
    // === neighbor allgather ===
    // === get a single value from each neighbor and store in contiguous memory
    // --- blocking neighbor allgather ---
    /// Gather messages from all neighboring processes.
    /// \tparam T type of the data to send, must meet the requirements as described in the
    /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
    /// \param senddata data to send to all neighbours
    /// \param recvdata pointer to continuous storage for incoming messages
    /// \note This is a collective operation and must be called (possibly by utilizing another
    /// overload) by all  processes in the communicator.
    template<typename T>
    void neighbor_allgather(const T &senddata, T *recvdata) const {
      MPI_Neighbor_allgather(&senddata, 1, detail::datatype_traits<T>::get_datatype(), recvdata,
                             1, detail::datatype_traits<T>::get_datatype(), comm_);
    }

    /// Gather messages from all neighboring processes.
    /// \tparam T type of the data to send, must meet the requirements as described in the
    /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
    /// \param senddata data to send to all neighbours
    /// \param sendl memory layout of the data to send
    /// \param recvdata pointer to continuous storage for incoming messages
    /// \param recvl memory layout of the data to receive
    /// \note This is a collective operation and must be called (possibly by utilizing another
    /// overload) by all processes in the communicator.
    template<typename T>
    void neighbor_allgather(const T *senddata, const layout<T> &sendl, T *recvdata,
                            const layout<T> &recvl) const {
      MPI_Neighbor_allgather(senddata, 1,
                             detail::datatype_traits<layout<T>>::get_datatype(sendl), recvdata,
                             1, detail::datatype_traits<layout<T>>::get_datatype(recvl), comm_);
    }

    // --- nonblocking neighbor allgather ---
    /// Gather messages from all neighboring processes in a non-blocking manner.
    /// \tparam T type of the data to send, must meet the requirements as described in the
    /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
    /// \param senddata data to send to all neighbours
    /// \param recvdata pointer to continuous storage for incoming messages
    /// \return request representing the ongoing message transfer
    /// \note This is a collective operation and must be called (possibly by utilizing another
    /// overload) by all processes in the communicator.
    template<typename T>
    mpl::irequest ineighbor_allgather(const T &senddata, T *recvdata) const {
      MPI_Request req;
      MPI_Ineighbor_allgather(&senddata, 1, detail::datatype_traits<T>::get_datatype(),
                              recvdata, 1, detail::datatype_traits<T>::get_datatype(), comm_,
                              &req);
      return impl::base_irequest{req};
    }

    /// Gather messages from all neighboring processes in a non-blocking manner.
    /// \tparam T type of the data to send, must meet the requirements as described in the
    /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
    /// \param senddata data to send to all neighbours
    /// \param sendl memory layout of the data to send
    /// \param recvdata pointer to continuous storage for incoming messages
    /// \param recvl memory layout of the data to receive
    /// \return request representing the ongoing message transfer
    /// \note This is a collective operation and must be called (possibly by utilizing another
    /// overload) by all processes in the communicator.
    template<typename T>
    mpl::irequest ineighbor_allgather(const T *senddata, const layout<T> &sendl, T *recvdata,
                                      const layout<T> &recvl) const {
      MPI_Request req;
      MPI_Ineighbor_allgather(
          senddata, 1, detail::datatype_traits<layout<T>>::get_datatype(sendl), recvdata, 1,
          detail::datatype_traits<layout<T>>::get_datatype(recvl), comm_, &req);
      return impl::base_irequest{req};
    }

    // === get varying amount of data from each neighbor and stores in non-contiguous memory
    // --- blocking neighbor allgather ---
    /// Gather messages with a variable amount of data from all neighbouring processes.
    /// \tparam T type of the data to send, must meet the requirements as described in the
    /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
    /// \param senddata data to send
    /// \param sendl memory layout of the data to send
    /// \param recvdata pointer to continuous storage for incoming messages
    /// \param recvls memory layouts of the data to receive
    /// \param recvdispls displacements of the data to receive
    /// \note This is a collective operation and must be called (possibly by utilizing another
    /// overload) by all processes in the communicator.
    template<typename T>
    void neighbor_allgatherv(const T *senddata, const layout<T> &sendl, T *recvdata,
                             const layouts<T> &recvls, const displacements &recvdispls) const {
      const int n(recvdispls.size());
      const displacements senddispls(n);
      const layouts<T> sendls(n, sendl);
      neighbor_alltoallv(senddata, sendls, senddispls, recvdata, recvls, recvdispls);
    }

    /// Gather messages with a variable amount of data from all neighbouring processes.
    /// \tparam T type of the data to send, must meet the requirements as described in the
    /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
    /// \param senddata data to send
    /// \param sendl memory layout of the data to send
    /// \param recvdata pointer to continuous storage for incoming messages
    /// \param recvls memory layouts of the data to receive
    /// \note This is a collective operation and must be called (possibly by utilizing another
    /// overload) by all processes in the communicator.
    template<typename T>
    void neighbor_allgatherv(const T *senddata, const layout<T> &sendl, T *recvdata,
                             const layouts<T> &recvls) const {
      neighbor_allgatherv(senddata, sendl, recvdata, recvls, displacements(size()));
    }

    // --- nonblocking neighbor allgather ---
    /// Gather messages with a variable amount of data from all neighbouring processes
    /// in a non-blocking manner.
    /// \tparam T type of the data to send, must meet the requirements as described in the
    /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
    /// \param senddata data to send
    /// \param sendl memory layout of the data to send
    /// \param recvdata pointer to continuous storage for incoming messages
    /// \param recvls memory layouts of the data to receive
    /// \param recvdispls displacements of the data to receive
    /// \return request representing the ongoing message transfer
    /// \note This is a collective operation and must be called (possibly by utilizing another
    /// overload) by all processes in the communicator.
    template<typename T>
    mpl::irequest ineighbor_allgatherv(const T *senddata, const layout<T> &sendl, T *recvdata,
                                       const layouts<T> &recvls,
                                       const displacements &recvdispls) const {
      const int n(recvdispls.size());
      const displacements senddispls(n);
      const layouts<T> sendls(n, sendl);
      return ineighbor_alltoallv(senddata, sendls, senddispls, recvdata, recvls, recvdispls);
    }

    /// Gather messages with a variable amount of data from all neighbouring processes
    /// in a non-blocking manner.
    /// \tparam T type of the data to send, must meet the requirements as described in the
    /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
    /// \param senddata data to send
    /// \param sendl memory layout of the data to send
    /// \param recvdata pointer to continuous storage for incoming messages
    /// \param recvls memory layouts of the data to receive
    /// \return request representing the ongoing message transfer
    /// \note This is a collective operation and must be called (possibly by utilizing another
    /// overload) by all processes in the communicator.
    template<typename T>
    mpl::irequest ineighbor_allgatherv(const T *senddata, const layout<T> &sendl, T *recvdata,
                                       const layouts<T> &recvls) const {
      return ineighbor_allgatherv(senddata, sendl, recvdata, recvls, displacements(size()));
    }

    // === neighbor all-to-all ===
    // === each rank sends a single value to each neighbor
    // --- blocking neighbor all-to-all ---
    /// Sends messages to all neighbouring processes and receives messages from all
    /// neighbouring processes.
    /// \tparam T type of the data to send, must meet the requirements as described in the
    /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
    /// \param senddata pointer to continuous storage for outgoing messages
    /// \param recvdata pointer to continuous storage for incoming messages
    /// \details Each process in the communicator sends one element of type \c T to each
    /// neighbouring process and receives one element of type \c T from each neighbouring
    /// process.  The i-th element in the array \c senddata is sent to the i-th neighbour.  When
    /// the function has finished, the i-th element in the array \c recvdata was received from
    /// the i-th neighbour.
    /// \note This is a collective operation and must be called (possibly by utilizing another
    /// overload) by all processes in the communicator.
    template<typename T>
    void neighbor_alltoall(const T *senddata, T *recvdata) const {
      MPI_Neighbor_alltoall(senddata, 1, detail::datatype_traits<T>::get_datatype(), recvdata,
                            1, detail::datatype_traits<T>::get_datatype(), comm_);
    }

    /// Sends messages to all neighbouring processes and receives messages from all
    /// neighbouring processes.
    /// \tparam T type of the data to send, must meet the requirements as described in the \ref
    /// data_types "data types" section
    /// \param senddata pointer to continuous storage for outgoing messages
    /// \param sendl memory layouts of the data to send
    /// \param recvdata pointer to continuous storage for incoming messages
    /// \param recvl memory layouts of the data to receive
    /// \details Each process in the communicator sends elements of type \c T to each
    /// neighbouring process and receives elements of type \c T from each neighbouring process.
    /// The memory layouts of the incoming and the outgoing messages are described by \c sendl
    /// and \c recvl. Both layouts might differ but must be compatible, i.e., must hold the same
    /// number of elements of type \c T.  The i-th memory block with the layout \c sendl in the
    /// array \c senddata is sent to the i-th neighbour.  When the function has finished, the
    /// i-th memory block with the layout \c recvl in the array \c recvdata was received from
    /// the i-th neigbour.
    /// \note This is a collective operation and must be called (possibly by utilizing another
    /// overload) by all processes in the communicator.
    template<typename T>
    void neighbor_alltoall(const T *senddata, const layout<T> &sendl, T *recvdata,
                           const layout<T> &recvl) const {
      MPI_Neighbor_alltoall(senddata, 1,
                            detail::datatype_traits<layout<T>>::get_datatype(sendl), recvdata,
                            1, detail::datatype_traits<layout<T>>::get_datatype(recvl), comm_);
    }

    // --- nonblocking neighbor all-to-all ---
    /// Sends messages to all neighbouring processes and receives messages from all
    /// neighbouring processes in a non-blocking manner.
    /// \tparam T type of the data to send, must meet the requirements as described in the
    /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
    /// \param senddata pointer to continuous storage for outgoing messages
    /// \param recvdata pointer to continuous storage for incoming messages
    /// \return request representing the ongoing message transfer
    /// \details Each process in the communicator sends one element of type \c T to each
    /// neighbouring process and receives one element of type \c T from each neighbouring
    /// process.  The i-th element in the array \c senddata is sent to the i-th neighbour.  When
    /// the function has finished, the i-th element in the array \c recvdata was received from
    /// the i-th neighbour.
    /// \note This is a collective operation and must be called (possibly by utilizing another
    /// overload) by all processes in the communicator.
    template<typename T>
    mpl::irequest ineighbor_alltoall(const T *senddata, T *recvdata) const {
      MPI_Request req;
      MPI_Ineighbor_alltoall(senddata, 1, detail::datatype_traits<T>::get_datatype(), recvdata,
                             1, detail::datatype_traits<T>::get_datatype(), comm_, &req);
      return impl::base_irequest{req};
    }

    /// Sends messages to all neighbouring processes and receives messages from all
    /// neighbouring processes in a non-blocking manner.
    /// \tparam T type of the data to send, must meet the requirements as described in the \ref
    /// data_types "data types" section
    /// \param senddata pointer to continuous storage for outgoing messages
    /// \param sendl memory layouts of the data to send
    /// \param recvdata pointer to continuous storage for incoming messages
    /// \param recvl memory layouts of the data to receive
    /// \return request representing the ongoing message transfer
    /// \details Each process in the communicator sends elements of type \c T to each
    /// neighbouring process and receives elements of type \c T from each neighbouring process.
    /// The memory layouts of the incoming and the outgoing messages are described by \c sendl
    /// and \c recvl. Both layouts might differ but must be compatible, i.e., must hold the same
    /// number of elements of type \c T.  The i-th memory block with the layout \c sendl in the
    /// array \c senddata is sent to the i-th neighbour.  When the function has finished, the
    /// i-th memory block with the layout \c recvl in the array \c recvdata was received from
    /// the i-th neighbour.
    /// \note This is a collective operation and must be called (possibly by utilizing another
    /// overload) by all processes in the communicator.
    template<typename T>
    mpl::irequest ineighbor_alltoall(const T *senddata, const layout<T> &sendl, T *recvdata,
                                     const layout<T> &recvl) const {
      MPI_Request req;
      MPI_Ineighbor_alltoall(
          senddata, 1, detail::datatype_traits<layout<T>>::get_datatype(sendl), recvdata, 1,
          detail::datatype_traits<layout<T>>::get_datatype(recvl), comm_, &req);
      return impl::base_irequest{req};
    }

    // === each rank sends a varying number of values to each neighbor with possibly different
    // layouts
    // --- blocking neighbor all-to-all ---
    /// Sends messages with a variable amount of data to all neighbouring processes and
    /// receives messages with a variable amount of data from all neighbouring processes.
    /// \tparam T type of the data to send, must meet the requirements as described in the \ref
    /// data_types "data types" section
    /// \param senddata pointer to continuous storage for outgoing messages
    /// \param sendls memory layouts of the data to send
    /// \param senddispls displacements of the data to send
    /// \param recvdata pointer to continuous storage for incoming messages
    /// \param recvls memory layouts of the data to receive
    /// \param recvdispls displacements of the data to receive
    /// \details Each process in the communicator sends elements of type \c T to each neighbor
    /// and receives elements of type \c T from each neighbour.  Send- and  receive-data are
    /// stored in consecutive blocks of variable size in the buffers \c senddata and
    /// \c recvdata, respectively. The i-th memory block with the layout <tt>sendls[i]</tt> in
    /// the array \c senddata starts <tt>senddispls[i]</tt> bytes after the address given in
    /// \c senddata. The i-th memory block is sent to the i-th neighbor. The i-th memory block
    /// with the layout <tt>recvls[i]</tt> in the array \c recvdata starts
    /// <tt>recvdispls[i]</tt> bytes after the address given in \c  recvdata. When the function
    /// has finished, the i-th memory block in the array \c recvdata was received from the i-th
    /// neighbor.
    /// \note This is a collective operation and must be called (possibly by utilizing another
    /// overload) by all processes in the communicator.
    template<typename T>
    void neighbor_alltoallv(const T *senddata, const layouts<T> &sendls,
                            const displacements &senddispls, T *recvdata,
                            const layouts<T> &recvls, const displacements &recvdispls) const {
      const std::vector<int> counts(recvls.size(), 1);
      static_assert(
          sizeof(decltype(*sendls())) == sizeof(MPI_Datatype),
          "compiler adds some unexpected padding, reinterpret cast will yield wrong results");
      MPI_Neighbor_alltoallw(senddata, counts.data(), senddispls(),
                             reinterpret_cast<const MPI_Datatype *>(sendls()), recvdata,
                             counts.data(), recvdispls(),
                             reinterpret_cast<const MPI_Datatype *>(recvls()), comm_);
    }

    /// Sends messages with a variable amount of data to all neighbouring processes and
    /// receives messages with a variable amount of data from all neighbouring processes.
    /// \tparam T type of the data to send, must meet the requirements as described in the
    /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
    /// \param senddata pointer to continuous storage for outgoing messages
    /// \param sendls memory layouts of the data to send
    /// \param recvdata pointer to continuous storage for incoming messages
    /// \param recvls memory layouts of the data to receive
    /// \details Each process in the communicator sends elements of type \c T to each neighbour
    /// and receives elements of type \c T from each neighbour.  Send- and receive-data are
    /// stored in consecutive blocks of variable size in the buffers \c senddata and
    /// \c recvdata, respectively. The i-th memory block with the layout <tt>sendls[i]</tt> in
    /// the array \c senddata starts at the address given in \c senddata. The i-th memory block
    /// is sent to the i-th neighbour. The i-th memory block with the layout <tt>recvls[i]</tt>
    /// in the array \c recvdata  starts at the address given in \c recvdata.  Note that the
    /// memory layouts need to include  appropriate holes at the beginning in order to avoid
    /// overlapping send blocks or receive blocks. When the function has finished, the i-th
    /// memory block in the array \c recvdata was received from the i-th neighbour.
    /// \note This is a collective operation and must be called (possibly by utilizing another
    /// overload) by all processes in the communicator.
    template<typename T>
    void neighbor_alltoallv(const T *senddata, const layouts<T> &sendls, T *recvdata,
                            const layouts<T> &recvls) const {
      const displacements sendrecvdispls(size());
      neighbor_alltoallv(senddata, sendls, sendrecvdispls, recvdata, recvls, sendrecvdispls);
    }

    // --- non-blocking neighbor all-to-all ---
    /// Sends messages with a variable amount of data to all neighbouring processes and
    /// receives messages with a variable amount of data from all neighbouring processes in a
    /// non-blocking manner.
    /// \tparam T type of the data to send, must meet the requirements as described in the \ref
    /// data_types "data types" section
    /// \param senddata pointer to continuous storage for outgoing messages
    /// \param sendls memory layouts of the data to send
    /// \param senddispls displacements of the data to send
    /// \param recvdata pointer to continuous storage for incoming messages
    /// \param recvls memory layouts of the data to receive
    /// \param recvdispls displacements of the data to receive
    /// \return request representing the ongoing message transfer
    /// \details Each process in the communicator sends elements of type \c T to each neighbor
    /// and receives elements of type \c T from each neighbour.  Send- and  receive-data are
    /// stored in consecutive blocks of variable size in the buffers \c senddata and
    /// \c recvdata, respectively. The i-th memory block with the layout <tt>sendls[i]</tt> in
    /// the array \c senddata starts <tt>senddispls[i]</tt> bytes after the address given in
    /// \c senddata. The i-th memory block is sent to the i-th neighbor. The i-th memory block
    /// with the layout <tt>recvls[i]</tt> in the array \c recvdata starts
    /// <tt>recvdispls[i]</tt> bytes after the address given in \c recvdata. When the function
    /// has finished, the i-th memory block in the array \c recvdata was received from the i-th
    /// neighbor.
    /// \note This is a collective operation and must be called (possibly by utilizing another
    /// overload) by all processes in the communicator.
    template<typename T>
    mpl::irequest ineighbor_alltoallv(const T *senddata, const layouts<T> &sendls,
                                      const displacements &senddispls, T *recvdata,
                                      const layouts<T> &recvls,
                                      const displacements &recvdispls) const {
      std::vector<int> counts(recvls.size(), 1);
      MPI_Request req;
      static_assert(
          sizeof(decltype(*sendls())) == sizeof(MPI_Datatype),
          "compiler adds some unexpected padding, reinterpret cast will yield wrong results");
      MPI_Ineighbor_alltoallw(senddata, counts.data(), senddispls(),
                              reinterpret_cast<const MPI_Datatype *>(sendls()), recvdata,
                              counts.data(), recvdispls(),
                              reinterpret_cast<const MPI_Datatype *>(recvls()), comm_, &req);
      return impl::base_irequest{req};
    }

    /// Sends messages with a variable amount of data to all neighbouring processes and
    /// receives messages with a variable amount of data from all neighbouring processes in a
    /// non-blocking manner.
    /// \tparam T type of the data to send, must meet the requirements as described in the
    /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
    /// \param senddata pointer to continuous storage for outgoing messages
    /// \param sendls memory layouts of the data to send
    /// \param recvdata pointer to continuous storage for incoming messages
    /// \param recvls memory layouts of the data to receive
    /// \return request representing the ongoing message transfer
    /// \details Each process in the communicator sends elements of type \c T to each neighbour
    /// and receives elements of type \c T from each neighbour.  Send- and receive-data are
    /// stored in consecutive blocks of variable size in the buffers \c senddata and
    /// \c recvdata, respectively. The i-th memory block with the layout <tt>sendls[i]</tt> in
    /// the array \c senddata starts at the address given in \c senddata. The i-th memory block
    /// is sent to the i-th neighbour. The i-th memory block with the layout <tt>recvls[i]</tt>
    /// in the array \c recvdata starts at the address given in \c recvdata.  Note that the
    /// memory layouts need to include appropriate holes at the beginning in order to avoid
    /// overlapping send blocks or receive blocks. When the function has finished, the i-th
    /// memory block in the array \c recvdata was received from the i-th neighbour.
    /// \note This is a collective operation and must be called (possibly by utilizing another
    /// overload) by all processes in the communicator.
    template<typename T>
    mpl::irequest ineighbor_alltoallv(const T *senddata, const layouts<T> &sendls, T *recvdata,
                                      const layouts<T> &recvls) const {
      const displacements sendrecvdispls(size());
      return ineighbor_alltoallv(senddata, sendls, sendrecvdispls, recvdata, recvls,
                                 sendrecvdispls);
    }
  };

}  // namespace mpl::impl

#endif
