#if !(defined MPL_GRAPH_COMM_HPP)

#define MPL_GRAPH_COMM_HPP

#include <mpi.h>
#include <algorithm>
#include <vector>
#include <utility>
#include <set>
#include <tuple>
#include <algorithm>

namespace mpl {

  class graph_communicator : public communicator {
  public:
    class edge_set : private std::set<std::pair<int, int>> {
      typedef std::set<std::pair<int, int>> base;
    public:
      using value_type=typename base::value_type;
      using size_type=typename base::size_type;
      using base::base;
      using base::size;
      using base::begin;
      using base::end;
      using base::cbegin;
      using base::cend;
      using base::insert;
      using base::operator=;
    };

    class node_list : private std::vector<int> {
      typedef std::vector<int> base;
    public:
      using value_type=typename base::value_type;
      using size_type=typename base::size_type;
      using base::base;
      using base::size;
      using base::begin;
      using base::end;
      using base::cbegin;
      using base::cend;
      using base::operator=;
      using base::operator[];
      using base::data;
    };

    graph_communicator()=default;

    explicit graph_communicator(const communicator &old_comm,
                                const edge_set &es,
                                bool reorder=true) {
      int nodes=0;
      for (const auto &e : es)
        nodes=std::max({ nodes, e.first+1, e.second+1 });
      int node=0;
      int degree=0;
      std::vector<int> edges, index(nodes, 0);
      for (const auto &e : es) {
        while (e.first>node) {
          ++node;
          degree=0;
        }
        edges.push_back(e.second);
        index[e.first]+=1;
      }
      std::partial_sum(index.begin(), index.end(), index.begin());
      MPI_Graph_create(old_comm.comm, nodes, index.data(), edges.data(), reorder, &comm);
    }

    graph_communicator(graph_communicator &&other) noexcept {
      comm=other.comm;
      other.comm=MPI_COMM_SELF;
    }

    void operator=(const graph_communicator &)= delete;

    graph_communicator &operator=(graph_communicator &&other) noexcept {
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

    int neighbors_count(int rank) const {
      int nneighbors;
      MPI_Graph_neighbors_count(comm, rank, &nneighbors);
      return nneighbors;
    };

    int neighbors_count() const {
      return neighbors_count(rank());
    };

    node_list neighbors(int rank) const {
      int maxneighbors=neighbors_count(rank);
      node_list nl(maxneighbors);
      MPI_Graph_neighbors(comm, rank, maxneighbors, nl.data());
      return nl;
    }

    node_list neighbors() const {
      return neighbors(rank());
    }

    // === neighbour collective ========================================
    // === neighbour allgather ===
    // === get a signle value from each neighbour and store in contiguous memory
    // --- blocking neighbour allgather ---
    template<typename T>
    void neighbour_allgather(const T &senddata, T *recvdata) const {
      MPI_Neighbor_allgather(&senddata, 1, datatype_traits<T>::get_datatype(),
                             recvdata, 1, datatype_traits<T>::get_datatype(),
                             comm);
    }

    template<typename T>
    void neighbour_allgather(const T *senddata, const layout <T> &sendl,
                             T *recvdata, const layout <T> &recvl) const {
      MPI_Neighbor_allgather(senddata, 1, datatype_traits<layout<T >>::get_datatype(sendl),
                             recvdata, 1, datatype_traits<layout<T >>::get_datatype(recvl),
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
      MPI_Ineighbor_allgather(senddata, 1, datatype_traits<layout<T >>::get_datatype(sendl),
                              recvdata, 1, datatype_traits<layout<T >>::get_datatype(recvl),
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

}

#endif
