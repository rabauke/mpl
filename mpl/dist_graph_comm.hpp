#if !(defined MPL_DIST_GRAPH_COMM_HPP)

#define MPL_DIST_GRAPH_COMM_HPP

#include <mpi.h>
#include <algorithm>
#include <vector>
#include <utility>
#include <set>
#include <tuple>
#include <algorithm>
#include <numeric>

namespace mpl {

  class dist_graph_communicator : public detail::topo_communicator {
  public:
    class source_set : private std::set<std::pair<int, int>> {
      using base = std::set<std::pair<int, int>>;

    public:
      using value_type = typename base::value_type;
      using size_type = typename base::size_type;
      using base::base;
      using base::size;
      using base::begin;
      using base::end;
      using base::cbegin;
      using base::cend;
      using base::insert;
      using base::operator=;
    };

    class dest_set : private std::set<std::pair<int, int>> {
      using base = std::set<std::pair<int, int>>;

    public:
      using value_type = typename base::value_type;
      using size_type = typename base::size_type;
      using base::base;
      using base::size;
      using base::begin;
      using base::end;
      using base::cbegin;
      using base::cend;
      using base::insert;
      using base::operator=;
    };

    dist_graph_communicator() = default;

    explicit dist_graph_communicator(const communicator &old_comm, const source_set &ss,
                                     const dest_set &ds, bool reorder = true) {
      std::vector<int> sources, sourceweigths;
      for (auto x : ss) {
        sources.push_back(x.first);
        sourceweigths.push_back(x.second);
      }
      std::vector<int> destinations, destinationweigths;
      for (auto x : ds) {
        destinations.push_back(x.first);
        destinationweigths.push_back(x.second);
      }
      MPI_Dist_graph_create_adjacent(old_comm.comm, sources.size(), sources.data(),
                                     sourceweigths.data(), destinations.size(),
                                     destinations.data(), destinationweigths.data(),
                                     MPI_INFO_NULL, reorder, &comm);
    }

    dist_graph_communicator(dist_graph_communicator &&other) noexcept {
      comm = other.comm;
      other.comm = MPI_COMM_SELF;
    }

    void operator=(const dist_graph_communicator &) = delete;

    dist_graph_communicator &operator=(dist_graph_communicator &&other) noexcept {
      if (this != &other) {
        int result1, result2;
        MPI_Comm_compare(comm, MPI_COMM_WORLD, &result1);
        MPI_Comm_compare(comm, MPI_COMM_SELF, &result2);
        if (result1 != MPI_IDENT and result2 != MPI_IDENT)
          MPI_Comm_free(&comm);
        comm = other.comm;
        other.comm = MPI_COMM_SELF;
      }
      return *this;
    }

    int indegree() const {
      int _indegree, _outdegree, _weighted;
      MPI_Dist_graph_neighbors_count(comm, &_indegree, &_outdegree, &_weighted);
      return _indegree;
    };

    int outdegree() const {
      int _indegree, _outdegree, _weighted;
      MPI_Dist_graph_neighbors_count(comm, &_indegree, &_outdegree, &_weighted);
      return _outdegree;
    };

    source_set inneighbors() const {
      int indeg{indegree()};
      std::vector<int> sources(indeg), sourceweigths(indeg);
      int outdeg{outdegree()};
      std::vector<int> destinations(outdeg), destinationweigths(outdeg);
      MPI_Dist_graph_neighbors(comm, indeg, sources.data(), sourceweigths.data(), outdeg,
                               destinations.data(), destinationweigths.data());
      source_set ss;
      for (int i = 0; i < indeg; ++i)
        ss.insert({sources[i], sourceweigths[i]});
      return ss;
    }

    dest_set outneighbors() const {
      int indeg{indegree()};
      std::vector<int> sources(indeg), sourceweigths(indeg);
      int outdeg{outdegree()};
      std::vector<int> destinations(outdeg), destinationweigths(outdeg);
      MPI_Dist_graph_neighbors(comm, indeg, sources.data(), sourceweigths.data(), outdeg,
                               destinations.data(), destinationweigths.data());
      dest_set ds;
      for (int i = 0; i < indeg; ++i)
        ds.insert({sources[i], sourceweigths[i]});
      return ds;
    }
  };

}  // namespace mpl

#endif
