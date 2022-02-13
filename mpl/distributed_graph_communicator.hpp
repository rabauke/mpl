#if !(defined MPL_DISTRIBUTED_GRAPH_COMMUNICATOR_HPP)

#define MPL_DISTRIBUTED_GRAPH_COMMUNICATOR_HPP

#include <mpi.h>
#include <algorithm>
#include <vector>
#include <utility>
#include <set>
#include <tuple>
#include <algorithm>
#include <numeric>

namespace mpl {

  /// Communicator with general graph topology.
  class distributed_graph_communicator : public impl::topology_communicator {
  public:
    /// Pair of rank and weight.
    class rank_weight_pair {
    public:
      int rank{0};
      int weight{0};
      /// Creates a rank-weight pair.
      rank_weight_pair(int rank, int weight = 0) : rank{rank}, weight{weight} {}
    };

  private:
    class less_weights {
    public:
      bool operator()(const rank_weight_pair &pair_1, const rank_weight_pair &pair_2) const {
        if (pair_1.rank == pair_2.rank)
          return pair_1.weight < pair_2.weight;
        else
          return pair_1.rank < pair_2.rank;
      }
    };

  public:
    /// Set of edges, pairs of nodes represented by non-negative integers.
    class neighbours_set : private std::set<rank_weight_pair, less_weights> {
      using base = std::set<rank_weight_pair, less_weights>;

    public:
      using value_type = typename base::value_type;
      using reference = typename base::reference;
      using const_reference = typename base::const_reference;
      using iterator = typename base::iterator;
      using const_iterator = typename base::const_iterator;

      /// Creates an empty set of edges.
      neighbours_set() = default;


      /// Creates a set of edges given by the list.
      /// \param init set of edges
      neighbours_set(std::initializer_list<value_type> init) : base(init) {}

      using base::operator=;
      using base::begin;
      using base::end;
      using base::cbegin;
      using base::cend;

      /// Determines the number edges.
      /// \return number of edges in the edge set
      [[nodiscard]] int size() const { return static_cast<int>(base::size()); }

      /// Add an additional edge to the set.
      /// \param edge tuple of two non-negative integers
      void add(const value_type &edge) { insert(edge); }
    };


    /// Creates an empty communicator with no associated process.
    distributed_graph_communicator() = default;

    /// Creates a new communicator which is equivalent to an existing one.
    /// \param other the other communicator to copy from
    /// \note This is a collective operation that needs to be carried out by all processes of
    /// the communicator other. Communicators should not be copied unless a new independent
    /// communicator is wanted. Communicators should be passed via references to functions to
    /// avoid unnecessary copying.
    distributed_graph_communicator(const distributed_graph_communicator &other) {
      MPI_Comm_dup(other.comm_, &comm_);
    }

    /// Move-constructs a communicator.
    /// \param other the other communicator to move from
    distributed_graph_communicator(distributed_graph_communicator &&other) noexcept {
      comm_ = other.comm_;
      other.comm_ = MPI_COMM_SELF;
    }

    /// Creates a new communicator with graph process topology.
    /// \param other communicator containing the processes to use in the creation of the new
    /// communicator
    /// \param sources ranks and associated weights of processes for which the calling process
    /// is a destination
    /// \param destinations ranks and associated weights of processes for which the calling
    /// process is a source
    /// \param reorder indicates if reordering is permitted, if false each process will have the
    /// same rank in the new communicator as in the old one
    /// \note This is a collective operation that needs to be carried out by all processes of
    /// the communicator.  The rank of the calling process and source ranks define a set of
    /// source-edges, and the rank of the calling process and destination ranks define a set of
    /// destination-edges.  The combination of all source-edges must equal the combination of
    /// all destination-edges in this collective operation.  Edge weights given in the sources
    /// and destinations arguments may affect the rank ordering but the specific meaning is
    /// determined by the underling MPI implementation.
    explicit distributed_graph_communicator(const communicator &other,
                                            const neighbours_set &sources,
                                            const neighbours_set &destinations,
                                            bool reorder = true) {
      std::vector<int> sources_vector;
      std::vector<int> source_weights;
      sources_vector.reserve(sources.size());
      source_weights.reserve(sources.size());
      for (const auto &x : sources) {
        sources_vector.push_back(x.rank);
        source_weights.push_back(x.weight);
      }
      std::vector<int> destinations_vector;
      std::vector<int> destination_weights;
      destinations_vector.reserve(destinations.size());
      destination_weights.reserve(destinations.size());
      for (const auto &x : destinations) {
        destinations_vector.push_back(x.rank);
        destination_weights.push_back(x.weight);
      }
      MPI_Dist_graph_create_adjacent(other.comm_, sources_vector.size(), sources_vector.data(),
                                     source_weights.data(), destinations.size(),
                                     destinations_vector.data(), destination_weights.data(),
                                     MPI_INFO_NULL, reorder, &comm_);
    }

    /// Copy-assigns and creates a new communicator with graph process topology which
    /// is equivalent to an existing one.
    /// \param other the other communicator to copy from
    /// \note This is a collective operation that needs to be carried out by all processes of
    /// the communicator other. Communicators should not be copied unless a new independent
    /// communicator is wanted. Communicators should be passed via references to functions to
    /// avoid unnecessary copying.
    distributed_graph_communicator &operator=(
        const distributed_graph_communicator &other) noexcept {
      if (this != &other) {
        if (is_valid()) {
          int result_1;
          MPI_Comm_compare(comm_, MPI_COMM_WORLD, &result_1);
          int result_2;
          MPI_Comm_compare(comm_, MPI_COMM_SELF, &result_2);
          if (result_1 != MPI_IDENT and result_2 != MPI_IDENT)
            MPI_Comm_free(&comm_);
        }
        MPI_Comm_dup(other.comm_, &comm_);
      }
      return *this;
    }

    /// Move-assigns a communicator.
    /// \param other the other communicator to move from
    /// \note This is a collective operation that needs to be carried out by all processes of
    /// the communicator other.
    distributed_graph_communicator &operator=(distributed_graph_communicator &&other) noexcept {
      if (this != &other) {
        int result_1;
        MPI_Comm_compare(comm_, MPI_COMM_WORLD, &result_1);
        int result_2;
        MPI_Comm_compare(comm_, MPI_COMM_SELF, &result_2);
        if (result_1 != MPI_IDENT and result_2 != MPI_IDENT)
          MPI_Comm_free(&comm_);
        comm_ = other.comm_;
        other.comm_ = MPI_COMM_SELF;
      }
      return *this;
    }

    /// Determines the number of edges into and out of this process.
    /// \return in- and out-degree
    [[nodiscard]] std::tuple<int, int> in_out_degree() const {
      int t_in_degree, t_out_degree, t_weighted;
      MPI_Dist_graph_neighbors_count(comm_, &t_in_degree, &t_out_degree, &t_weighted);
      return {t_in_degree, t_out_degree};
    };

    /// Determines the number of edges into this process.
    /// \return in-degree
    [[nodiscard]] int in_degree() const { return std::get<0>(in_out_degree()); };

    /// Determines the number of edges out of this process.
    /// \return out-degree
    [[nodiscard]] int out_degree() const { return std::get<1>(in_out_degree()); };

    /// Determines the ranks of the processes for which the calling process is a
    /// destination.
    /// \return in-neighbours with associated weights
    [[nodiscard]] neighbours_set in_neighbors() const {
      const auto [in_deg, out_deg]{in_out_degree()};
      std::vector<int> sources(in_deg), source_weights(in_deg);
      std::vector<int> destinations(out_deg), destination_weights(out_deg);
      MPI_Dist_graph_neighbors(comm_, in_deg, sources.data(), source_weights.data(), out_deg,
                               destinations.data(), destination_weights.data());
      neighbours_set neighbours;
      for (int i{0}; i < in_deg; ++i)
        neighbours.add({sources[i], source_weights[i]});
      return neighbours;
    }

    /// Determines the ranks of the processes for which the calling process is a
    /// source.
    /// \return out-neighbours with associated weights
    [[nodiscard]] neighbours_set out_neighbors() const {
      const auto [in_deg, out_deg]{in_out_degree()};
      std::vector<int> sources(in_deg), source_weights(in_deg);
      std::vector<int> destinations(out_deg), destination_weights(out_deg);
      MPI_Dist_graph_neighbors(comm_, in_deg, sources.data(), source_weights.data(), out_deg,
                               destinations.data(), destination_weights.data());
      neighbours_set neighbours;
      for (int i{0}; i < in_deg; ++i)
        neighbours.add({destinations[i], destination_weights[i]});
      return neighbours;
    }
  };

  /// Checks if rank-weight pair is equal.
  /// \param pair_1 1st rank-weight pair to compare
  /// \param pair_2 2nd rank-weight pair to compare
  /// \return true if equal
  inline bool operator==(const distributed_graph_communicator::rank_weight_pair &pair_1,
                         const distributed_graph_communicator::rank_weight_pair &pair_2) {
    return pair_1.rank == pair_2.rank and pair_1.weight == pair_2.weight;
  }

  /// Checks if rank-weight pair is not equal.
  /// \param pair_1 1st rank-weight pair to compare
  /// \param pair_2 2nd rank-weight pair to compare
  /// \return true if not equal
  inline bool operator!=(const distributed_graph_communicator::rank_weight_pair &pair_1,
                         const distributed_graph_communicator::rank_weight_pair &pair_2) {
    return pair_1.rank != pair_2.rank or pair_1.weight != pair_2.weight;
  }

}  // namespace mpl

#endif
