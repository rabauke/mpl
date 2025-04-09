#if !(defined MPL_GRAPH_COMMUNICATOR_HPP)

#define MPL_GRAPH_COMMUNICATOR_HPP

#include <mpi.h>
#include <vector>
#include <tuple>
#include <set>
#include <algorithm>
#include <numeric>


namespace mpl {

  /// Communicator with general graph topology.
  class graph_communicator : public impl::topology_communicator {
  public:
    /// Set of edges, pairs of nodes represented by non-negative integers.
    class edge_set : private std::set<std::tuple<int, int>> {
      using base = std::set<std::tuple<int, int>>;

    public:
      using value_type = typename base::value_type;
      using reference = typename base::reference;
      using const_reference = typename base::const_reference;
      using iterator = typename base::iterator;
      using const_iterator = typename base::const_iterator;

      /// Creates an empty set of edges.
      edge_set() = default;


      /// Creates a set of edges given by the list.
      /// \param init set of edges
      edge_set(std::initializer_list<value_type> init) : base(init) {
      }

      using base::operator=;
      using base::begin;
      using base::end;
      using base::cbegin;
      using base::cend;

      /// Determines the number edges.
      /// \return number of edges in the edge set
      [[nodiscard]] int size() const {
        return static_cast<int>(base::size());
      }

      /// Add an edge to the set.
      /// \param edge tuple of two non-negative integers
      void add(const value_type &edge) {
        insert(edge);
      }
    };


    /// Set of nodes represented by integers.
    class node_list : private std::vector<int> {
      using base = std::vector<int>;
      using base::data;

    public:
      using value_type = typename base::value_type;
      using reference = typename base::reference;
      using const_reference = typename base::const_reference;
      using iterator = typename base::iterator;
      using const_iterator = typename base::const_iterator;

      /// Creates an empty node list.
      node_list() = default;

      /// Creates non-empty list of nodes.
      /// \param nodes number of elements of the node list
      explicit node_list(int nodes) : base(nodes, 0) {
      }

      /// Creates non-empty list of nodes with nodes given by the list.
      /// \param init vector components
      node_list(std::initializer_list<int> init) : base(init) {
      }

      using base::operator=;
      using base::begin;
      using base::end;
      using base::cbegin;
      using base::cend;

      /// Determines the number of nodes.
      /// \return number of nodes in the list
      [[nodiscard]] int size() const {
        return static_cast<int>(base::size());
      }

      /// Access a list element.
      /// \param index non-negative index to the list element
      reference operator[](int index) {
        return base::operator[](index);
      }

      /// Access a list element.
      /// \param index non-negative index to the list element
      const_reference operator[](int index) const {
        return base::operator[](index);
      }

      /// Add an element to the end of the node list.
      /// \param node the node that is added
      void add(int node) {
        push_back(node);
      }

      friend class graph_communicator;
    };


    /// Creates an empty communicator with no associated process.
    graph_communicator() = default;

    /// Creates a new communicator which is equivalent to an existing one.
    /// \param other the other communicator to copy from
    /// \note This is a collective operation that needs to be carried out by all processes of
    /// the communicator \c other. Communicators should not be copied unless a new independent
    /// communicator is wanted. Communicators should be passed via references to functions to
    /// avoid unnecessary copying.
    graph_communicator(const graph_communicator &other) {
      MPI_Comm_dup(other.comm_, &comm_);
    }

    /// Move-constructs a communicator.
    /// \param other the other communicator to move from
    graph_communicator(graph_communicator &&other) noexcept {
      comm_ = other.comm_;
      other.comm_ = MPI_COMM_SELF;
    }

    /// Creates a new communicator with graph process topology.
    /// \param other communicator containing the processes to use in the creation of the new
    /// communicator
    /// \param edges represents graph edges of the new communicator
    /// \param reorder indicates if reordering is permitted, if false each process will have the
    /// same rank in the new communicator as in the old one
    /// \note This is a collective operation that needs to be carried out by all processes of
    /// the communicator \c other with the same arguments.
    explicit graph_communicator(const communicator &other, const edge_set &edges,
                                bool reorder = true) {
      int nodes{0};
      for (const auto &e : edges) {
#if defined MPL_DEBUG
        if (std::get<0>(e) < 0 or std::get<1>(e) < 0)
          throw invalid_argument();
#endif
        nodes = std::max({nodes, std::get<0>(e) + 1, std::get<1>(e) + 1});
      }
      std::vector<int> edges_list, index(nodes, 0);
      edges_list.reserve(edges.size());
      // the following works because the edge set is ordered with respect to the pairs of
      // edge-node numbers
      for (const auto &e : edges) {
        edges_list.push_back(std::get<1>(e));
        ++index[std::get<0>(e)];
      }
      std::partial_sum(index.begin(), index.end(), index.begin());
      MPI_Graph_create(other.comm_, nodes, index.data(), edges_list.data(), reorder, &comm_);
    }

    /// Copy-assigns and creates a new communicator with graph process topology which
    /// is equivalent to an existing one.
    /// \param other the other communicator to copy from
    /// \note This is a collective operation that needs to be carried out by all processes of
    /// the communicator \c other. Communicators should not be copied unless a new independent
    /// communicator is wanted. Communicators should be passed via references to functions to
    /// avoid unnecessary copying.
    graph_communicator &operator=(const graph_communicator &other) noexcept {
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
    /// the communicator \c other.
    graph_communicator &operator=(graph_communicator &&other) noexcept {
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

    /// Determines the number of neighbours of some process.
    /// \param rank process rank
    /// \return number of direct neighbours of the process with the given rank
    [[nodiscard]] int degree(int rank) const {
      int n_neighbors;
      MPI_Graph_neighbors_count(this->comm_, rank, &n_neighbors);
      return n_neighbors;
    };

    /// Determines the number of neighbours of the calling process.
    /// \return number of direct neighbours of the calling process
    [[nodiscard]] int degree() const {
      return degree(rank());
    };

    /// Determines the neighbours of some process.
    /// \param rank process rank
    /// \return direct neighbours of the process with the given rank
    [[nodiscard]] node_list neighbors(int rank) const {
      const int max_neighbors{degree(rank)};
      node_list nl(max_neighbors);
      MPI_Graph_neighbors(comm_, rank, max_neighbors, nl.data());
      return nl;
    }

    /// Determines the neighbours of the calling process.
    /// \return direct neighbours of the calling process
    [[nodiscard]] node_list neighbors() const {
      return neighbors(rank());
    }
  };


  /// Equality test for node lists.
  /// \param l_1 1st node list to compare
  /// \param l_2 2nd node list to compare
  /// \return true if equal
  inline bool operator==(const graph_communicator::node_list &l_1,
                         const graph_communicator::node_list &l_2) {
    return l_1.size() == l_2.size() and std::equal(l_1.begin(), l_1.end(), l_2.begin());
  }


  /// Inequality test for node lists.
  /// Equality test for node lists.
  /// \param l_1 1st node list to compare
  /// \param l_2 2nd node list to compare
  /// \return true if not equal
  inline bool operator!=(const graph_communicator::node_list &l_1,
                         const graph_communicator::node_list &l_2) {
    return not(l_1 == l_2);
  }

}  // namespace mpl

#endif
