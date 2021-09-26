#if !(defined MPL_CARTESIAN_COMMUNICATOR_HPP)

#define MPL_CARTESIAN_COMMUNICATOR_HPP

#include <mpi.h>
#include <algorithm>
#include <vector>
#include <tuple>

namespace mpl {

  /// \brief Helper class to represent source and destination ranks within a Cartesian
  /// communicator.
  /// \see cartesian_communicator::shift
  struct shift_ranks {
    int source{0};
    int destination{0};
  };

  //--------------------------------------------------------------------

  /// \brief Communicator with Cartesian topology.
  class cartesian_communicator : public impl::topology_communicator {
  public:
    /// \brief Periodicity indicator for a dimension in a Cartesian process topology.
    enum class periodicity_tag {
      /// dimension is non-periodic
      non_periodic,
      /// dimension is periodic
      periodic
    };

    /// indicates that a dimension in a Cartesian process topology is non-periodic
    static constexpr periodicity_tag non_periodic = periodicity_tag::non_periodic;
    /// indicates that a dimension in a Cartesian process topology is periodic
    static constexpr periodicity_tag periodic = periodicity_tag::periodic;


    /// \brief Inclusion indicator that is employed when creating a new communicator with
    /// Cartesian process topology
    enum class included_tag : int {
      /// dimension is excluded from the new communicator
      excluded = 0,
      /// dimension is included in the new communicator
      included = 1
    };

    /// indicates that a dimension is excluded from the new communicator
    static constexpr included_tag excluded = included_tag::excluded;
    /// indicates that a dimension is included in the new communicator
    static constexpr included_tag included = included_tag::included;


    /// \brief Represents a coordinate (position) in a Cartesian process topology.
    class coordinate_type : private std::vector<int> {
      using base = std::vector<int>;

    public:
      using value_type = typename base::value_type;
      using size_type = typename base::size_type;
      using base::base;
      using base::size;
      using base::begin;
      using base::end;
      using base::cbegin;
      using base::cend;
      using base::data;
      using base::push_back;
      using base::operator=;
      using base::operator[];
    };


    /// \brief Represents the inclusion or exclusion along all along dimensions of a Cartesian
    /// process topology when creating an new communicator.
    class included_tags : private std::vector<included_tag> {
      using base = std::vector<included_tag>;

    public:
      using value_type = typename base::value_type;
      using size_type = typename base::size_type;
      using base::base;
      using base::size;
      using base::begin;
      using base::end;
      using base::cbegin;
      using base::cend;
      using base::data;
      using base::push_back;
      using base::operator=;
      using base::operator[];
    };


    /// \brief Characterizes the dimensionality, size and periodicity of a communicator with
    /// Cartesian process topology.
    class dimensions {
      std::vector<int> dims_, periodic_;

    public:
      /// \brief Type to represent the dimensionality of a communicator with Cartesian process
      /// topology.
      using size_type = std::vector<int>::size_type;

      class dimension_periodicity_proxy {
        int &dim_;
        int &is_periodic_;

        dimension_periodicity_proxy(int &dim, int &is_periodic)
            : dim_{dim}, is_periodic_{is_periodic} {}

      public:
        template<std::size_t N>
        [[nodiscard]] decltype(auto) get() const {
          if constexpr (N == 0)
            return dim_ * 1;
          else if constexpr (N == 1)
            return is_periodic_ == 0 ? periodicity_tag::non_periodic
                                     : periodicity_tag::periodic;
        }

        dimension_periodicity_proxy &operator=(const std::tuple<int, periodicity_tag> &t) {
          dim_ = std::get<0>(t);
          is_periodic_ = std::get<1>(t) == periodic;
          return *this;
        }

        friend dimensions;
      };


      class const_dimension_periodicity_proxy {
        const int &dim_;
        const int &is_periodic_;

        const_dimension_periodicity_proxy(const int &dim, const int &is_periodic)
            : dim_{dim}, is_periodic_{is_periodic} {}

      public:
        template<std::size_t N>
        [[nodiscard]] decltype(auto) get() const {
          if constexpr (N == 0)
            return dim_ * 1;
          else if constexpr (N == 1)
            return is_periodic_ == 0 ? periodicity_tag::non_periodic
                                     : periodicity_tag::periodic;
        }

        friend dimensions;
      };


      /// \brief Constructs a new empty dimensions object.
      dimensions() = default;

      /// \brief Constructs a new dimensions object.
      /// \param size dimensionality (number of Cartesian dimensions)
      /// \details Characterizes a communicator with Cartesian process topology. Its dimension
      /// equals the give parameter. Along all dimensions, no periodicity is defined.  The size,
      /// i.e., the number of processes, along each dimension is zero.
      /// \note A dimension object that is created by this constructor must be passed to
      /// \ref dims_create before a new Cartesian communicator can be created.
      explicit dimensions(int size) : dims_(size, 0), periodic_(size, 0) {}

      /// \brief Constructs a new dimensions object.
      /// \details Characterizes a communicator with Cartesian process topology. Its dimension
      /// equals the number of list elements. The periodicity along the i-th dimension is given
      /// by the i-th list element.  The size, i.e., the number of processes, along each
      /// dimension is zero.
      dimensions(std::initializer_list<periodicity_tag> list) {
        dims_.reserve(list.size());
        periodic_.reserve(list.size());
        for (const auto &the_periodicity : list)
          add(0, the_periodicity);
      }

      /// \brief Constructs a new dimensions object.
      /// \details Characterizes a communicator with Cartesian process topology. Its dimension
      /// equals the number of list elements. The size and the periodicity along the i-th
      /// dimension is given by the i-th list element.
      dimensions(std::initializer_list<std::tuple<int, periodicity_tag>> list) {
        dims_.reserve(list.size());
        periodic_.reserve(list.size());
        for (const auto &[the_size, the_periodicity] : list) {
          add(the_size, the_periodicity);
        }
      }

      /// \brief Adds a additional dimension to a list of dimensions.
      /// \param size the size of the new dimension
      /// \param periodicity the periodicity of the new dimension
      void add(int size, periodicity_tag periodicity) {
        dims_.push_back(size);
        periodic_.push_back(periodicity == periodicity_tag::periodic);
      }

      /// \brief Determines the dimensionality.
      /// \return dimensionality (number of dimensions)
      [[nodiscard]] int dimensionality() const { return static_cast<int>(dims_.size()); }

      /// \brief Determines the number of processes along a dimension.
      /// \param dimension the rank of the dimension
      /// \return the number of processes
      [[nodiscard]] int size(size_type dimension) const { return dims_[dimension]; }

      /// \brief Determines the periodicity of a dimension.
      /// \param dimension the rank of the dimension
      /// \return the periodicity
      [[nodiscard]] periodicity_tag periodicity(size_type dimension) const {
        return periodic_[dimension] == 0 ? non_periodic : periodic;
      }

      /// \brief Determines number of processes along a dimension and the periodicity of a
      /// dimension.
      /// \param dimension the rank of the dimension
      /// \return the number of processes adn the periodicity
      [[nodiscard]] const_dimension_periodicity_proxy operator[](size_type dimension) const
      {
        return {dims_[dimension], periodic_[dimension]};
      }

      /// \brief Determines number of processes along a dimension and the periodicity of a
      /// dimension.
      /// \param dimension the rank of the dimension
      /// \return the number of processes adn the periodicity
      [[nodiscard]] dimension_periodicity_proxy operator[](size_type i)
      {
        return {dims_[i], periodic_[i]};
      }

      friend class cartesian_communicator;
      friend dimensions dims_create(int, dimensions);
    };


    /// \brief Creates an empty communicator with no associated process.
    cartesian_communicator() = default;

    /// \brief Creates a new communicator with Cartesian process topology.
    /// \param other communicator containing the processes to use in the creation of the new
    /// communicator
    /// \param dims represents the dimensional information of the process grid
    /// \param reorder indicates if reordering is permitted, if false each process will have the
    /// same rank in the new communicator as in the old one
    /// \note This is a collective operation that needs to be carried out by all processes of
    /// the communicator other with the same arguments.
    explicit cartesian_communicator(const communicator &other, const dimensions &dims,
                                    bool reorder = true) {
      MPI_Cart_create(other.comm_, dims.dims_.size(), dims.dims_.data(), dims.periodic_.data(),
                      reorder, &comm_);
    }

    /// \brief Creates a new communicator with Cartesian process topology by partitioning a
    /// Cartesian topology.
    /// \param other communicator containing the processes to use in the creation of the new
    /// communicator
    /// \param is_included indicates along which dimensions to arrange sub-grids
    /// \note This is a collective operation that needs to be carried out by all processes of
    /// the communicator other with the same arguments.
    explicit cartesian_communicator(const cartesian_communicator &other,
                                    const included_tags &is_included) {
#if defined MPL_DEBUG
      if (is_included.size() != other.dimensionality())
        throw invalid_size();
#endif
      MPI_Cart_sub(other.comm_, reinterpret_cast<const int *>(is_included.data()), &comm_);
    }

    /// \brief Move-constructs a communicator.
    /// \param other the other communicator to move from
    cartesian_communicator(cartesian_communicator &&other) noexcept {
      comm_ = other.comm_;
      other.comm_ = MPI_COMM_SELF;
    }

    /// \brief Copy-assigns and creates a new communicator with Cartesian process topology which
    /// is equivalent to an existing one.
    /// \param other the other communicator to copy from
    /// \note This is a collective operation that needs to be carried out by all processes of
    /// the communicator other. Communicators should not be copied unless a new independent
    /// communicator is wanted. Communicators should be passed via references to functions to
    /// avoid unnecessary copying.
    cartesian_communicator &operator=(const cartesian_communicator &other) noexcept {
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

    /// \brief Move-assigns a communicator.
    /// \param other the other communicator to move from
    /// \note This is a collective operation that needs to be carried out by all processes of
    /// the communicator other.
     cartesian_communicator &operator=(cartesian_communicator &&other) noexcept {
      if (this != &other) {
        int result_1{0}, result_2{0};
        MPI_Comm_compare(comm_, MPI_COMM_WORLD, &result_1);
        MPI_Comm_compare(comm_, MPI_COMM_SELF, &result_2);
        if (result_1 != MPI_IDENT and result_2 != MPI_IDENT)
          MPI_Comm_free(&comm_);
        comm_ = other.comm_;
        other.comm_ = MPI_COMM_SELF;
      }
      return *this;
    }

    /// \brief Determines the communicator's dimensionality.
    /// \return number of dimensions of the Cartesian topology
    [[nodiscard]] int dimensionality() const {
      int t_dimensionality{0};
      MPI_Cartdim_get(this->comm_, &t_dimensionality);
      return t_dimensionality;
    }

    using communicator::rank;

    /// \brief Determines process rank of a process at a given Cartesian location.
    /// \param coordinate Cartesian location
    /// \return process rank
    [[nodiscard]] int rank(const coordinate_type &coordinate) const {
      int t_rank{0};
      MPI_Cart_rank(this->comm_, coordinate.data(), &t_rank);
      return t_rank;
    }

    /// \brief Determines the Cartesian location of a process with a given rank.
    /// \param rank process rank
    /// \return Cartesian location
    [[nodiscard]] coordinate_type coordinate(int rank) const {
      coordinate_type coordinates(dimensionality());
      MPI_Cart_coords(comm_, rank, coordinates.size(), coordinates.data());
      return coordinates;
    }

    /// \brief Determines the Cartesian location of this process.
    /// \return Cartesian location
    [[nodiscard]] coordinate_type coordinate() const {
      const int t_dimensionality{dimensionality()};
      dimensions t_dimensions(t_dimensionality);
      coordinate_type t_coordinate(t_dimensionality);
      MPI_Cart_get(comm_, t_dimensionality, t_dimensions.dims_.data(),
                   t_dimensions.periodic_.data(),
                   t_coordinate.data());
      return t_coordinate;
    }

    /// \brief Determines the size and the periodicity of each dimension of the communicator with
    /// Cartesian topology.
    /// \return size and periodicity of each dimension
    [[nodiscard]] dimensions get_dimensions() const {
      const int t_dimensionality{dimensionality()};
      dimensions t_dimensions(t_dimensionality);
      coordinate_type t_coordinate(t_dimensionality);
      MPI_Cart_get(comm_, t_dimensionality, t_dimensions.dims_.data(),
                   t_dimensions.periodic_.data(),
                   t_coordinate.data());
      return t_dimensions;
    }

    /// \brief Finds the ranks of processes that can be reached by shifting the Cartesian grid.
    /// \param direction shift direction
    /// \param displacement shift size
    /// \details This method permits to find the two processes that would respectively reach,
    /// and be reached by, the calling process by shifting the Cartesian grid by the given
    /// number of displacements along the given direction. In case no such process exists,
    /// located outside the boundaries of a non-periodic dimension for instance, proc_null is
    /// returned instead.
    [[nodiscard]] shift_ranks shift(int direction, int displacement) const {
      shift_ranks ranks;
      MPI_Cart_shift(this->comm_, direction, displacement, &ranks.source, &ranks.destination);
      return ranks;
    }
  };

  //--------------------------------------------------------------------

  /// \brief Decomposes a given number of processes over a Cartesian grid made of the number of
  /// dimensions specified.
  /// \param size total number of processes (the size of the communicator)
  /// \param dims dimension object indicating possible restrictions for the process partitioning
  /// \return dimension object
  /// \details The method attempts to balance the distribution by minimising the difference in
  /// the number of processes assigned to each dimension. One can restrict the number of process
  /// to allocate to any dimension by specifying a non-zero size for a given dimension in the
  /// parameter dims. If the method is not able to find a decomposition while respecting
  /// the restrictions given, the routine throws an exception invalid_dims.
  inline cartesian_communicator::dimensions dims_create(
      int size, cartesian_communicator::dimensions dims) {
    if (MPI_Dims_create(size, static_cast<int>(dims.dims_.size()), dims.dims_.data()) !=
        MPI_SUCCESS)
      throw invalid_dim();
    return dims;
  }

}  // namespace mpl


namespace std {
  template<>
  struct tuple_size<mpl::cartesian_communicator::dimensions::dimension_periodicity_proxy>
      : std::integral_constant<std::size_t, 2> {};

  template<std::size_t N>
  struct tuple_element<N,
                       mpl::cartesian_communicator::dimensions::dimension_periodicity_proxy> {
    using type =
        decltype(std::declval<
                     mpl::cartesian_communicator::dimensions::dimension_periodicity_proxy>()
                     .get<N>());
  };
}  // namespace std

#endif
