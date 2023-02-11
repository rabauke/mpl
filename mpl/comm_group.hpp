#if !(defined MPL_COMM_GROUP_HPP)

#define MPL_COMM_GROUP_HPP

#include <mpi.h>
#include <type_traits>
#include <thread>
#include <optional>
#include <mpl/layout.hpp>
#include <mpl/vector.hpp>
#include <mpl/command_line.hpp>
#include <mpl/info.hpp>

namespace mpl {

  class group;

  class communicator;

  class cartesian_communicator;

  class graph_communicator;

  class inter_communicator;

  class file;

  namespace environment::detail {

    class env;

  }  // namespace environment::detail

  //--------------------------------------------------------------------

  /// Return value of matching probe operations.
  struct mprobe_status {
    /// message handle to be used in a matching receive operation
    message_t message;
    /// status of the pending incoming message
    status_t status;
  };

  //--------------------------------------------------------------------

  /// Represents a group of processes.
  class group {
    MPI_Group gr_{MPI_GROUP_EMPTY};

  public:
    /// Group equality types.
    enum class equality_type {
      /// groups are identical, i.e., groups have same the members in same rank order
      identical = MPI_IDENT,
      /// groups are similar, i.e., groups have same tha members in different rank order
      similar = MPI_SIMILAR,
      /// groups are unequal, i.e., groups have different sets of members
      unequal = MPI_UNEQUAL
    };

    /// Indicates that groups are identical, i.e., groups have same the members in same rank
    /// order.
    static constexpr equality_type identical = equality_type::identical;
    /// Indicates that groups are similar, i.e., groups have same tha members in different rank
    /// order.
    static constexpr equality_type similar = equality_type::similar;
    /// Indicates that groups are unequal, i.e., groups have different sets of members.
    static constexpr equality_type unequal = equality_type::unequal;

    /// Indicates the creation of a union of two groups.
    class Union_tag {};
    /// Indicates the creation of a union of two groups.
    static constexpr Union_tag Union{};

    /// Indicates the creation of an intersection of two groups.
    class intersection_tag {};
    /// Indicates the creation of an intersection of two groups.
    static constexpr intersection_tag intersection{};

    /// Indicates the creation of a difference of two groups.
    class difference_tag {};
    /// Indicates the creation of a difference of two groups.
    static constexpr difference_tag difference{};

    /// Indicates the creation of a subgroup by including members of an existing group.
    class include_tag {};
    /// Indicates the creation of a subgroup by including members of an existing group.
    static constexpr include_tag include{};

    /// Indicates the creation of a subgroup by excluding members of an existing group.
    class exclude_tag {};
    /// Indicates the creation of a subgroup by excluding members of an existing group.
    static constexpr exclude_tag exclude{};

    /// Creates an empty process group.
    group() = default;

    /// Creates a new process group by copying an existing one.
    /// \param other the other group to copy from
    /// \note Process groups should not be copied unless a new independent group is wanted.
    /// Process groups should be passed via references to functions to avoid unnecessary
    /// copying.
    group(const group &other);

    /// Move-constructs a process group.
    /// \param other the other group to move from
    group(group &&other) noexcept : gr_{other.gr_} {
      other.gr_ = MPI_GROUP_EMPTY;
    }

    /// Creates a new group that consists of all processes of the given communicator.
    /// \param comm the communicator
    explicit group(const communicator &comm);

    /// Creates a new group that consists of all processes of the local group of the
    /// given inter-communicator.
    /// \param comm the inter-communicator
    explicit group(const inter_communicator &comm);

    /// Creates a new group that consists of all processes of the local group of the
    /// given %file.
    /// \param f the file
    explicit group(const file &f);

    /// Creates a new group that consists of the union of two existing process groups.
    /// \param tag indicates the unification of two existing process groups
    /// \param other_1 first existing process group
    /// \param other_2 second existing process group
    explicit group(Union_tag tag, const group &other_1, const group &other_2);

    /// Creates a new group that consists of the intersection of two existing process
    /// groups.
    /// \param tag indicates the intersection of two existing process groups
    /// \param other_1 first existing process group
    /// \param other_2 second existing process group
    explicit group(intersection_tag tag, const group &other_1, const group &other_2);

    /// Creates a new group that consists of the difference of two existing process
    /// groups.
    /// \param tag indicates the difference of two existing process groups
    /// \param other_1 first existing process group
    /// \param other_2 second existing process group
    explicit group(difference_tag tag, const group &other_1, const group &other_2);

    /// Creates a new group by including members of an existing process group.
    /// \param tag indicates inclusion from an existing process group
    /// \param other existing process group
    /// \param rank set of ranks to include
    explicit group(include_tag tag, const group &other, const ranks &rank);

    /// Creates a new group by excluding members of an existing process group.
    /// \param tag indicates exclusion from an existing process group
    /// \param other existing process group
    /// \param rank set of ranks to exclude
    explicit group(exclude_tag tag, const group &other, const ranks &rank);

    /// Destructs a process group.
    ~group() {
      int result;
      MPI_Group_compare(gr_, MPI_GROUP_EMPTY, &result);
      if (result != MPI_IDENT)
        MPI_Group_free(&gr_);
    }

    /// Copy-assigns a process group.
    /// \param other the other group to move from
    /// \return this group
    /// \note Process groups should not be copied unless a new independent group is wanted.
    /// Process groups should be passed via references to functions to avoid unnecessary
    /// copying.
    group &operator=(const group &other) {
      if (this != &other) {
        int result;
        MPI_Group_compare(gr_, MPI_GROUP_EMPTY, &result);
        if (result != MPI_IDENT)
          MPI_Group_free(&gr_);
        MPI_Group_excl(other.gr_, 0, nullptr, &gr_);
      }
      return *this;
    }

    /// Move-assigns a process group.
    /// \param other the other group to move from
    /// \return this group
    group &operator=(group &&other) noexcept {
      if (this != &other) {
        int result;
        MPI_Group_compare(gr_, MPI_GROUP_EMPTY, &result);
        if (result != MPI_IDENT)
          MPI_Group_free(&gr_);
        gr_ = other.gr_;
        other.gr_ = MPI_GROUP_EMPTY;
      }
      return *this;
    }

    /// Get the underlying MPI handle of the group.
    /// \return MPI handle of the group
    /// \note This function returns a non-owning handle to the underlying MPI group, which may
    /// be useful when refactoring legacy MPI applications to MPL.
    /// \warning The handle must not be used to modify the MPI group that the handle points
    /// to. This method will be removed in a future version.
    [[nodiscard]] MPI_Group native_handle() const {
      return gr_;
    }

    /// Determines the total number of processes in a process group.
    /// \return number of processes
    [[nodiscard]] int size() const {
      int result;
      MPI_Group_size(gr_, &result);
      return result;
    }

    /// Determines the rank within a process group.
    /// \return the rank of the calling process in the group
    [[nodiscard]] int rank() const {
      int result;
      MPI_Group_rank(gr_, &result);
      return result;
    }

    /// Determines the relative numbering of the same process in two different groups.
    /// \param rank a valid rank in the given process group
    /// \param other process group
    /// \return corresponding rank in this process group
    [[nodiscard]] int translate(int rank, const group &other) const {
      int other_rank;
      MPI_Group_translate_ranks(gr_, 1, &rank, other.gr_, &other_rank);
      return other_rank;
    }

    /// Determines the relative numbering of the same process in two different groups.
    /// \param rank a set valid ranks in the given process group
    /// \param other process group
    /// \return corresponding ranks in this process group
    [[nodiscard]] ranks translate(const ranks &rank, const group &other) const {
      ranks other_rank(rank.size());
      MPI_Group_translate_ranks(gr_, static_cast<int>(rank.size()), rank(), other.gr_,
                                other_rank());
      return other_rank;
    }

    /// Tests for identity of process groups.
    /// \param other process group to compare with
    /// \return true if identical
    bool operator==(const group &other) const {
      int result;
      MPI_Group_compare(gr_, other.gr_, &result);
      return result == MPI_IDENT;
    }

    /// Tests for identity of process groups.
    /// \param other process group to compare with
    /// \return true if not identical
    bool operator!=(const group &other) const {
      int result;
      MPI_Group_compare(gr_, other.gr_, &result);
      return result != MPI_IDENT;
    }

    /// Compares to another process group.
    /// \param other process group to compare with
    /// \return equality type
    [[nodiscard]] equality_type compare(const group &other) const {
      int result;
      MPI_Group_compare(gr_, other.gr_, &result);
      return static_cast<equality_type>(result);
    }

    friend class communicator;
  };

  //--------------------------------------------------------------------

  namespace impl {

    class base_communicator {
    protected:
      struct isend_irecv_state {
        MPI_Request req{};
        int source{MPI_ANY_SOURCE};
        int tag{MPI_ANY_TAG};
        MPI_Datatype datatype{MPI_DATATYPE_NULL};
        int count{MPI_UNDEFINED};
      };

      static int isend_irecv_query(void *state, MPI_Status *s) {
        isend_irecv_state *sendrecv_state{static_cast<isend_irecv_state *>(state)};
        MPI_Status_set_elements(s, sendrecv_state->datatype, sendrecv_state->count);
        MPI_Status_set_cancelled(s, 0);
        s->MPI_SOURCE = sendrecv_state->source;
        s->MPI_TAG = sendrecv_state->tag;
        return MPI_SUCCESS;
      }

      static int isend_irecv_free(void *state) {
        isend_irecv_state *sendrecv_state{static_cast<isend_irecv_state *>(state)};
        delete sendrecv_state;
        return MPI_SUCCESS;
      }

      static int isend_irecv_cancel([[maybe_unused]] void *state,
                                    [[maybe_unused]] int complete) {
        return MPI_SUCCESS;
      }

      void check_dest([[maybe_unused]] int dest) const {
#if defined MPL_DEBUG
        if (dest != proc_null and (dest < 0 or dest >= size()))
          throw invalid_rank();
#endif
      }

      void check_source([[maybe_unused]] int source) const {
#if defined MPL_DEBUG
        if (source != proc_null and source != any_source and (source < 0 or source >= size()))
          throw invalid_rank();
#endif
      }

      void check_send_tag([[maybe_unused]] tag_t t) const {
#if defined MPL_DEBUG
        if (static_cast<int>(t) < 0 or static_cast<int>(t) > static_cast<int>(tag_t::up()))
          throw invalid_tag();
#endif
      }

      void check_recv_tag([[maybe_unused]] tag_t t) const {
#if defined MPL_DEBUG
        if (static_cast<int>(t) != static_cast<int>(tag_t::any()) and
            (static_cast<int>(t) < 0 or static_cast<int>(t) > static_cast<int>(tag_t::up())))
          throw invalid_tag();
#endif
      }

      void check_root([[maybe_unused]] int root_rank) const {
#if defined MPL_DEBUG
        if ((root_rank < 0 or root_rank >= size()) and root_rank != mpl::root and
            root_rank != mpl::proc_null)
          throw invalid_rank();
#endif
      }

      void check_nonroot([[maybe_unused]] int root_rank) const {
#if defined MPL_DEBUG
        check_nonroot(root_rank);
        if (root_rank == rank())
          throw invalid_rank();
#endif
      }

      template<typename T>
      void check_size([[maybe_unused]] const layouts<T> &l) const {
#if defined MPL_DEBUG
        if (static_cast<int>(l.size()) > size())
          throw invalid_size();
#endif
      }

      void check_size([[maybe_unused]] const displacements &d) const {
#if defined MPL_DEBUG
        if (static_cast<int>(d.size()) > size())
          throw invalid_size();
#endif
      }

      void check_count([[maybe_unused]] int count) const {
#if defined MPL_DEBUG
        if (count == MPI_UNDEFINED)
          throw invalid_count();
#endif
      }

      template<typename T>
      void check_container_size([[maybe_unused]] const T &container,
                                detail::basic_or_fixed_size_type) const {
      }

      template<typename T>
      void check_container_size([[maybe_unused]] const T &container,
                                detail::stl_container) const {
#if defined MPL_DEBUG
        if (container.size() >
            static_cast<decltype(container.size())>(std::numeric_limits<int>::max()))
          throw invalid_count();
#endif
      }

      template<typename T>
      void check_container_size(const T &container) const {
        check_container_size(container,
                             typename detail::datatype_traits<T>::data_type_category{});
      }

      MPI_Comm comm_{MPI_COMM_NULL};

    public:
      /// Indicates the creation of a new communicator by an operation that is collective
      /// for all processes in the given communicator.
      class comm_collective_tag {};
      /// Indicates the creation of a new communicator by an operation that is collective
      /// for all processes in the given communicator.
      static constexpr comm_collective_tag comm_collective{};

      /// Indicates the creation of a new communicator by an operation that is collective
      /// for all processes in the given group.
      class group_collective_tag {};
      /// Indicates the creation of a new communicator by an operation that is collective
      /// for all processes in the given group.
      static constexpr group_collective_tag group_collective{};

      /// Indicates the creation of a new communicator by spitting an existing communicator
      /// into disjoint subgroups.
      class split_tag {};
      /// Indicates the creation of a new communicator by spitting an existing communicator
      /// into disjoint subgroups.
      static constexpr split_tag split{};

      /// Indicates the creation of a new communicator by spitting an existing communicator
      /// into disjoint subgroups each of which can create a shared memory region.
      class split_shared_memory_tag {};
      /// Indicates the creation of a new communicator by spitting an existing communicator
      /// into disjoint subgroups each of which can create a shared memory region.
      static constexpr split_shared_memory_tag split_shared_memory{};

    protected:
      base_communicator() = default;
      explicit base_communicator(MPI_Comm comm) : comm_(comm) {
      }

      ~base_communicator() {
        if (is_valid()) {
          int result_1;
          MPI_Comm_compare(comm_, MPI_COMM_WORLD, &result_1);
          int result_2;
          MPI_Comm_compare(comm_, MPI_COMM_SELF, &result_2);
          if (result_1 != MPI_IDENT and result_2 != MPI_IDENT)
            MPI_Comm_free(&comm_);
        }
      }

      base_communicator &operator=(const base_communicator &other) noexcept {
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

      base_communicator &operator=(base_communicator &&other) noexcept {
        if (this != &other) {
          if (is_valid()) {
            int result_1;
            MPI_Comm_compare(comm_, MPI_COMM_WORLD, &result_1);
            int result_2;
            MPI_Comm_compare(comm_, MPI_COMM_SELF, &result_2);
            if (result_1 != MPI_IDENT and result_2 != MPI_IDENT)
              MPI_Comm_free(&comm_);
          }
          comm_ = other.comm_;
          other.comm_ = MPI_COMM_NULL;
        }
        return *this;
      }

      [[nodiscard]] int size() const {
        int result;
        MPI_Comm_size(comm_, &result);
        return result;
      }

      [[nodiscard]] int rank() const {
        int result;
        MPI_Comm_rank(comm_, &result);
        return result;
      }

      void info(const mpl::info &i) const {
        MPI_Comm_set_info(comm_, i.info_);
      }

      [[nodiscard]] mpl::info info() const {
        MPI_Info i;
        MPI_Comm_get_info(comm_, &i);
        return mpl::info{i};
      }

      bool operator==(const base_communicator &other) const {
        int result;
        MPI_Comm_compare(comm_, other.comm_, &result);
        return result == MPI_IDENT;
      }

      bool operator!=(const base_communicator &other) const {
        int result;
        MPI_Comm_compare(comm_, other.comm_, &result);
        return result != MPI_IDENT;
      }

    public:
      /// Get the underlying MPI handle of the communicator.
      /// \return MPI handle of the communicator
      /// \note This function returns a non-owning handle to the underlying MPI communicator,
      /// which may be useful when refactoring legacy MPI applications to MPL.
      /// \warning The handle must not be used to modify the MPI communicator that the handle
      /// points to. This method will be removed in a future version.
      [[nodiscard]] MPI_Comm native_handle() const {
        return comm_;
      }

      /// Checks if a communicator is valid, i.e., is not an empty communicator with no
      /// associated process.
      /// \return true if communicator is valid
      /// \note A default constructed communicator is a non valid communicator.
      [[nodiscard]] bool is_valid() const {
        return comm_ != MPI_COMM_NULL;
      }

      /// Aborts all processes associated to the communicator.
      /// \param err error code, becomes the return code of the main program
      /// \note Method provides just a "best attempt" to abort processes.
      void abort(int err) const {
        MPI_Abort(comm_, err);
      }

      // === point to point ==============================================

      // === standard send ===
      // --- blocking standard send ---
    private:
      template<typename T>
      void send(const T &data, int destination, tag_t t,
                detail::basic_or_fixed_size_type) const {
        MPI_Send(&data, 1, detail::datatype_traits<T>::get_datatype(), destination,
                 static_cast<int>(t), comm_);
      }

      template<typename T>
      void send(const T &data, int destination, tag_t t,
                detail::contiguous_const_stl_container) const {
        using value_type = typename T::value_type;
        const vector_layout<value_type> l(data.size());
        send(data.size() > 0 ? &data[0] : nullptr, l, destination, t);
      }

      template<typename T>
      void send(const T &data, int destination, tag_t t, detail::stl_container) const {
        using value_type = detail::remove_const_from_members_t<typename T::value_type>;
        detail::vector<value_type> serial_data(data.size(), std::begin(data));
        const vector_layout<value_type> l(serial_data.size());
        send(serial_data.data(), l, destination, t);
      }

    public:
      /// Sends a message with a single value via a blocking standard send operation.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section or an STL container
      /// that holds elements that comply with the mentioned requirements
      /// \param data value to send
      /// \param destination rank of the receiving process
      /// \param t tag associated to this message
      /// \note Sending STL containers is a convenience feature, which may have non-optimal
      /// performance characteristics. Use alternative overloads in performance critical code
      /// sections.
      template<typename T>
      void send(const T &data, int destination, tag_t t = tag_t(0)) const {
        check_dest(destination);
        check_send_tag(t);
        check_container_size(data);
        send(data, destination, t, typename detail::datatype_traits<T>::data_type_category{});
      }

      /// Sends a message with a several values having a specific memory layout via a
      /// blocking standard send operation.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param data pointer to the data to send
      /// \param l memory layout of the data to send
      /// \param destination rank of the receiving process
      /// \param t tag associated to this message
      template<typename T>
      void send(const T *data, const layout<T> &l, int destination, tag_t t = tag_t(0)) const {
        check_dest(destination);
        check_send_tag(t);
        MPI_Send(data, 1, detail::datatype_traits<layout<T>>::get_datatype(l), destination,
                 static_cast<int>(t), comm_);
      }

      /// Sends a message with a several values given by a pair of iterators via a
      /// blocking standard send operation.
      /// \tparam iterT iterator type, must fulfill the requirements of a
      /// <a
      /// href="https://en.cppreference.com/w/cpp/named_req/ForwardIterator">LegacyForwardIterator</a>,
      /// the iterator's value-type must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim  section
      /// \param begin iterator pointing to the first data value to send
      /// \param end iterator pointing one element beyond the last data value to send
      /// \param destination rank of the receiving process
      /// \param t tag associated to this message
      /// \note This is a convenience method, which may have non-optimal performance
      /// characteristics. Use alternative overloads in performance critical code sections.
      template<typename iterT>
      void send(iterT begin, iterT end, int destination, tag_t t = tag_t(0)) const {
        using value_type = typename std::iterator_traits<iterT>::value_type;
        static_assert(std::is_reference_v<decltype(*begin)>,
                      "iterator de-referencing must yield a reference");
        if constexpr (detail::is_contiguous_iterator_v<iterT>) {
          const vector_layout<value_type> l(std::distance(begin, end));
          send(&(*begin), l, destination, t);
        } else {
          const iterator_layout<value_type> l(begin, end);
          send(&(*begin), l, destination, t);
        }
      }

      // --- non-blocking standard send ---
    private:
      template<typename T>
      base_irequest isend(const T &data, int destination, tag_t t,
                          detail::basic_or_fixed_size_type) const {
        MPI_Request req;
        MPI_Isend(&data, 1, detail::datatype_traits<T>::get_datatype(), destination,
                  static_cast<int>(t), comm_, &req);
        return base_irequest{req};
      }

      template<typename T>
      void isend(const T &data, int destination, tag_t t, isend_irecv_state *isend_state,
                 detail::contiguous_const_stl_container) const {
        using value_type = typename T::value_type;
        const int count(data.size());
        MPI_Datatype datatype{detail::datatype_traits<value_type>::get_datatype()};
        MPI_Request req;
        MPI_Isend(data.size() > 0 ? &data[0] : nullptr, count, datatype, destination,
                  static_cast<int>(t), comm_, &req);
        MPI_Status s;
        MPI_Wait(&req, &s);
        isend_state->source = s.MPI_SOURCE;
        isend_state->tag = s.MPI_TAG;
        isend_state->datatype = datatype;
        isend_state->count = 0;
        MPI_Grequest_complete(isend_state->req);
      }

      template<typename T>
      void isend(const T &data, int destination, tag_t t, isend_irecv_state *isend_state,
                 detail::stl_container) const {
        using value_type = detail::remove_const_from_members_t<typename T::value_type>;
        const detail::vector<value_type> serial_data(data.size(), std::begin(data));
        isend(serial_data, destination, t, isend_state,
              detail::contiguous_const_stl_container{});
      }

      template<typename T, typename C>
      base_irequest isend(const T &data, int destination, tag_t t, C) const {
        isend_irecv_state *send_state{new isend_irecv_state()};
        MPI_Request req;
        MPI_Grequest_start(isend_irecv_query, isend_irecv_free, isend_irecv_cancel, send_state,
                           &req);
        send_state->req = req;
        std::thread thread([this, &data, destination, t, send_state]() {
          isend(data, destination, t, send_state, C{});
        });
        thread.detach();
        return base_irequest{req};
      }

    public:
      /// Sends a message with a single value via a non-blocking standard send operation.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section or an STL container
      /// that holds elements that comply with the mentioned requirements
      /// \param data value to send
      /// \param destination rank of the receiving process
      /// \param t tag associated to this message
      /// \return request representing the ongoing message transfer
      /// \note Sending STL containers is a  convenience feature, which may have non-optimal
      /// performance characteristics. Use alternative overloads in performance critical code
      /// sections.
      template<typename T>
      irequest isend(const T &data, int destination, tag_t t = tag_t(0)) const {
        check_dest(destination);
        check_send_tag(t);
        check_container_size(data);
        return isend(data, destination, t,
                     typename detail::datatype_traits<T>::data_type_category{});
      }

      /// Sends a message with several values having a specific memory layout via a
      /// non-blocking standard send operation.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim  section
      /// \param data pointer to the data to send
      /// \param l memory layout of the data to send
      /// \param destination rank of the receiving process
      /// \param t tag associated to this message
      /// \return request representing the ongoing message transfer
      template<typename T>
      irequest isend(const T *data, const layout<T> &l, int destination,
                     tag_t t = tag_t(0)) const {
        check_dest(destination);
        check_send_tag(t);
        MPI_Request req;
        MPI_Isend(data, 1, detail::datatype_traits<layout<T>>::get_datatype(l), destination,
                  static_cast<int>(t), comm_, &req);
        return base_irequest{req};
      }

      /// Sends a message with a several values given by a pair of iterators via a
      /// non-blocking standard send operation.
      /// \tparam iterT iterator type, must fulfill the requirements of a
      /// <a
      /// href="https://en.cppreference.com/w/cpp/named_req/ForwardIterator">LegacyForwardIterator</a>,
      /// the iterator's value-type must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param begin iterator pointing to the first data value to send
      /// \param end iterator pointing one element beyond the last data value to send
      /// \param destination rank of the receiving process
      /// \param t tag associated to this message
      /// \note This is a convenience method, which may have non-optimal performance
      /// characteristics. Use alternative overloads in performance critical code sections.
      /// \return request representing the ongoing message transfer
      template<typename iterT>
      irequest isend(iterT begin, iterT end, int destination, tag_t t = tag_t(0)) const {
        using value_type = typename std::iterator_traits<iterT>::value_type;
        static_assert(std::is_reference_v<decltype(*begin)>,
                      "iterator de-referencing must yield a reference");
        if constexpr (detail::is_contiguous_iterator_v<iterT>) {
          const vector_layout<value_type> l(std::distance(begin, end));
          return isend(&(*begin), l, destination, t);
        } else {
          const iterator_layout<value_type> l(begin, end);
          return isend(&(*begin), l, destination, t);
        }
      }

      // --- persistent standard send ---
      /// Creates a persistent communication request to send a message with a single
      /// value via a blocking standard send operation.
      /// \tparam T type of the data to send, must  meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param data value to send
      /// \param destination rank of the receiving process
      /// \param t tag associated to this message
      /// \return persistent communication request
      /// \note Sending STL containers is not supported.
      template<typename T>
      prequest send_init(const T &data, int destination, tag_t t = tag_t(0)) const {
        check_dest(destination);
        check_send_tag(t);
        MPI_Request req;
        MPI_Send_init(&data, 1, detail::datatype_traits<T>::get_datatype(), destination,
                      static_cast<int>(t), comm_, &req);
        return base_prequest{req};
      }

      /// Creates a persistent communication request to send a message with a several
      /// values having a specific memory layout via a blocking standard send operation.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param data pointer to the data to send
      /// \param l memory layout of the data to send
      /// \param destination rank of the receiving process
      /// \param t tag associated to this message
      /// \return persistent communication request
      template<typename T>
      prequest send_init(const T *data, const layout<T> &l, int destination,
                         tag_t t = tag_t(0)) const {
        check_dest(destination);
        check_send_tag(t);
        MPI_Request req;
        MPI_Send_init(data, 1, detail::datatype_traits<layout<T>>::get_datatype(l), destination,
                      static_cast<int>(t), comm_, &req);
        return base_prequest{req};
      }

      /// Creates a persistent communication request to send a message with a several
      /// values given by a pair of iterators via a blocking standard send operation.
      /// \tparam iterT iterator type, must fulfill the requirements of a
      /// <a
      /// href="https://en.cppreference.com/w/cpp/named_req/ForwardIterator">LegacyForwardIterator</a>,
      /// the iterator's value-type must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param begin iterator pointing to the first data value to send
      /// \param end iterator pointing one element beyond the last data value to send
      /// \param destination rank of the receiving process
      /// \param t tag associated to this message
      /// \return persistent communication request
      /// \note This is a convenience method, which may have non-optimal performance
      /// characteristics. Use alternative overloads in performance critical code sections.
      template<typename iterT>
      prequest send_init(iterT begin, iterT end, int destination, tag_t t = tag_t(0)) const {
        using value_type = typename std::iterator_traits<iterT>::value_type;
        if constexpr (detail::is_contiguous_iterator_v<iterT>) {
          const vector_layout<value_type> l(std::distance(begin, end));
          return send_init(&(*begin), l, destination, t);
        } else {
          const iterator_layout<value_type> l(begin, end);
          return send_init(&(*begin), l, destination, t);
        }
      }

      // === buffered send ===
      // --- determine buffer size ---
      /// Determines the message buffer size.
      /// \tparam T type of the data to send in a later buffered send operation, must meet the
      /// requirements as described in the \verbatim embed:rst:inline :doc:`data_types` \endverbatim
      /// section
      /// \param number quantity of elements of type \c T to send in a single buffered message or
      /// in a series of  buffered send operations
      /// \return message buffer size
      template<typename T>
      [[nodiscard]] int bsend_size(int number = 1) const {
        int pack_size{0};
        MPI_Pack_size(number, detail::datatype_traits<T>::get_datatype(), comm_, &pack_size);
        return pack_size + MPI_BSEND_OVERHEAD;
      }

      /// Determines the message buffer size.
      /// \tparam T type of the data to send in a later buffered send operation, must meet the
      /// requirements as described in the \verbatim embed:rst:inline :doc:`data_types` \endverbatim
      /// section
      /// \param l layout of the data
      /// \param number quantity of buffered send operations with the given data type and layout
      /// \return message buffer size
      template<typename T>
      [[nodiscard]] int bsend_size(const layout<T> &l, int number = 1) const {
        int pack_size{0};
        MPI_Pack_size(number, detail::datatype_traits<layout<T>>::get_datatype(l), comm_,
                      &pack_size);
        return pack_size + MPI_BSEND_OVERHEAD;
      }

      // --- blocking buffered send ---
    private:
      template<typename T>
      void bsend(const T &data, int destination, tag_t t,
                 detail::basic_or_fixed_size_type) const {
        MPI_Bsend(&data, 1, detail::datatype_traits<T>::get_datatype(), destination,
                  static_cast<int>(t), comm_);
      }

      template<typename T>
      void bsend(const T &data, int destination, tag_t t,
                 detail::contiguous_const_stl_container) const {
        using value_type = typename T::value_type;
        const vector_layout<value_type> l(data.size());
        bsend(data.size() > 0 ? &data[0] : nullptr, l, destination, t);
      }

      template<typename T>
      void bsend(const T &data, int destination, tag_t t, detail::stl_container) const {
        using value_type = detail::remove_const_from_members_t<typename T::value_type>;
        detail::vector<value_type> serial_data(data.size(), std::begin(data));
        const vector_layout<value_type> l(serial_data.size());
        bsend(serial_data.data(), l, destination, t);
      }

    public:
      /// Sends a message with a single value via a buffered send operation.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section or an STL container
      /// that holds elements that
      /// comply with the mentioned requirements
      /// \param data value to send
      /// \param destination rank of the receiving process
      /// \param t tag associated to this message
      /// \note Sending STL containers is a convenience feature, which may have non-optimal
      /// performance characteristics. Use alternative overloads in performance critical code
      /// sections.
      template<typename T>
      void bsend(const T &data, int destination, tag_t t = tag_t(0)) const {
        check_dest(destination);
        check_send_tag(t);
        check_container_size(data);
        bsend(data, destination, t, typename detail::datatype_traits<T>::data_type_category{});
      }

      /// Sends a message with a several values having a specific memory layout via a
      /// buffered send operation.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim  section
      /// \param data pointer to the data to send
      /// \param l memory layout of the data to send
      /// \param destination rank of the receiving process
      /// \param t tag associated to this message
      template<typename T>
      void bsend(const T *data, const layout<T> &l, int destination, tag_t t = tag_t(0)) const {
        check_dest(destination);
        check_send_tag(t);
        MPI_Bsend(data, 1, detail::datatype_traits<layout<T>>::get_datatype(l), destination,
                  static_cast<int>(t), comm_);
      }

      /// Sends a message with a several values given by a pair of iterators via a
      /// buffered send operation.
      /// \tparam iterT iterator type, must fulfill the requirements of a
      /// <a
      /// href="https://en.cppreference.com/w/cpp/named_req/ForwardIterator">LegacyForwardIterator</a>,
      /// the iterator's value-type must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim  section
      /// \param begin iterator pointing to the first data value to send
      /// \param end iterator pointing one element beyond the last data value to send
      /// \param destination rank of the receiving process
      /// \param t tag associated to this message
      /// \note This is a convenience method, which may have non-optimal performance
      /// characteristics. Use alternative overloads in performance critical code sections.
      template<typename iterT>
      void bsend(iterT begin, iterT end, int destination, tag_t t = tag_t(0)) const {
        using value_type = typename std::iterator_traits<iterT>::value_type;
        static_assert(std::is_reference_v<decltype(*begin)>,
                      "iterator de-referencing must yield a reference");
        if constexpr (detail::is_contiguous_iterator_v<iterT>) {
          const vector_layout<value_type> l(std::distance(begin, end));
          bsend(&(*begin), l, destination, t);
        } else {
          const iterator_layout<value_type> l(begin, end);
          bsend(&(*begin), l, destination, t);
        }
      }

      // --- non-blocking buffered send ---
    private:
      template<typename T>
      irequest ibsend(const T &data, int destination, tag_t t,
                      detail::basic_or_fixed_size_type) const {
        MPI_Request req;
        MPI_Ibsend(&data, 1, detail::datatype_traits<T>::get_datatype(), destination,
                   static_cast<int>(t), comm_, &req);
        return base_irequest{req};
      }

      template<typename T>
      void ibsend(const T &data, int destination, tag_t t, isend_irecv_state *isend_state,
                  detail::contiguous_const_stl_container) const {
        using value_type = typename T::value_type;
        const int count(data.size());
        MPI_Datatype datatype{detail::datatype_traits<value_type>::get_datatype()};
        MPI_Request req;
        MPI_Ibsend(data.size() > 0 ? &data[0] : nullptr, count, datatype, destination,
                   static_cast<int>(t), comm_, &req);
        MPI_Status s;
        MPI_Wait(&req, &s);
        isend_state->source = s.MPI_SOURCE;
        isend_state->tag = s.MPI_TAG;
        isend_state->datatype = datatype;
        isend_state->count = 0;
        MPI_Grequest_complete(isend_state->req);
      }

      template<typename T>
      void ibsend(const T &data, int destination, tag_t t, isend_irecv_state *isend_state,
                  detail::stl_container) const {
        using value_type = detail::remove_const_from_members_t<typename T::value_type>;
        detail::vector<value_type> serial_data(data.size(), std::begin(data));
        ibsend(serial_data, destination, t, isend_state,
               detail::contiguous_const_stl_container{});
      }

      template<typename T, typename C>
      irequest ibsend(const T &data, int destination, tag_t t, C) const {
        isend_irecv_state *send_state{new isend_irecv_state()};
        MPI_Request req;
        MPI_Grequest_start(isend_irecv_query, isend_irecv_free, isend_irecv_cancel, send_state,
                           &req);
        send_state->req = req;
        std::thread thread([this, &data, destination, t, send_state]() {
          ibsend(data, destination, t, send_state, C{});
        });
        thread.detach();
        return base_irequest{req};
      }

    public:
      /// Sends a message with a single value via a non-blocking buffered send operation.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section or an STL container
      /// that holds elements that comply with the mentioned requirements
      /// \param data value to send
      /// \param destination rank of the receiving process
      /// \param t tag associated to this message
      /// \return request representing the ongoing message transfer
      /// \note Sending STL containers is a convenience feature, which may have non-optimal
      /// performance characteristics. Use alternative overloads in performance critical code
      /// sections.
      /// \anchor communicator_ibsend
      template<typename T>
      irequest ibsend(const T &data, int destination, tag_t t = tag_t(0)) const {
        check_dest(destination);
        check_send_tag(t);
        check_container_size(data);
        return ibsend(data, destination, t,
                      typename detail::datatype_traits<T>::data_type_category{});
      }

      /// Sends a message with several values having a specific memory layout via a
      /// non-blocking buffered send operation.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim  section
      /// \param data pointer to the data to send
      /// \param l memory layout of the data to send
      /// \param destination rank of the receiving process
      /// \param t tag associated to this message
      /// \return request representing the ongoing message transfer
      template<typename T>
      irequest ibsend(const T *data, const layout<T> &l, int destination,
                      tag_t t = tag_t(0)) const {
        check_dest(destination);
        check_send_tag(t);
        MPI_Request req;
        MPI_Ibsend(data, 1, detail::datatype_traits<layout<T>>::get_datatype(l), destination,
                   static_cast<int>(t), comm_, &req);
        return base_irequest{req};
      }

      /// Sends a message with a several values given by a pair of iterators via a
      /// non-blocking buffered send operation.
      /// \tparam iterT iterator type, must fulfill the requirements of a
      /// <a
      /// href="https://en.cppreference.com/w/cpp/named_req/ForwardIterator">LegacyForwardIterator</a>,
      /// the iterator's value-type must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim  section
      /// \param begin iterator pointing to the first data value to send
      /// \param end iterator pointing one element beyond the last data value to send
      /// \param destination rank of the receiving process
      /// \param t tag associated to this message
      /// \note This is a convenience method, which may have non-optimal performance
      /// characteristics. Use alternative overloads in performance critical code sections.
      /// \return request representing the ongoing message transfer
      template<typename iterT>
      irequest ibsend(iterT begin, iterT end, int destination, tag_t t = tag_t(0)) const {
        using value_type = typename std::iterator_traits<iterT>::value_type;
        static_assert(std::is_reference_v<decltype(*begin)>,
                      "iterator de-referencing must yield a reference");
        if constexpr (detail::is_contiguous_iterator_v<iterT>) {
          const vector_layout<value_type> l(std::distance(begin, end));
          return ibsend(&(*begin), l, destination, t);
        } else {
          const iterator_layout<value_type> l(begin, end);
          return ibsend(&(*begin), l, destination, t);
        }
      }

      // --- persistent buffered send ---
      /// Creates a persistent communication request to send a message with a single
      /// value via a buffered send operation.
      /// \tparam T type of the data to send, must meet the  requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param data value to send
      /// \param destination rank of the receiving process
      /// \param t tag associated to this message
      /// \return persistent communication request
      /// \note Sending STL containers is not supported.
      template<typename T>
      prequest bsend_init(const T &data, int destination, tag_t t = tag_t(0)) const {
        check_dest(destination);
        check_send_tag(t);
        MPI_Request req;
        MPI_Bsend_init(&data, 1, detail::datatype_traits<T>::get_datatype(), destination,
                       static_cast<int>(t), comm_, &req);
        return base_prequest{req};
      }

      /// Creates a persistent communication request to send a message with a several
      /// values having a specific memory layout via a buffered send operation.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param data pointer to the data to send
      /// \param l memory layout of the data to send
      /// \param destination rank of the receiving process
      /// \param t tag associated to this message
      /// \return persistent communication request
      template<typename T>
      prequest bsend_init(const T *data, const layout<T> &l, int destination,
                          tag_t t = tag_t(0)) const {
        check_dest(destination);
        check_send_tag(t);
        MPI_Request req;
        MPI_Bsend_init(data, 1, detail::datatype_traits<layout<T>>::get_datatype(l),
                       destination, static_cast<int>(t), comm_, &req);
        return base_prequest{req};
      }

      /// Creates a persistent communication request to send a message with a several
      /// values given by a pair of iterators via a buffered send operation.
      /// \tparam iterT iterator type, must fulfill the requirements of a
      /// <a
      /// href="https://en.cppreference.com/w/cpp/named_req/ForwardIterator">LegacyForwardIterator</a>,
      /// the iterator's value-type must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param begin iterator pointing to the first data value to send
      /// \param end iterator pointing one element beyond the last data value to send
      /// \param destination rank of the receiving process
      /// \param t tag associated to this message
      /// \return persistent communication request
      /// \note This is a convenience method, which may have non-optimal performance
      /// characteristics. Use alternative overloads in performance critical code sections.
      template<typename iterT>
      prequest bsend_init(iterT begin, iterT end, int destination, tag_t t = tag_t(0)) const {
        using value_type = typename std::iterator_traits<iterT>::value_type;
        static_assert(std::is_reference_v<decltype(*begin)>,
                      "iterator de-referencing must yield a reference");
        if constexpr (detail::is_contiguous_iterator_v<iterT>) {
          const vector_layout<value_type> l(std::distance(begin, end));
          return bsend_init(&(*begin), l, destination, t);
        } else {
          const iterator_layout<value_type> l(begin, end);
          return bsend_init(&(*begin), l, destination, t);
        }
      }

      // === synchronous send ===
      // --- blocking synchronous send ---
    private:
      template<typename T>
      void ssend(const T &data, int destination, tag_t t,
                 detail::basic_or_fixed_size_type) const {
        MPI_Ssend(&data, 1, detail::datatype_traits<T>::get_datatype(), destination,
                  static_cast<int>(t), comm_);
      }

      template<typename T>
      void ssend(const T &data, int destination, tag_t t,
                 detail::contiguous_const_stl_container) const {
        using value_type = typename T::value_type;
        const vector_layout<value_type> l(data.size());
        ssend(data.size() > 0 ? &data[0] : nullptr, l, destination, t);
      }

      template<typename T>
      void ssend(const T &data, int destination, tag_t t, detail::stl_container) const {
        using value_type = detail::remove_const_from_members_t<typename T::value_type>;
        detail::vector<value_type> serial_data(data.size(), std::begin(data));
        const vector_layout<value_type> l(serial_data.size());
        ssend(serial_data.data(), l, destination, t);
      }

    public:
      /// Sends a message with a single value via a blocking synchronous send operation.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section or an STL container
      /// that holds elements that comply with the mentioned requirements
      /// \param data value to send
      /// \param destination rank of the receiving process
      /// \param t tag associated to this message
      /// \note Sending STL containers is a convenience feature, which may have non-optimal
      /// performance characteristics. Use alternative overloads in performance critical code
      /// sections.
      template<typename T>
      void ssend(const T &data, int destination, tag_t t = tag_t(0)) const {
        check_dest(destination);
        check_send_tag(t);
        ssend(data, destination, t, typename detail::datatype_traits<T>::data_type_category{});
      }

      /// Sends a message with a several values having a specific memory layout via a
      /// blocking synchronous send operation.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param data pointer to the data to send
      /// \param l memory layout of the data to send
      /// \param destination rank of the receiving process
      /// \param t tag associated to this message
      template<typename T>
      void ssend(const T *data, const layout<T> &l, int destination, tag_t t = tag_t(0)) const {
        check_dest(destination);
        check_send_tag(t);
        MPI_Ssend(data, 1, detail::datatype_traits<layout<T>>::get_datatype(l), destination,
                  static_cast<int>(t), comm_);
      }

      /// Sends a message with a several values given by a pair of iterators via a
      /// blocking synchronous send operation.
      /// \tparam iterT iterator type, must fulfill the requirements of a
      /// <a
      /// href="https://en.cppreference.com/w/cpp/named_req/ForwardIterator">LegacyForwardIterator</a>,
      /// the iterator's value-type must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param begin iterator pointing to the first data value to send
      /// \param end iterator pointing one element beyond the last data value to send
      /// \param destination rank of the receiving process
      /// \param t tag associated to this message
      /// \note This is a convenience method, which may have non-optimal performance
      /// characteristics. Use alternative overloads in performance critical code sections.
      template<typename iterT>
      void ssend(iterT begin, iterT end, int destination, tag_t t = tag_t(0)) const {
        using value_type = typename std::iterator_traits<iterT>::value_type;
        static_assert(std::is_reference_v<decltype(*begin)>,
                      "iterator de-referencing must yield a reference");
        if constexpr (detail::is_contiguous_iterator_v<iterT>) {
          const vector_layout<value_type> l(std::distance(begin, end));
          ssend(&(*begin), l, destination, t);
        } else {
          const iterator_layout<value_type> l(begin, end);
          ssend(&(*begin), l, destination, t);
        }
      }

      // --- non-blocking synchronous send ---
    private:
      template<typename T>
      irequest issend(const T &data, int destination, tag_t t,
                      detail::basic_or_fixed_size_type) const {
        MPI_Request req;
        MPI_Issend(&data, 1, detail::datatype_traits<T>::get_datatype(), destination,
                   static_cast<int>(t), comm_, &req);
        return base_irequest{req};
      }

      template<typename T>
      void issend(const T &data, int destination, tag_t t, isend_irecv_state *isend_state,
                  detail::contiguous_const_stl_container) const {
        using value_type = typename T::value_type;
        const int count(data.size());
        MPI_Datatype datatype{detail::datatype_traits<value_type>::get_datatype()};
        MPI_Request req;
        MPI_Issend(data.size() > 0 ? &data[0] : nullptr, count, datatype, destination,
                   static_cast<int>(t), comm_, &req);
        MPI_Status s;
        MPI_Wait(&req, &s);
        isend_state->source = s.MPI_SOURCE;
        isend_state->tag = s.MPI_TAG;
        isend_state->datatype = datatype;
        isend_state->count = 0;
        MPI_Grequest_complete(isend_state->req);
      }

      template<typename T>
      void issend(const T &data, int destination, tag_t t, isend_irecv_state *isend_state,
                  detail::stl_container) const {
        using value_type = detail::remove_const_from_members_t<typename T::value_type>;
        detail::vector<value_type> serial_data(data.size(), std::begin(data));
        issend(serial_data, destination, t, isend_state,
               detail::contiguous_const_stl_container{});
      }

      template<typename T, typename C>
      irequest issend(const T &data, int destination, tag_t t, C) const {
        isend_irecv_state *send_state{new isend_irecv_state()};
        MPI_Request req;
        MPI_Grequest_start(isend_irecv_query, isend_irecv_free, isend_irecv_cancel, send_state,
                           &req);
        send_state->req = req;
        std::thread thread([this, &data, destination, t, send_state]() {
          issend(data, destination, t, send_state, C{});
        });
        thread.detach();
        return base_irequest{req};
      }

    public:
      /// Sends a message with a single value via a non-blocking synchronous send
      /// operation.
      /// \tparam T type of the data to send, must meet the requirements as described
      /// in the \verbatim embed:rst:inline :doc:`data_types` \endverbatim section or an STL
      /// container that holds elements that comply with the mentioned requirements
      /// \param data value to send
      /// \param destination rank of the receiving process
      /// \param t tag associated to this message
      /// \return request representing the ongoing message transfer
      /// \note Sending STL containers  is a convenience feature, which may have non-optimal
      /// performance characteristics. Use alternative overloads in performance critical code
      /// sections.
      template<typename T>
      irequest issend(const T &data, int destination, tag_t t = tag_t(0)) const {
        check_dest(destination);
        check_send_tag(t);
        check_container_size(data);
        return issend(data, destination, t,
                      typename detail::datatype_traits<T>::data_type_category{});
      }

      /// Sends a message with several values having a specific memory layout via a
      /// non-blocking synchronous send operation.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param data pointer to the data to send
      /// \param l memory layout of the data to send
      /// \param destination rank of the receiving process
      /// \param t tag associated to this message
      /// \return request representing the ongoing message transfer
      template<typename T>
      irequest issend(const T *data, const layout<T> &l, int destination,
                      tag_t t = tag_t(0)) const {
        check_dest(destination);
        check_send_tag(t);
        MPI_Request req;
        MPI_Issend(data, 1, detail::datatype_traits<layout<T>>::get_datatype(l), destination,
                   static_cast<int>(t), comm_, &req);
        return base_irequest{req};
      }

      /// Sends a message with a several values given by a pair of iterators via a
      /// non-blocking synchronous send operation.
      /// \tparam iterT iterator type, must fulfill the requirements of a
      /// <a
      /// href="https://en.cppreference.com/w/cpp/named_req/ForwardIterator">LegacyForwardIterator</a>,
      /// the iterator's value-type must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param begin iterator pointing to the first data value to send
      /// \param end iterator pointing one element beyond the last data value to send
      /// \param destination rank of the receiving process
      /// \param t tag associated to this message
      /// \note This is a convenience method, which may have non-optimal performance
      /// characteristics. Use alternative overloads in performance critical code sections.
      /// \return request representing the ongoing message transfer
      template<typename iterT>
      irequest issend(iterT begin, iterT end, int destination, tag_t t = tag_t(0)) const {
        using value_type = typename std::iterator_traits<iterT>::value_type;
        static_assert(std::is_reference_v<decltype(*begin)>,
                      "iterator de-referencing must yield a reference");
        if constexpr (detail::is_contiguous_iterator_v<iterT>) {
          const vector_layout<value_type> l(std::distance(begin, end));
          return issend(&(*begin), l, destination, t);
        } else {
          const iterator_layout<value_type> l(begin, end);
          return issend(&(*begin), l, destination, t);
        }
      }

      // --- persistent synchronous send ---
      /// Creates a persistent communication request to send a message with a single
      /// value via a blocking synchronous send operation. \tparam T type of the data to send,
      /// must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param data value to send
      /// \param destination rank of the receiving process
      /// \param t tag associated to this message
      /// \return persistent communication request
      /// \note Sending STL containers is not supported.
      template<typename T>
      prequest ssend_init(const T &data, int destination, tag_t t = tag_t(0)) const {
        check_dest(destination);
        check_send_tag(t);
        MPI_Request req;
        MPI_Ssend_init(&data, 1, detail::datatype_traits<T>::get_datatype(), destination,
                       static_cast<int>(t), comm_, &req);
        return base_prequest{req};
      }

      /// Creates a persistent communication request to send a message with a several
      /// values having a specific memory layout via a blocking synchronous send operation.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param data pointer to the data to send
      /// \param l memory layout of the data to send
      /// \param destination rank of the receiving process
      /// \param t tag associated to this message
      /// \return persistent communication request
      template<typename T>
      prequest ssend_init(const T *data, const layout<T> &l, int destination,
                          tag_t t = tag_t(0)) const {
        check_dest(destination);
        check_send_tag(t);
        MPI_Request req;
        MPI_Ssend_init(data, 1, detail::datatype_traits<layout<T>>::get_datatype(l),
                       destination, static_cast<int>(t), comm_, &req);
        return base_prequest{req};
      }

      /// Creates a persistent communication request to send a message with a several
      /// values given by a pair of iterators via a blocking synchronous send operation.
      /// \tparam iterT iterator type, must fulfill the requirements of a
      /// <a
      /// href="https://en.cppreference.com/w/cpp/named_req/ForwardIterator">LegacyForwardIterator</a>,
      /// the iterator's value-type must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param begin iterator pointing to the first data value to send
      /// \param end iterator pointing one element beyond the last data value to send
      /// \param destination rank of the receiving process
      /// \param t tag associated to this message
      /// \return persistent communication request
      /// \note This is a convenience method, which may have non-optimal performance
      /// characteristics. Use alternative overloads in performance critical code sections.
      template<typename iterT>
      prequest ssend_init(iterT begin, iterT end, int destination, tag_t t = tag_t(0)) const {
        using value_type = typename std::iterator_traits<iterT>::value_type;
        static_assert(std::is_reference_v<decltype(*begin)>,
                      "iterator de-referencing must yield a reference");
        if constexpr (detail::is_contiguous_iterator_v<iterT>) {
          const vector_layout<value_type> l(std::distance(begin, end));
          return ssend_init(&(*begin), l, destination, t);
        } else {
          const iterator_layout<value_type> l(begin, end);
          return ssend_init(&(*begin), l, destination, t);
        }
      }

      // === ready send ===
      // --- blocking ready send ---
    private:
      template<typename T>
      void rsend(const T &data, int destination, tag_t t,
                 detail::basic_or_fixed_size_type) const {
        MPI_Rsend(&data, 1, detail::datatype_traits<T>::get_datatype(), destination,
                  static_cast<int>(t), comm_);
      }

      template<typename T>
      void rsend(const T &data, int destination, tag_t t,
                 detail::contiguous_const_stl_container) const {
        using value_type = typename T::value_type;
        const vector_layout<value_type> l(data.size());
        rsend(data.size() > 0 ? &data[0] : nullptr, l, destination, t);
      }

      template<typename T>
      void rsend(const T &data, int destination, tag_t t, detail::stl_container) const {
        using value_type = detail::remove_const_from_members_t<typename T::value_type>;
        detail::vector<value_type> serial_data(data.size(), std::begin(data));
        const vector_layout<value_type> l(serial_data.size());
        rsend(serial_data.data(), l, destination, t);
      }

    public:
      /// Sends a message with a single value via a blocking ready send operation.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section or an STL container
      /// that holds elements that comply with the mentioned requirements
      /// \param data value to send
      /// \param destination rank of the receiving process
      /// \param t tag associated to this message
      /// \note Sending STL containers is a convenience feature, which may have non-optimal
      /// performance characteristics. Use alternative overloads in performance critical code
      /// sections.
      template<typename T>
      void rsend(const T &data, int destination, tag_t t = tag_t(0)) const {
        check_dest(destination);
        check_send_tag(t);
        check_container_size(data);
        rsend(data, destination, t, typename detail::datatype_traits<T>::data_type_category{});
      }

      /// Sends a message with a several values having a specific memory layout via a
      /// blocking ready send operation.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param data pointer to the data to send
      /// \param l memory layout of the data to send
      /// \param destination rank of the receiving process
      /// \param t tag associated to this message
      template<typename T>
      void rsend(const T *data, const layout<T> &l, int destination, tag_t t = tag_t(0)) const {
        check_dest(destination);
        check_send_tag(t);
        MPI_Rsend(data, 1, detail::datatype_traits<layout<T>>::get_datatype(l), destination,
                  static_cast<int>(t), comm_);
      }

      /// Sends a message with a several values given by a pair of iterators via a
      /// blocking ready send operation.
      /// \tparam iterT iterator type, must fulfill the requirements of a
      /// <a
      /// href="https://en.cppreference.com/w/cpp/named_req/ForwardIterator">LegacyForwardIterator</a>,
      /// the iterator's value-type must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param begin iterator pointing to the first data value to send
      /// \param end iterator pointing one element beyond the last data value to send
      /// \param destination rank of the receiving process
      /// \param t tag associated to this message
      /// \note This is a convenience method, which may have non-optimal performance
      /// characteristics. Use alternative overloads in performance critical code sections.
      template<typename iterT>
      void rsend(iterT begin, iterT end, int destination, tag_t t = tag_t(0)) const {
        using value_type = typename std::iterator_traits<iterT>::value_type;
        static_assert(std::is_reference_v<decltype(*begin)>,
                      "iterator de-referencing must yield a reference");
        if constexpr (detail::is_contiguous_iterator_v<iterT>) {
          const vector_layout<value_type> l(std::distance(begin, end));
          rsend(&(*begin), l, destination, t);
        } else {
          const iterator_layout<value_type> l(begin, end);
          rsend(&(*begin), l, destination, t);
        }
      }

      // --- non-blocking ready send ---
    private:
      template<typename T>
      irequest irsend(const T &data, int destination, tag_t t,
                      detail::basic_or_fixed_size_type) const {
        MPI_Request req;
        MPI_Irsend(&data, 1, detail::datatype_traits<T>::get_datatype(), destination,
                   static_cast<int>(t), comm_, &req);
        return base_irequest{req};
      }

      template<typename T>
      void irsend(const T &data, int destination, tag_t t, isend_irecv_state *isend_state,
                  detail::contiguous_const_stl_container) const {
        using value_type = typename T::value_type;
        const int count(data.size());
        MPI_Datatype datatype{detail::datatype_traits<value_type>::get_datatype()};
        MPI_Request req;
        MPI_Irsend(data.size() > 0 ? &data[0] : nullptr, count, datatype, destination,
                   static_cast<int>(t), comm_, &req);
        MPI_Status s;
        MPI_Wait(&req, &s);
        isend_state->source = s.MPI_SOURCE;
        isend_state->tag = s.MPI_TAG;
        isend_state->datatype = datatype;
        isend_state->count = 0;
        MPI_Grequest_complete(isend_state->req);
      }

      template<typename T>
      void irsend(const T &data, int destination, tag_t t, isend_irecv_state *isend_state,
                  detail::stl_container) const {
        using value_type = detail::remove_const_from_members_t<typename T::value_type>;
        detail::vector<value_type> serial_data(data.size(), std::begin(data));
        irsend(serial_data, destination, t, isend_state,
               detail::contiguous_const_stl_container{});
      }

      template<typename T, typename C>
      irequest irsend(const T &data, int destination, tag_t t, C) const {
        isend_irecv_state *send_state{new isend_irecv_state()};
        MPI_Request req;
        MPI_Grequest_start(isend_irecv_query, isend_irecv_free, isend_irecv_cancel, send_state,
                           &req);
        send_state->req = req;
        std::thread thread([this, &data, destination, t, send_state]() {
          irsend(data, destination, t, send_state, C{});
        });
        thread.detach();
        return base_irequest{req};
      }

    public:
      /// Sends a message with a single value via a non-blocking ready send operation.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section or an STL container
      /// that holds elements that comply with the mentioned requirements
      /// \param data value to send
      /// \param destination rank of the receiving process
      /// \param t tag associated to this message
      /// \return request representing the ongoing message transfer
      /// \note Sending STL containers is a convenience feature, which may have non-optimal
      /// performance characteristics. Use alternative overloads in performance critical code
      /// sections.
      template<typename T>
      irequest irsend(const T &data, int destination, tag_t t = tag_t(0)) const {
        check_dest(destination);
        check_send_tag(t);
        check_container_size(data);
        return irsend(data, destination, t,
                      typename detail::datatype_traits<T>::data_type_category{});
      }

      /// Sends a message with several values having a specific memory layout via a
      /// non-blocking ready send operation.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param data pointer to the data to send
      /// \param l memory layout of the data to send
      /// \param destination rank of the receiving process
      /// \param t tag associated to this message
      /// \return request representing the ongoing message transfer
      template<typename T>
      irequest irsend(const T *data, const layout<T> &l, int destination,
                      tag_t t = tag_t(0)) const {
        check_dest(destination);
        check_send_tag(t);
        MPI_Request req;
        MPI_Irsend(data, 1, detail::datatype_traits<layout<T>>::get_datatype(l), destination,
                   static_cast<int>(t), comm_, &req);
        return base_irequest{req};
      }

      /// Sends a message with a several values given by a pair of iterators via a
      /// non-blocking ready send operation.
      /// \tparam iterT iterator type, must fulfill the requirements of a
      /// <a
      /// href="https://en.cppreference.com/w/cpp/named_req/ForwardIterator">LegacyForwardIterator</a>,
      /// the iterator's value-type must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param begin iterator pointing to the first data value to send
      /// \param end iterator pointing one element beyond the last data value to send
      /// \param destination rank of the receiving process
      /// \param t tag associated to this message
      /// \return request representing the ongoing message transfer
      /// \note This is a convenience method, which may have non-optimal performance
      /// characteristics. Use alternative overloads in performance critical code sections.
      template<typename iterT>
      irequest irsend(iterT begin, iterT end, int destination, tag_t t = tag_t(0)) const {
        using value_type = typename std::iterator_traits<iterT>::value_type;
        static_assert(std::is_reference_v<decltype(*begin)>,
                      "iterator de-referencing must yield a reference");
        if constexpr (detail::is_contiguous_iterator_v<iterT>) {
          const vector_layout<value_type> l(std::distance(begin, end));
          return irsend(&(*begin), l, destination, t);
        } else {
          const iterator_layout<value_type> l(begin, end);
          return irsend(&(*begin), l, destination, t);
        }
      }

      // --- persistent ready send ---
      /// Creates a persistent communication request to send a message with a single
      /// value via a blocking ready send operation.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param data value to send
      /// \param destination rank of the receiving process
      /// \param t tag associated to this message
      /// \return persistent communication request
      /// \note Sending STL containers is not supported.
      template<typename T>
      prequest rsend_init(const T &data, int destination, tag_t t = tag_t(0)) const {
        check_dest(destination);
        check_send_tag(t);
        MPI_Request req;
        MPI_Rsend_init(&data, 1, detail::datatype_traits<T>::get_datatype(), destination,
                       static_cast<int>(t), comm_, &req);
        return base_prequest{req};
      }

      /// Creates a persistent communication request to send a message with a several
      /// values having a specific memory layout via a blocking ready send operation.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param data pointer to the data to send
      /// \param l memory layout of the data to send
      /// \param destination rank of the receiving process
      /// \param t tag associated to this message
      /// \return persistent communication request
      template<typename T>
      prequest rsend_init(const T *data, const layout<T> &l, int destination,
                          tag_t t = tag_t(0)) const {
        check_dest(destination);
        check_send_tag(t);
        MPI_Request req;
        MPI_Rsend_init(data, 1, detail::datatype_traits<layout<T>>::get_datatype(l),
                       destination, static_cast<int>(t), comm_, &req);
        return base_prequest{req};
      }

      /// Creates a persistent communication request to send a message with a several
      /// values given by a pair of iterators via a blocking ready send operation.
      /// \tparam iterT iterator type, must fulfill the requirements of a
      /// <a
      /// href="https://en.cppreference.com/w/cpp/named_req/ForwardIterator">LegacyForwardIterator</a>,
      /// the iterator's value-type must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param begin iterator pointing to the first data value to send
      /// \param end iterator pointing one element beyond the last data value to send
      /// \param destination rank of the receiving process
      /// \param t tag associated to this message
      /// \return persistent communication request
      /// \note This is a convenience method, which may have non-optimal performance
      /// characteristics. Use alternative overloads in performance critical code sections.
      template<typename iterT>
      prequest rsend_init(iterT begin, iterT end, int destination, tag_t t = tag_t(0)) const {
        using value_type = typename std::iterator_traits<iterT>::value_type;
        static_assert(std::is_reference_v<decltype(*begin)>,
                      "iterator de-referencing must yield a reference");
        if constexpr (detail::is_contiguous_iterator_v<iterT>) {
          const vector_layout<value_type> l(std::distance(begin, end));
          return rsend_init(&(*begin), l, destination, t);
        } else {
          const iterator_layout<value_type> l(begin, end);
          return rsend_init(&(*begin), l, destination, t);
        }
      }

      // === receive ===
      // --- blocking receive ---
    private:
      template<typename T>
      status_t recv(T &data, int source, tag_t t, detail::basic_or_fixed_size_type) const {
        status_t s;
        MPI_Recv(&data, 1, detail::datatype_traits<T>::get_datatype(), source,
                 static_cast<int>(t), comm_, static_cast<MPI_Status *>(&s));
        return s;
      }

      template<typename T>
      status_t recv(T &data, int source, tag_t t, detail::contiguous_stl_container) const {
        using value_type = typename T::value_type;
        status_t s;
        auto *ps{static_cast<MPI_Status *>(&s)};
        MPI_Message message;
        MPI_Mprobe(source, static_cast<int>(t), comm_, &message, ps);
        int count{0};
        MPI_Get_count(ps, detail::datatype_traits<value_type>::get_datatype(), &count);
        check_count(count);
        if constexpr (detail::has_resize_v<T>)
          data.resize(count);
        MPI_Mrecv(data.size() > 0 ? &data[0] : nullptr, count,
                  detail::datatype_traits<value_type>::get_datatype(), &message, ps);
        return s;
      }

      template<typename T>
      status_t recv(T &data, int source, tag_t t, detail::stl_container) const {
        using value_type = detail::remove_const_from_members_t<typename T::value_type>;
        status_t s;
        auto *ps{static_cast<MPI_Status *>(&s)};
        MPI_Message message;
        MPI_Mprobe(source, static_cast<int>(t), comm_, &message, ps);
        int count{0};
        MPI_Get_count(ps, detail::datatype_traits<value_type>::get_datatype(), &count);
        check_count(count);
        detail::vector<value_type> serial_data(count, detail::uninitialized{});
        MPI_Mrecv(serial_data.data(), count,
                  detail::datatype_traits<value_type>::get_datatype(), &message, ps);
        T new_data(serial_data.begin(), serial_data.end());
        data.swap(new_data);
        return s;
      }

    public:
      /// Receives a message with a single value.
      /// \tparam T type of the data to receive, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section or an STL container
      /// that holds elements that comply with the mentioned requirements
      /// \param data value to receive
      /// \param source rank of the sending process
      /// \param t tag associated to this message
      /// \return status of the receive operation
      /// \note Receiving STL containers is a convenience feature, which may have non-optimal
      /// performance characteristics. Use alternative overloads in performance critical code
      /// sections.
      /// \anchor communicator_recv
      template<typename T>
      status_t recv(T &data, int source, tag_t t = tag_t(0)) const {
        check_source(source);
        check_recv_tag(t);
        return recv(data, source, t, typename detail::datatype_traits<T>::data_type_category{});
      }

      /// Receives a message with a several values having a specific memory layout.
      /// \tparam T type of the data to receive, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param data pointer to the data to receive
      /// \param l memory layout of the data to receive
      /// \param source rank of the sending process
      /// \param t tag associated to this message
      /// \return status of the receive operation
      template<typename T>
      status_t recv(T *data, const layout<T> &l, int source, tag_t t = tag_t(0)) const {
        check_source(source);
        check_recv_tag(t);
        status_t s;
        MPI_Recv(data, 1, detail::datatype_traits<layout<T>>::get_datatype(l), source,
                 static_cast<int>(t), comm_, static_cast<MPI_Status *>(&s));
        return s;
      }

      /// Receives a message with a several values given by a pair of iterators.
      /// \tparam iterT iterator type, must fulfill the requirements of a
      /// <a
      /// href="https://en.cppreference.com/w/cpp/named_req/ForwardIterator">LegacyForwardIterator</a>,
      /// the iterator's value-type must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param begin iterator pointing to the first data value to receive
      /// \param end iterator pointing one element beyond the last data value to receive
      /// \param source rank of the sending process
      /// \param t tag associated to this message
      /// \return status of the receive operation
      /// \note This is a convenience method, which may have non-optimal performance
      /// characteristics. Use alternative overloads in performance critical code sections.
      template<typename iterT>
      status_t recv(iterT begin, iterT end, int source, tag_t t = tag_t(0)) const {
        using value_type = typename std::iterator_traits<iterT>::value_type;
        static_assert(std::is_reference_v<decltype(*begin)>,
                      "iterator de-referencing must yield a reference");
        if constexpr (detail::is_contiguous_iterator_v<iterT>) {
          const vector_layout<value_type> l(std::distance(begin, end));
          return recv(&(*begin), l, source, t);
        } else {
          const iterator_layout<value_type> l(begin, end);
          return recv(&(*begin), l, source, t);
        }
      }

      // --- non-blocking receive ---
    private:
      template<typename T>
      irequest irecv(T &data, int source, tag_t t, detail::basic_or_fixed_size_type) const {
        MPI_Request req;
        MPI_Irecv(&data, 1, detail::datatype_traits<T>::get_datatype(), source,
                  static_cast<int>(t), comm_, &req);
        return base_irequest{req};
      }

      template<typename T>
      void irecv(T &data, int source, tag_t t, isend_irecv_state *irecv_state,
                 detail::stl_container) const {
        using value_type = detail::remove_const_from_members_t<typename T::value_type>;
        const status_t s{recv(data, source, t)};
        irecv_state->source = s.source();
        irecv_state->tag = static_cast<int>(s.tag());
        irecv_state->datatype = detail::datatype_traits<value_type>::get_datatype();
        irecv_state->count = s.get_count<value_type>();
        MPI_Grequest_complete(irecv_state->req);
      }

      template<typename T, typename C>
      irequest irecv(T &data, int source, tag_t t, C) const {
        isend_irecv_state *recv_state{new isend_irecv_state()};
        MPI_Request req;
        MPI_Grequest_start(isend_irecv_query, isend_irecv_free, isend_irecv_cancel, recv_state,
                           &req);
        recv_state->req = req;
        std::thread thread([this, &data, source, t, recv_state]() {
          irecv(data, source, t, recv_state, C{});
        });
        thread.detach();
        return base_irequest{req};
      }

    public:
      /// Receives a message with a single value via a non-blocking receive operation.
      /// \tparam T type of the data to receive, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section or an STL container
      /// that holds elements that comply with the mentioned requirements
      /// \param data value to receive
      /// \param source rank of the sending process
      /// \param t tag associated to this message
      /// \return request representing the ongoing receive operation
      /// \note Receiving STL containers is a convenience feature, which may have non-optimal
      /// performance characteristics. Use alternative overloads in performance critical code
      /// sections.
      template<typename T>
      irequest irecv(T &data, int source, tag_t t = tag_t(0)) const {
        check_source(source);
        check_recv_tag(t);
        return irecv(data, source, t,
                     typename detail::datatype_traits<T>::data_type_category{});
      }

      /// Receives a message with several values having a specific memory layout via a
      /// non-blocking receive operation.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param data pointer to the data to receive
      /// \param l memory layout of the data to receive
      /// \param source rank of the sending process
      /// \param t tag associated to this message
      /// \return request representing the ongoing receive operation
      template<typename T>
      irequest irecv(T *data, const layout<T> &l, int source, tag_t t = tag_t(0)) const {
        check_source(source);
        check_recv_tag(t);
        MPI_Request req;
        MPI_Irecv(data, 1, detail::datatype_traits<layout<T>>::get_datatype(l), source,
                  static_cast<int>(t), comm_, &req);
        return base_irequest{req};
      }

      /// Receives a message with a several values given by a pair of iterators via a
      /// non-blocking receive operation.
      /// \tparam iterT iterator type, must fulfill the requirements of a
      /// <a
      /// href="https://en.cppreference.com/w/cpp/named_req/ForwardIterator">LegacyForwardIterator</a>,
      /// the iterator's value-type must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param begin iterator pointing to the first data value to receive
      /// \param end iterator pointing one element beyond the last data value to receive
      /// \param source rank of the sending process
      /// \param t tag associated to this message
      /// \return request representing the ongoing message transfer
      /// \note This is a convenience method, which may have non-optimal performance
      /// characteristics. Use alternative overloads in performance critical code sections.
      template<typename iterT>
      irequest irecv(iterT begin, iterT end, int source, tag_t t = tag_t(0)) const {
        using value_type = typename std::iterator_traits<iterT>::value_type;
        static_assert(std::is_lvalue_reference_v<decltype(*begin)>,
                      "iterator de-referencing must yield a reference");
        if constexpr (detail::is_contiguous_iterator_v<iterT>) {
          const vector_layout<value_type> l(std::distance(begin, end));
          return irecv(&(*begin), l, source, t);
        } else {
          const iterator_layout<value_type> l(begin, end);
          return irecv(&(*begin), l, source, t);
        }
      }

      // --- persistent receive ---
      /// Creates a persistent communication request to receive a message with a single
      /// value via a blocking receive operation.
      /// \tparam T type of the data to receive, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param data value to receive
      /// \param source rank of the sending process
      /// \param t tag associated to this message
      /// \return persistent communication request
      /// \note Receiving STL containers is not supported.
      template<typename T>
      prequest recv_init(T &data, int source, tag_t t = tag_t(0)) const {
        check_source(source);
        check_recv_tag(t);
        MPI_Request req;
        MPI_Recv_init(&data, 1, detail::datatype_traits<T>::get_datatype(), source,
                      static_cast<int>(t), comm_, &req);
        return base_prequest{req};
      }

      /// Creates a persistent communication request to receive a message with a several
      /// values having a specific memory layout via a blocking standard send operation.
      /// \tparam T type of the data to receive, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param data pointer to the data to receive
      /// \param l memory layout of the data to receive
      /// \param source rank of the sending process
      /// \param t tag associated to this message
      /// \return persistent communication request
      template<typename T>
      prequest recv_init(T *data, const layout<T> &l, int source, tag_t t = tag_t(0)) const {
        check_source(source);
        check_recv_tag(t);
        MPI_Request req;
        MPI_Recv_init(data, 1, detail::datatype_traits<layout<T>>::get_datatype(l), source,
                      static_cast<int>(t), comm_, &req);
        return base_prequest{req};
      }

      /// Creates a persistent communication request to receive a message with a several
      /// values given by a pair of iterators via a blocking receive operation.
      /// \tparam iterT iterator type, must fulfill the requirements of a
      /// <a
      /// href="https://en.cppreference.com/w/cpp/named_req/ForwardIterator">LegacyForwardIterator</a>,
      /// the iterator's value-type must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param begin iterator pointing to the first data value to receive
      /// \param end iterator pointing one element beyond the last data value to receive
      /// \param source rank of the sending ing process
      /// \param t tag associated to this message
      /// \return persistent communication request
      /// \note This is a convenience method, which may have non-optimal performance
      /// characteristics. Use alternative overloads in performance critical code sections.
      template<typename iterT>
      prequest recv_init(iterT begin, iterT end, int source, tag_t t = tag_t(0)) const {
        using value_type = typename std::iterator_traits<iterT>::value_type;
        if constexpr (detail::is_contiguous_iterator_v<iterT>) {
          const vector_layout<value_type> l(std::distance(begin, end));
          return recv_init(&(*begin), l, source, t);
        } else {
          const iterator_layout<value_type> l(begin, end);
          return recv_init(&(*begin), l, source, t);
        }
      }

      // === probe ===
      // --- blocking probe ---
      /// Blocking test for an incoming message.
      /// \param source rank of the sending process
      /// \param t tag associated to this message
      /// \return status of the pending message
      [[nodiscard]] status_t probe(int source, tag_t t = tag_t(0)) const {
        check_source(source);
        check_recv_tag(t);
        status_t s;
        MPI_Probe(source, static_cast<int>(t), comm_, static_cast<MPI_Status *>(&s));
        return s;
      }

      // --- non-blocking probe ---
      /// Non-blocking test for an incoming message.
      /// \param source rank of the sending process
      /// \param t tag associated to this message
      /// \return status of the pending message if there is any pending message
      [[nodiscard]] std::optional<status_t> iprobe(int source, tag_t t = tag_t(0)) const {
        check_source(source);
        check_recv_tag(t);
        int result;
        status_t s;
        MPI_Iprobe(source, static_cast<int>(t), comm_, &result, static_cast<MPI_Status *>(&s));
        if (result == 0)
          return {};
        else
          return s;
      }

      // === matching probe ===
      // --- blocking matching probe ---
      /// Blocking matched test for an incoming message.
      /// \param source rank of the sending process
      /// \param t tag associated to this message
      /// \return message handle and status of the pending message
      [[nodiscard]] mprobe_status mprobe(int source, tag_t t = tag_t(0)) const {
        check_source(source);
        check_recv_tag(t);
        status_t s;
        message_t m;
        MPI_Mprobe(source, static_cast<int>(t), comm_, &m, static_cast<MPI_Status *>(&s));
        return {m, s};
      }

      // --- non-blocking matching probe ---
      /// Blocking matched test for an incoming message.
      /// \param source rank of the sending process
      /// \param t tag associated to this message
      /// \return message handle and status of the pending message if there is a pending message
      /// by the given source and with the given tag
      [[nodiscard]] std::optional<mprobe_status> improbe(int source, tag_t t = tag_t(0)) const {
        check_source(source);
        check_recv_tag(t);
        int result;
        status_t s;
        message_t m;
        MPI_Improbe(source, static_cast<int>(t), comm_, &result, &m,
                    static_cast<MPI_Status *>(&s));
        if (result == 0)
          return {};
        else
          return mprobe_status{m, s};
      }

      // === matching receive ===
      // --- blocking matching receive ---
    private:
      template<typename T>
      status_t mrecv(T &data, message_t &m, detail::basic_or_fixed_size_type) const {
        status_t s;
        MPI_Mrecv(&data, 1, detail::datatype_traits<T>::get_datatype(), &m,
                  static_cast<MPI_Status *>(&s));
        return s;
      }

    public:
      /// Receives a message with a single value by a message handle.
      /// \tparam T type of the data to receive, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section or an STL container
      /// that holds elements that comply with the mentioned requirements
      /// \param data value to receive
      /// \param m message handle of message to receive
      /// \return status of the receive operation
      /// \note Receiving STL containers is not supported.
      template<typename T>
      status_t mrecv(T &data, message_t &m) const {
        return mrecv(data, m, typename detail::datatype_traits<T>::data_type_category{});
      }

      /// Receives a message with a several values having a specific memory layout by a
      /// message handle.
      /// \tparam T type of the data to receive, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param data pointer to the data to receive
      /// \param l memory layout of the data to receive
      /// \param m message handle of message to receive
      /// \return status of the receive operation
      template<typename T>
      status_t mrecv(T *data, const layout<T> &l, message_t &m) const {
        status_t s;
        MPI_Mrecv(data, 1, detail::datatype_traits<layout<T>>::get_datatype(l), &m,
                  static_cast<MPI_Status *>(&s));
        return s;
      }

      /// Receives a message with a several values given by a pair of iterators by a
      /// message handle.
      /// \tparam iterT iterator type, must fulfill the requirements of a
      /// <a
      /// href="https://en.cppreference.com/w/cpp/named_req/ForwardIterator">LegacyForwardIterator</a>,
      /// the iterator's value-type must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param begin iterator pointing to the first data value to receive
      /// \param end iterator pointing one element beyond the last data value to receive
      /// \param m message handle of message to receive
      /// \return status of the receive operation
      /// \note This is a convenience method, which may have non-optimal performance
      /// characteristics. Use alternative overloads in performance critical code sections.
      template<typename iterT>
      status_t mrecv(iterT begin, iterT end, message_t &m) const {
        using value_type = typename std::iterator_traits<iterT>::value_type;
        static_assert(std::is_reference_v<decltype(*begin)>,
                      "iterator de-referencing must yield a reference");
        if constexpr (detail::is_contiguous_iterator_v<iterT>) {
          const vector_layout<value_type> l(std::distance(begin, end));
          return mrecv(&(*begin), l, m);
        } else {
          const iterator_layout<value_type> l(begin, end);
          return mrecv(&(*begin), l, m);
        }
      }

      // --- non-blocking matching receive ---
    private:
      template<typename T>
      irequest imrecv(T &data, message_t &m, detail::basic_or_fixed_size_type) const {
        MPI_Request req;
        MPI_Imrecv(&data, 1, detail::datatype_traits<T>::get_datatype(), &m, &req);
        return base_irequest{req};
      }

    public:
      /// Receives a message with a single value via a non-blocking receive operation by
      /// a message handle.
      /// \tparam T type of the data to receive, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param data pointer to the data to receive
      /// \param m message handle of message to receive
      /// \return request representing the ongoing receive operation
      /// \note Receiving STL containers is not supported.
      template<typename T>
      irequest imrecv(T &data, message_t &m) const {
        return imrecv(data, m, typename detail::datatype_traits<T>::data_type_category{});
      }

      /// Receives a message with several values having a specific memory layout via a
      /// non-blocking receive operation by a message handle.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param data pointer to the data to receive
      /// \param l memory layout of the data to receive
      /// \param m message handle of message to receive
      /// \return request representing the ongoing receive operation
      template<typename T>
      irequest imrecv(T *data, const layout<T> &l, message_t &m) const {
        MPI_Request req;
        MPI_Imrecv(data, 1, detail::datatype_traits<layout<T>>::get_datatype(l), &m, &req);
        return base_irequest{req};
      }

      /// Receives a message with a several values given by a pair of iterators via a
      /// non-blocking receive operation.
      /// \tparam iterT iterator type, must fulfill the requirements of a
      /// <a
      /// href="https://en.cppreference.com/w/cpp/named_req/ForwardIterator">LegacyForwardIterator</a>,
      /// the iterator's value-type must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param begin iterator pointing to the first data value to receive
      /// \param end iterator pointing one element beyond the last data value to receive
      /// \param m message handle of message to receive
      /// \return request representing the ongoing message transfer
      /// \note This is a convenience method, which may have non-optimal performance
      /// characteristics. Use alternative overloads in performance critical code sections.
      template<typename iterT>
      irequest imrecv(iterT begin, iterT end, message_t &m) const {
        using value_type = typename std::iterator_traits<iterT>::value_type;
        static_assert(std::is_lvalue_reference_v<decltype(*begin)>,
                      "iterator de-referencing must yield a reference");
        if constexpr (detail::is_contiguous_iterator_v<iterT>) {
          const vector_layout<value_type> l(std::distance(begin, end));
          return imrecv(&(*begin), l, m);
        } else {
          const iterator_layout<value_type> l(begin, end);
          return imrecv(&(*begin), l, m);
        }
      }

      // === send and receive ===
      // --- send and receive ---
      /// Sends a message and receives a message in a single operation.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param send_data data to send
      /// \param destination rank of the receiving process
      /// \param send_tag tag associated to the data to send
      /// \param recv_data data to receive
      /// \param source rank of the sending process
      /// \param recv_tag tag associated to the data to receive
      /// \return status of the receive operation
      template<typename T>
      status_t sendrecv(const T &send_data, int destination, tag_t send_tag, T &recv_data,
                        int source, tag_t recv_tag) const {
        check_dest(destination);
        check_source(source);
        check_send_tag(send_tag);
        check_recv_tag(recv_tag);
        status_t s;
        MPI_Sendrecv(&send_data, 1, detail::datatype_traits<T>::get_datatype(), destination,
                     static_cast<int>(send_tag), &recv_data, 1,
                     detail::datatype_traits<T>::get_datatype(), source,
                     static_cast<int>(recv_tag), comm_, static_cast<MPI_Status *>(&s));
        return s;
      }

      /// Sends a message and receives a message in a single operation.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param send_data data to send
      /// \param sendl memory layout of the data to send
      /// \param destination rank of the receiving process
      /// \param send_tag tag associated to the data to send
      /// \param recv_data data to receive
      /// \param recvl memory layout of the data to receive
      /// \param source rank of the sending process
      /// \param recv_tag tag associated to the data to receive
      /// \return status of the receive operation
      template<typename T>
      status_t sendrecv(const T *send_data, const layout<T> &sendl, int destination,
                        tag_t send_tag, T *recv_data, const layout<T> &recvl, int source,
                        tag_t recv_tag) const {
        check_dest(destination);
        check_source(source);
        check_send_tag(send_tag);
        check_recv_tag(recv_tag);
        status_t s;
        MPI_Sendrecv(send_data, 1, detail::datatype_traits<layout<T>>::get_datatype(sendl),
                     destination, static_cast<int>(send_tag), recv_data, 1,
                     detail::datatype_traits<layout<T>>::get_datatype(recvl), source,
                     static_cast<int>(recv_tag), comm_, static_cast<MPI_Status *>(&s));
        return s;
      }

      /// Sends a message and receives a message in a single operation.
      /// \tparam iterT1 iterator type, must fulfill the requirements of a
      /// <a
      /// href="https://en.cppreference.com/w/cpp/named_req/ForwardIterator">LegacyForwardIterator</a>,
      /// the iterator's value-type must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \tparam iterT2 iterator type, must fulfill the requirements of a
      /// <a
      /// href="https://en.cppreference.com/w/cpp/named_req/ForwardIterator">LegacyForwardIterator</a>,
      /// the iterator's value-type must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param begin_1 iterator pointing to the first data value to send
      /// \param end_1 iterator pointing one element beyond the last data value to send
      /// \param destination rank of the receiving process
      /// \param send_tag tag associated to the data to send
      /// \param begin_2 iterator pointing to the first data value to receive
      /// \param end_2 iterator pointing one element beyond the last data value to receive
      /// \param source rank of the sending process
      /// \param recv_tag tag associated to the data to receive
      /// \return status of the receive operation
      template<typename iterT1, typename iterT2>
      status_t sendrecv(iterT1 begin_1, iterT1 end_1, int destination, tag_t send_tag,
                        iterT2 begin_2, iterT2 end_2, int source, tag_t recv_tag) const {
        using value_type_1 = typename std::iterator_traits<iterT1>::value_type;
        using value_type_2 = typename std::iterator_traits<iterT2>::value_type;
        if constexpr (detail::is_contiguous_iterator_v<iterT1> and
                      detail::is_contiguous_iterator_v<iterT2>) {
          const vector_layout<value_type_1> l_1(std::distance(begin_1, end_1));
          const vector_layout<value_type_2> l_2(std::distance(begin_2, end_2));
          return sendrecv(&(*begin_1), l_1, destination, send_tag, &(*begin_2), l_2, source,
                          recv_tag);
        } else if constexpr (detail::is_contiguous_iterator_v<iterT1>) {
          const vector_layout<value_type_1> l_1(std::distance(begin_1, end_1));
          const iterator_layout<value_type_2> l_2(begin_2, end_2);
          return sendrecv(&(*begin_1), l_1, destination, send_tag, &(*begin_2), l_2, source,
                          recv_tag);
        } else if constexpr (detail::is_contiguous_iterator_v<iterT2>) {
          const iterator_layout<value_type_2> l_1(begin_1, end_1);
          const vector_layout<value_type_2> l_2(std::distance(begin_2, end_2));
          return sendrecv(&(*begin_1), l_1, destination, send_tag, &(*begin_2), l_2, source,
                          recv_tag);
        } else {
          const iterator_layout<value_type_1> l_1(begin_1, end_1);
          const iterator_layout<value_type_2> l_2(begin_2, end_2);
          return sendrecv(&(*begin_1), l_1, destination, send_tag, &(*begin_2), l_2, source,
                          recv_tag);
        }
      }

      // --- send, receive and replace ---
      /// Sends a message and receives a message in a single operation.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param data data to send, will hold the received data
      /// \param destination rank of the receiving process
      /// \param send_tag tag associated to the data to send
      /// \param source rank of the sending process
      /// \param recv_tag tag associated to the data to receive
      /// \return status of the receive operation
      template<typename T>
      status_t sendrecv_replace(T &data, int destination, tag_t send_tag, int source,
                                tag_t recv_tag) const {
        check_dest(destination);
        check_source(source);
        check_send_tag(send_tag);
        check_recv_tag(recv_tag);
        status_t s;
        MPI_Sendrecv_replace(&data, 1, detail::datatype_traits<T>::get_datatype(), destination,
                             static_cast<int>(send_tag), source, static_cast<int>(recv_tag),
                             comm_, static_cast<MPI_Status *>(&s));
        return s;
      }

      /// Sends a message and receives a message in a single operation.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param data data to send, will hold the received data
      /// \param l memory layout of the data to send and receive
      /// \param destination rank of the receiving process
      /// \param send_tag tag associated to the data to send
      /// \param source rank of the sending process
      /// \param recv_tag tag associated to the data to receive
      /// \return status of the receive operation
      template<typename T>
      status_t sendrecv_replace(T *data, const layout<T> &l, int destination, tag_t send_tag,
                                int source, tag_t recv_tag) const {
        check_dest(destination);
        check_source(source);
        check_send_tag(send_tag);
        check_recv_tag(recv_tag);
        status_t s;
        MPI_Sendrecv_replace(data, 1, detail::datatype_traits<layout<T>>::get_datatype(l),
                             destination, static_cast<int>(send_tag), source,
                             static_cast<int>(recv_tag), comm_, static_cast<MPI_Status *>(&s));
        return s;
      }

      /// Sends a message and receives a message in a single operation.
      /// \tparam iterT iterator type, must fulfill the requirements of a
      /// <a
      /// href="https://en.cppreference.com/w/cpp/named_req/ForwardIterator">LegacyForwardIterator</a>,
      /// the iterator's value-type must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param begin iterator pointing to the first data value to send and to receive
      /// \param end iterator pointing one element beyond the last data value to send and to
      /// receive
      /// \param destination rank of the receiving process
      /// \param send_tag tag associated to the data to send
      /// \param source rank of the sending process
      /// \param recvtag tag associated to the data to receive
      /// \return status of the receive operation
      template<typename iterT>
      status_t sendrecv_replace(iterT begin, iterT end, int destination, tag_t send_tag,
                                int source, tag_t recvtag) const {
        using value_type = typename std::iterator_traits<iterT>::value_type;
        if constexpr (detail::is_contiguous_iterator_v<iterT>) {
          const vector_layout<value_type> l(std::distance(begin, end));
          return sendrecv_replace(&(*begin), l, destination, send_tag, source, recvtag);
        } else {
          const iterator_layout<value_type> l(begin, end);
          return sendrecv_replace(&(*begin), l, destination, send_tag, source, recvtag);
        }
      }

      // === collective ==================================================
      // === barrier ===
      // --- blocking barrier ---
      /// Blocks until all processes in the communicator have reached this method.
      /// \note This is a collective operation and must be called by all processes in the
      /// communicator.
      void barrier() const {
        MPI_Barrier(comm_);
      }

      // --- non-blocking barrier ---
      /// Notifies the process that it has reached the barrier and returns immediately.
      /// \return communication request
      /// \note This is a collective operation and must be called by all processes in the
      /// communicator.
      [[nodiscard]] irequest ibarrier() const {
        MPI_Request req;
        MPI_Ibarrier(comm_, &req);
        return base_irequest{req};
      }

      // === broadcast ===
      // --- blocking broadcast ---
      /// Broadcasts a message from a process to all other processes.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param root_rank rank of the sending process
      /// \param data buffer for sending/receiving data
      /// \note This is a collective operation and must be called by all processes in the
      /// communicator.
      template<typename T>
      void bcast(int root_rank, T &data) const {
        check_root(root_rank);
        MPI_Bcast(&data, 1, detail::datatype_traits<T>::get_datatype(), root_rank, comm_);
      }

      /// Broadcasts a message from a process to all other processes.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param root_rank rank of the sending process
      /// \param data buffer for sending/receiving data
      /// \param l memory layout of the data to send/receive
      /// \note This is a collective operation and must be called by all processes in the
      /// communicator.
      template<typename T>
      void bcast(int root_rank, T *data, const layout<T> &l) const {
        check_root(root_rank);
        MPI_Bcast(data, 1, detail::datatype_traits<layout<T>>::get_datatype(l), root_rank,
                  comm_);
      }

      // --- non-blocking broadcast ---
      /// Broadcasts a message from a process to all other processes in a non-blocking
      /// manner.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param root_rank rank of the sending process
      /// \param data buffer for sending/receiving data
      /// \return request representing the ongoing message transfer
      /// \note This is a collective operation and must be called by all processes in the
      /// communicator.
      template<typename T>
      irequest ibcast(int root_rank, T &data) const {
        check_root(root_rank);
        MPI_Request req;
        MPI_Ibcast(&data, 1, detail::datatype_traits<T>::get_datatype(), root_rank, comm_,
                   &req);
        return base_irequest{req};
      }

      /// Broadcasts a message from a process to all other processes in a non-blocking
      /// manner.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param root_rank rank of the sending process
      /// \param data buffer for sending/receiving data
      /// \param l memory layout of the data to send/receive
      /// \return request representing the ongoing message transfer
      /// \note This is a collective operation and must be called by all processes in the
      /// communicator.
      template<typename T>
      irequest ibcast(int root_rank, T *data, const layout<T> &l) const {
        check_root(root_rank);
        MPI_Request req;
        MPI_Ibcast(data, 1, detail::datatype_traits<layout<T>>::get_datatype(l), root_rank,
                   comm_, &req);
        return base_irequest{req};
      }

      // === gather ===
      // === root gets a single value from each rank and stores in contiguous memory
      // --- blocking gather ---
      /// Gather messages from all processes at a single root process.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param root_rank rank of the receiving process
      /// \param send_data data to send
      /// \param recv_data pointer to continuous storage for incoming messages, may be a null
      /// pointer at non-root processes
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T>
      void gather(int root_rank, const T &send_data, T *recv_data) const {
        check_root(root_rank);
        MPI_Gather(&send_data, 1, detail::datatype_traits<T>::get_datatype(), recv_data, 1,
                   detail::datatype_traits<T>::get_datatype(), root_rank, comm_);
      }

      /// Gather messages from all processes at a single root process.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param root_rank rank of the receiving process
      /// \param send_data data buffer for sending data
      /// \param sendl memory layout of the data to send
      /// \param recv_data pointer to continuous storage for incoming messages, may be a null
      /// pointer at non-root processes
      /// \param recvl memory layout of the data to receive
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T>
      void gather(int root_rank, const T *send_data, const layout<T> &sendl, T *recv_data,
                  const layout<T> &recvl) const {
        check_root(root_rank);
        MPI_Gather(send_data, 1, detail::datatype_traits<layout<T>>::get_datatype(sendl),
                   recv_data, 1, detail::datatype_traits<layout<T>>::get_datatype(recvl),
                   root_rank, comm_);
      }

      // --- non-blocking gather ---
      /// Gather messages from all processes at a single root process in a non-blocking
      /// manner.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param root_rank rank of the receiving process
      /// \param send_data data to send
      /// \param recv_data pointer to continuous storage for incoming messages, may be a null
      /// pointer at non-root processes
      /// \return request representing the ongoing message transfer
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T>
      irequest igather(int root_rank, const T &send_data, T *recv_data) const {
        check_root(root_rank);
        MPI_Request req;
        MPI_Igather(&send_data, 1, detail::datatype_traits<T>::get_datatype(), recv_data, 1,
                    detail::datatype_traits<T>::get_datatype(), root_rank, comm_, &req);
        return base_irequest{req};
      }

      /// Gather messages from all processes at a single root process in a non-blocking
      /// manner.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param root_rank rank of the receiving process
      /// \param send_data data buffer for sending data
      /// \param sendl memory layout of the data to send
      /// \param recv_data pointer to continuous storage for incoming messages, may be a null
      /// pointer at non-root processes
      /// \param recvl memory layout of the data to receive
      /// \return request representing the ongoing message transfer
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T>
      irequest igather(int root_rank, const T *send_data, const layout<T> &sendl, T *recv_data,
                       const layout<T> &recvl) const {
        check_root(root_rank);
        MPI_Request req;
        MPI_Igather(send_data, 1, detail::datatype_traits<layout<T>>::get_datatype(sendl),
                    recv_data, 1, detail::datatype_traits<layout<T>>::get_datatype(recvl),
                    root_rank, comm_, &req);
        return base_irequest{req};
      }

      // --- blocking gather, non-root variant ---
      /// Gather messages from all processes at a single root process.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param root_rank rank of the receiving process
      /// \param send_data data to send
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator. This particular overload can only be
      /// called by non-root processes.
      template<typename T>
      void gather(int root_rank, const T &send_data) const {
        check_nonroot(root_rank);
        MPI_Gather(&send_data, 1, detail::datatype_traits<T>::get_datatype(), 0, 0,
                   MPI_DATATYPE_NULL, root_rank, comm_);
      }

      /// Gather messages from all processes at a single root process.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param root_rank rank of the receiving process
      /// \param send_data data buffer for sending data
      /// \param sendl memory layout of the data to send
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator. This particular overload can only be
      /// called by non-root processes.
      template<typename T>
      void gather(int root_rank, const T *send_data, const layout<T> &sendl) const {
        check_nonroot(root_rank);
        MPI_Gather(send_data, 1, detail::datatype_traits<layout<T>>::get_datatype(sendl), 0, 0,
                   MPI_DATATYPE_NULL, root_rank, comm_);
      }

      // --- non-blocking gather, non-root variant ---
      /// Gather messages from all processes at a single root process in a non-blocking
      /// manner.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param root_rank rank of the receiving process
      /// \param send_data data to send
      /// \return request representing the ongoing message transfer
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator. This particular overload can only be
      /// called by non-root processes.
      template<typename T>
      irequest igather(int root_rank, const T &send_data) const {
        check_nonroot(root_rank);
        MPI_Request req;
        MPI_Igather(&send_data, 1, detail::datatype_traits<T>::get_datatype(), 0, 0,
                    MPI_DATATYPE_NULL, root_rank, comm_, &req);
        return base_irequest{req};
      }

      /// Gather messages from all processes at a single root process in a non-blocking
      /// manner.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param root_rank rank of the receiving process
      /// \param send_data data buffer for sending data
      /// \param sendl memory layout of the data to send
      /// \return request representing the ongoing message transfer
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator. This particular overload can only be
      /// called by non-root processes.
      template<typename T>
      irequest igather(int root_rank, const T *send_data, const layout<T> &sendl) const {
        check_nonroot(root_rank);
        MPI_Request req;
        MPI_Igather(send_data, 1, detail::datatype_traits<layout<T>>::get_datatype(sendl), 0, 0,
                    MPI_DATATYPE_NULL, root_rank, comm_, &req);
        return base_irequest{req};
      }
      // === root gets varying amount of data from each rank and stores in non-contiguous memory
      // --- blocking gather ---
      /// Gather messages with a variable amount of data from all processes at a single
      /// root process.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param root_rank rank of the receiving process
      /// \param send_data data to send
      /// \param sendl memory layout of the data to send
      /// \param recv_data pointer to continuous storage for incoming messages, may be a null
      /// pointer at non-root processes
      /// \param recvls memory layouts of the data to receive by the root rank
      /// \param recvdispls displacements of the data to receive by the root rank
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T>
      void gatherv(int root_rank, const T *send_data, const layout<T> &sendl, T *recv_data,
                   const layouts<T> &recvls, const displacements &recvdispls) const {
        check_root(root_rank);
        check_size(recvls);
        check_size(recvdispls);
        const int n{size()};
        displacements senddispls(n);
        layouts<T> sendls(n);
        sendls[root_rank] = sendl;
        if (rank() == root_rank)
          alltoallv(send_data, sendls, senddispls, recv_data, recvls, recvdispls);
        else
          alltoallv(send_data, sendls, senddispls, recv_data, mpl::layouts<T>(n), recvdispls);
      }

      /// Gather messages with a variable amount of data from all processes at a single
      /// root process.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param root_rank rank of the receiving process
      /// \param send_data data to send
      /// \param sendl memory layout of the data to send
      /// \param recv_data pointer to continuous storage for incoming messages, may be a null
      /// pointer at non-root processes
      /// \param recvls memory layouts of the data to receive by the root rank
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T>
      void gatherv(int root_rank, const T *send_data, const layout<T> &sendl, T *recv_data,
                   const layouts<T> &recvls) const {
        gatherv(root_rank, send_data, sendl, recv_data, recvls, displacements(size()));
      }

      // --- non-blocking gather ---
      /// Gather messages with a variable amount of data from all processes at a single
      /// root process in a non-blocking manner.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param root_rank rank of the receiving process
      /// \param send_data data to send
      /// \param sendl memory layout of the data to send
      /// \param recv_data pointer to continuous storage for incoming messages, may be a null
      /// pointer at non-root processes
      /// \param recvls memory layouts of the data to receive by the root rank
      /// \param recvdispls displacements of the data to receive by the root rank
      /// \return request representing the ongoing message transfer
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T>
      irequest igatherv(int root_rank, const T *send_data, const layout<T> &sendl, T *recv_data,
                        const layouts<T> &recvls, const displacements &recvdispls) const {
        check_root(root_rank);
        check_size(recvls);
        check_size(recvdispls);
        const int n{size()};
        displacements senddispls(n);
        layouts<T> sendls(n);
        sendls[root_rank] = sendl;
        if (rank() == root_rank)
          return ialltoallv(send_data, sendls, senddispls, recv_data, recvls, recvdispls);
        else
          return ialltoallv(send_data, sendls, senddispls, recv_data, mpl::layouts<T>(n),
                            recvdispls);
      }

      /// Gather messages with a variable amount of data from all processes at a single
      /// root process in a non-blocking manner.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param root_rank rank of the receiving process
      /// \param send_data data to send
      /// \param sendl memory layout of the data to send
      /// \param recv_data pointer to continuous storage for incoming messages, may be a null
      /// pointer at non-root processes
      /// \param recvls memory layouts of the data to receive by the root rank
      /// \return request representing the ongoing message transfer
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T>
      irequest igatherv(int root_rank, const T *send_data, const layout<T> &sendl, T *recv_data,
                        const layouts<T> &recvls) const {
        return igatherv(root_rank, send_data, sendl, recv_data, recvls, displacements(size()));
      }

      // --- blocking gather, non-root variant ---
      /// Gather messages with a variable amount of data from all processes at a single
      /// root process.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param root_rank rank of the receiving process
      /// \param send_data data to send
      /// \param sendl memory layout of the data to send
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator. This particular overload can only be
      /// called by non-root processes.
      template<typename T>
      void gatherv(int root_rank, const T *send_data, const layout<T> &sendl) const {
        check_nonroot(root_rank);
        const int n{size()};
        displacements sendrecvdispls(n);
        layouts<T> sendls(n);
        sendls[root_rank] = sendl;
        alltoallv(send_data, sendls, sendrecvdispls, static_cast<T *>(nullptr),
                  mpl::layouts<T>(n), sendrecvdispls);
      }

      // --- non-blocking gather, non-root variant ---
      /// Gather messages with a variable amount of data from all processes at a single
      /// root process in a non-blocking manner.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param root_rank rank of the receiving process
      /// \param send_data data to send
      /// \param sendl memory layout of the data to send
      /// \return request representing the ongoing message transfer
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator. This particular overload can only be
      /// called by non-root processes.
      template<typename T>
      irequest igatherv(int root_rank, const T *send_data, const layout<T> &sendl) const {
        check_nonroot(root_rank);
        const int n{size()};
        displacements sendrecvdispls(n);
        layouts<T> sendls(n);
        sendls[root_rank] = sendl;
        return ialltoallv(send_data, sendls, sendrecvdispls, static_cast<T *>(nullptr),
                          mpl::layouts<T>(n), sendrecvdispls);
      }

      // === allgather ===
      // === get a single value from each rank and stores in contiguous memory
      // --- blocking allgather ---
      /// Gather messages from all processes and distribute result to all processes.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param send_data data to send
      /// \param recv_data pointer to continuous storage for incoming messages
      /// \note This is a collective operation and must be called (possibly by utilizing
      /// another overload) by all processes in the communicator.
      template<typename T>
      void allgather(const T &send_data, T *recv_data) const {
        MPI_Allgather(&send_data, 1, detail::datatype_traits<T>::get_datatype(), recv_data, 1,
                      detail::datatype_traits<T>::get_datatype(), comm_);
      }

      /// Gather messages from all processes and distribute result to all processes.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param send_data data to send
      /// \param sendl memory layout of the data to send
      /// \param recv_data pointer to continuous storage for incoming messages
      /// \param recvl memory layout of the data to receive
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T>
      void allgather(const T *send_data, const layout<T> &sendl, T *recv_data,
                     const layout<T> &recvl) const {
        MPI_Allgather(send_data, 1, detail::datatype_traits<layout<T>>::get_datatype(sendl),
                      recv_data, 1, detail::datatype_traits<layout<T>>::get_datatype(recvl),
                      comm_);
      }

      // --- non-blocking allgather ---
      /// Gather messages from all processes and distribute result to all processes in a
      /// non-blocking manner.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param send_data data to send
      /// \param recv_data pointer to continuous storage for incoming messages
      /// \return request representing the ongoing message transfer
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T>
      irequest iallgather(const T &send_data, T *recv_data) const {
        MPI_Request req;
        MPI_Iallgather(&send_data, 1, detail::datatype_traits<T>::get_datatype(), recv_data, 1,
                       detail::datatype_traits<T>::get_datatype(), comm_, &req);
        return base_irequest{req};
      }

      /// Gather messages from all processes and distribute result to all processes in a
      /// non-blocking manner.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param send_data data to send
      /// \param sendl memory layout of the data to send
      /// \param recv_data pointer to continuous storage for incoming messages
      /// \param recvl memory layout of the data to receive
      /// \return request representing the ongoing message transfer
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T>
      irequest iallgather(const T *send_data, const layout<T> &sendl, T *recv_data,
                          const layout<T> &recvl) const {
        MPI_Request req;
        MPI_Iallgather(send_data, 1, detail::datatype_traits<layout<T>>::get_datatype(sendl),
                       recv_data, 1, detail::datatype_traits<layout<T>>::get_datatype(recvl),
                       comm_, &req);
        return base_irequest{req};
      }

      // === get varying amount of data from each rank and stores in non-contiguous memory
      // --- blocking allgather ---
      /// Gather messages with a variable amount of data from all processes and
      /// distribute result to all processes.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param send_data data to send
      /// \param sendl memory layout of the data to send
      /// \param recv_data pointer to continuous storage for incoming messages
      /// \param recvls memory layouts of the data to receive
      /// \param recvdispls displacements of the data to receive
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T>
      void allgatherv(const T *send_data, const layout<T> &sendl, T *recv_data,
                      const layouts<T> &recvls, const displacements &recvdispls) const {
        check_size(recvls);
        check_size(recvdispls);
        const int n{size()};
        displacements senddispls(n);
        layouts<T> sendls(n, sendl);
        alltoallv(send_data, sendls, senddispls, recv_data, recvls, recvdispls);
      }

      /// Gather messages with a variable amount of data from all processes and
      /// distribute result to all processes.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param send_data data to send
      /// \param sendl memory layout of the data to send
      /// \param recv_data pointer to continuous storage for incoming messages
      /// \param recvls memory layouts of the data to receive
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T>
      void allgatherv(const T *send_data, const layout<T> &sendl, T *recv_data,
                      const layouts<T> &recvls) const {
        allgatherv(send_data, sendl, recv_data, recvls, displacements(size()));
      }

      // --- non-blocking allgather ---
      /// Gather messages with a variable amount of data from all processes and
      /// distribute result to all processes in a non-blocking manner.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param send_data data to send
      /// \param sendl memory layout of the data to send
      /// \param recv_data pointer to continuous storage for incoming messages
      /// \param recvls memory layouts of the data to receive
      /// \param recvdispls displacements of the data to receive
      /// \return request representing the ongoing message transfer
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T>
      irequest iallgatherv(const T *send_data, const layout<T> &sendl, T *recv_data,
                           const layouts<T> &recvls, const displacements &recvdispls) const {
        check_size(recvls);
        check_size(recvdispls);
        const int n{size()};
        displacements senddispls(n);
        layouts<T> sendls(n, sendl);
        return ialltoallv(send_data, sendls, senddispls, recv_data, recvls, recvdispls);
      }

      /// Gather messages with a variable amount of data from all processes and
      /// distribute result to all processes in a non-blocking manner.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param send_data data to send
      /// \param sendl memory layout of the data to send
      /// \param recv_data pointer to continuous storage for incoming messages
      /// \param recvls memory layouts of the data to receive
      /// \return request representing the ongoing message transfer
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T>
      irequest iallgatherv(const T *send_data, const layout<T> &sendl, T *recv_data,
                           const layouts<T> &recvls) const {
        return iallgatherv(send_data, sendl, recv_data, recvls, displacements(size()));
      }

      // === scatter ===
      // === root sends a single value from contiguous memory to each rank
      // --- blocking scatter ---
      /// Scatter messages from a single root process to all processes.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param root_rank rank of the sending process
      /// \param send_data pointer to continuous storage for outgoing messages, may be a null
      /// pointer at non-root processes
      /// \param recv_data data to receive
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T>
      void scatter(int root_rank, const T *send_data, T &recv_data) const {
        check_root(root_rank);
        MPI_Scatter(send_data, 1, detail::datatype_traits<T>::get_datatype(), &recv_data, 1,
                    detail::datatype_traits<T>::get_datatype(), root_rank, comm_);
      }

      /// Scatter messages from a single root process to all processes.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param root_rank rank of the sending process
      /// \param send_data pointer to continuous storage for outgoing messages, may be a null
      /// pointer at non-root processes
      /// \param sendl memory layout of the data to send
      /// \param recv_data data to receive
      /// \param recvl memory layout of the data to receive
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T>
      void scatter(int root_rank, const T *send_data, const layout<T> &sendl, T *recv_data,
                   const layout<T> &recvl) const {
        check_root(root_rank);
        MPI_Scatter(send_data, 1, detail::datatype_traits<layout<T>>::get_datatype(sendl),
                    recv_data, 1, detail::datatype_traits<layout<T>>::get_datatype(recvl),
                    root_rank, comm_);
      }

      // --- non-blocking scatter ---
      /// Scatter messages from a single root process to all processes in a non-blocking
      /// manner.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param root_rank rank of the sending process
      /// \param send_data pointer to continuous storage for outgoing messages, may be a null
      /// pointer at non-root processes
      /// \param recv_data data to receive
      /// \return request representing the ongoing message transfer
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T>
      irequest iscatter(int root_rank, const T *send_data, T &recv_data) const {
        check_root(root_rank);
        MPI_Request req;
        MPI_Iscatter(send_data, 1, detail::datatype_traits<T>::get_datatype(), &recv_data, 1,
                     detail::datatype_traits<T>::get_datatype(), root_rank, comm_, &req);
        return base_irequest{req};
      }

      /// Scatter messages from a single root process to all processes in a non-blocking
      /// manner.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param root_rank rank of the sending process
      /// \param send_data pointer to continuous storage for outgoing messages, may be a null
      /// pointer at non-root processes
      /// \param sendl memory layout of the data to send
      /// \param recv_data data to receive
      /// \param recvl memory layout of the data to receive
      /// \return request representing the ongoing message transfer
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T>
      irequest iscatter(int root_rank, const T *send_data, const layout<T> &sendl, T *recv_data,
                        const layout<T> &recvl) const {
        check_root(root_rank);
        MPI_Request req;
        MPI_Iscatter(send_data, 1, detail::datatype_traits<layout<T>>::get_datatype(sendl),
                     recv_data, 1, detail::datatype_traits<layout<T>>::get_datatype(recvl),
                     root_rank, comm_, &req);
        return base_irequest{req};
      }

      // --- blocking scatter, non-root variant ---
      /// Scatter messages from a single root process to all processes.
      /// \param root_rank rank of the sending process
      /// \param recv_data data to receive
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator. This particular overload can only be
      /// called by non-root processes.
      template<typename T>
      void scatter(int root_rank, T &recv_data) const {
        check_nonroot(root_rank);
        MPI_Scatter(0, 0, MPI_DATATYPE_NULL, &recv_data, 1,
                    detail::datatype_traits<T>::get_datatype(), root_rank, comm_);
      }

      /// Scatter messages from a single root process to all processes.
      /// \param root_rank rank of the sending process
      /// \param recv_data data to receive
      /// \param recvl memory layout of the data to receive
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator. This particular overload can only be
      /// called by non-root processes.
      template<typename T>
      void scatter(int root_rank, T *recv_data, const layout<T> &recvl) const {
        check_root(root_rank);
        MPI_Scatter(0, 0, MPI_DATATYPE_NULL, recv_data, 1,
                    detail::datatype_traits<layout<T>>::get_datatype(recvl), root_rank, comm_);
      }

      // --- non-blocking scatter, non-root variant ---
      /// Scatter messages from a single root process to all processes in a non-blocking
      /// manner.
      /// \param root_rank rank of the sending process
      /// \param recv_data data to receive
      /// \return request representing the ongoing message transfer
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator. This particular overload can only be
      /// called by non-root processes.
      template<typename T>
      irequest iscatter(int root_rank, T &recv_data) const {
        check_nonroot(root_rank);
        MPI_Request req;
        MPI_Iscatter(0, 0, MPI_DATATYPE_NULL, &recv_data, 1,
                     detail::datatype_traits<T>::get_datatype(), root_rank, comm_, &req);
        return base_irequest{req};
      }

      /// Scatter messages from a single root process to all processes in a non-blocking
      /// manner.
      /// \param root_rank rank of the sending process
      /// \param recv_data data to receive
      /// \param recvl memory layout of the data to receive
      /// \return request representing the ongoing message transfer
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator. This particular overload can only be
      /// called by non-root processes.
      template<typename T>
      irequest iscatter(int root_rank, T *recv_data, const layout<T> &recvl) const {
        check_nonroot(root_rank);
        MPI_Request req;
        MPI_Iscatter(0, 0, MPI_DATATYPE_NULL, recv_data, 1,
                     detail::datatype_traits<layout<T>>::get_datatype(recvl), root_rank, comm_,
                     &req);
        return base_irequest{req};
      }

      // === root sends varying amount of data from non-contiguous memory to each rank
      // --- blocking scatter ---
      /// Scatter messages with a variable amount of data from a single root process to all
      /// processes.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param root_rank rank of the sending process
      /// \param send_data pointer to continuous storage for outgoing messages, may be a null
      /// pointer at non-root processes
      /// \param sendls memory layouts of the data to send
      /// \param senddispls displacements of the data to send by the root rank
      /// \param recv_data pointer to continuous storage for incoming messages
      /// \param recvl memory layout of the data to receive by the root rank
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T>
      void scatterv(int root_rank, const T *send_data, const layouts<T> &sendls,
                    const displacements &senddispls, T *recv_data,
                    const layout<T> &recvl) const {
        check_root(root_rank);
        check_size(sendls);
        check_size(senddispls);
        const int n{size()};
        displacements recvdispls(n);
        layouts<T> recvls(n);
        recvls[root_rank] = recvl;
        if (rank() == root_rank)
          alltoallv(send_data, sendls, senddispls, recv_data, recvls, recvdispls);
        else
          alltoallv(send_data, sendls, senddispls, recv_data, mpl::layouts<T>(n), recvdispls);
      }

      /// Scatter messages with a variable amount of data from a single root process to all
      /// processes.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param root_rank rank of the sending process
      /// \param send_data pointer to continuous storage for outgoing messages, may be a null
      /// pointer at non-root processes
      /// \param sendls memory layouts of the data to send
      /// \param recv_data pointer to continuous storage for incoming messages
      /// \param recvl memory layout of the data to receive by the root rank
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T>
      void scatterv(int root_rank, const T *send_data, const layouts<T> &sendls, T *recv_data,
                    const layout<T> &recvl) const {
        scatterv(root_rank, send_data, sendls, displacements(size()), recv_data, recvl);
      }

      // --- non-blocking scatter ---
      /// Scatter messages with a variable amount of data from a single root process to all
      /// processes in a non-blocking manner.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param root_rank rank of the sending process
      /// \param send_data pointer to continuous storage for outgoing messages, may be a null
      /// pointer at non-root processes
      /// \param sendls memory layouts of the data to send
      /// \param senddispls displacements of the data to send by the root rank
      /// \param recv_data pointer to continuous storage for incoming messages
      /// \param recvl memory layout of the data to receive by the root rank
      /// \return request representing the ongoing message transfer
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T>
      irequest iscatterv(int root_rank, const T *send_data, const layouts<T> &sendls,
                         const displacements &senddispls, T *recv_data,
                         const layout<T> &recvl) const {
        check_root(root_rank);
        check_size(sendls);
        check_size(senddispls);
        const int n{size()};
        displacements recvdispls(n);
        layouts<T> recvls(n);
        recvls[root_rank] = recvl;
        if (rank() == root_rank)
          return ialltoallv(send_data, sendls, senddispls, recv_data, recvls, recvdispls);
        else
          return ialltoallv(send_data, sendls, senddispls, recv_data, mpl::layouts<T>(n),
                            recvdispls);
      }

      /// Scatter messages with a variable amount of data from a single root process to all
      /// processes  in a non-blocking manner.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param root_rank rank of the sending process
      /// \param send_data pointer to continuous storage for outgoing messages, may be a null
      /// pointer at non-root processes
      /// \param sendls memory layouts of the data to send
      /// \param recv_data pointer to continuous storage for incoming messages
      /// \param recvl memory layout of the data to receive by the root rank
      /// \return request representing the ongoing message transfer
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T>
      irequest iscatterv(int root_rank, const T *send_data, const layouts<T> &sendls,
                         T *recv_data, const layout<T> &recvl) const {
        return iscatterv(root_rank, send_data, sendls, displacements(size()), recv_data, recvl);
      }

      // --- blocking scatter, non-root variant ---
      /// Scatter messages with a variable amount of data from a single root process to all
      /// processes.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param root_rank rank of the sending process
      /// \param recv_data pointer to continuous storage for incoming messages
      /// \param recvl memory layout of the data to receive by the root rank
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator. This particular overload can only be
      /// called by non-root processes.
      template<typename T>
      void scatterv(int root_rank, T *recv_data, const layout<T> &recvl) const {
        check_root(root_rank);
        const int n{size()};
        displacements sendrecvdispls(n);
        layouts<T> recvls(n);
        recvls[root_rank] = recvl;
        alltoallv(static_cast<const T *>(nullptr), mpl::layouts<T>(n), sendrecvdispls,
                  recv_data, recvls, sendrecvdispls);
      }

      // --- non-blocking scatter, non-root variant ---
      /// Scatter messages with a variable amount of data from a single root process to all
      /// processes in a non-blocking manner.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param root_rank rank of the sending process
      /// \param recv_data pointer to continuous storage for incoming messages
      /// \param recvl memory layout of the data to receive by the root rank
      /// \return request representing the ongoing message transfer
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator. This particular overload can only be
      /// called by non-root processes.
      template<typename T>
      irequest iscatterv(int root_rank, T *recv_data, const layout<T> &recvl) const {
        check_root(root_rank);
        const int n{size()};
        displacements sendrecvdispls(n);
        layouts<T> recvls(n);
        recvls[root_rank] = recvl;
        return ialltoallv(static_cast<const T *>(nullptr), mpl::layouts<T>(n), sendrecvdispls,
                          recv_data, recvls, sendrecvdispls);
      }

      // === all-to-all ===
      // === each rank sends a single value to each rank
      // --- blocking all-to-all ---
      /// Sends messages to all processes and receives messages from all processes.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param send_data pointer to continuous storage for outgoing messages
      /// \param recv_data pointer to continuous storage for incoming messages
      /// \details Each process in the communicator sends one element of type \c T to each
      /// process (including itself) and receives one element of type \c T from each process.
      /// The i-th element in the array \c send_data is sent to the i-th process.  When the
      /// function has finished, the i-th element in the array \c recv_data was received from
      /// the i-th process.
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T>
      void alltoall(const T *send_data, T *recv_data) const {
        MPI_Alltoall(send_data, 1, detail::datatype_traits<T>::get_datatype(), recv_data, 1,
                     detail::datatype_traits<T>::get_datatype(), comm_);
      }

      /// Sends messages to all processes and receives messages from all processes.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param send_data pointer to continuous storage for outgoing messages
      /// \param sendl memory layouts of the data to send
      /// \param recv_data pointer to continuous storage for incoming messages
      /// \param recvl memory layouts of the data to receive
      /// \details Each process in the communicator sends elements of type \c T to each process
      /// (including itself) and receives elements of type \c T from each process. The memory
      /// layouts of the incoming and the outgoing messages are described by \c sendl and
      /// \c recvl. Both layouts might differ but must be compatible, i.e., must hold the same
      /// number of elements of type \c T.  The i-th memory block with the layout \c sendl in
      /// the array \c send_data is sent to the i-th process.  When the function has finished,
      /// the i-th memory block with the layout \c recvl in the array \c recv_data was received
      /// from the i-th process.
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T>
      void alltoall(const T *send_data, const layout<T> &sendl, T *recv_data,
                    const layout<T> &recvl) const {
        MPI_Alltoall(send_data, 1, detail::datatype_traits<layout<T>>::get_datatype(sendl),
                     recv_data, 1, detail::datatype_traits<layout<T>>::get_datatype(recvl),
                     comm_);
      }

      // --- non-blocking all-to-all ---
      /// Sends messages to all processes and receives messages from all processes in a
      /// non-blocking manner.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param send_data pointer to continuous storage for outgoing messages
      /// \param recv_data pointer to continuous storage for incoming messages
      /// \return request representing the ongoing message transfer
      /// \details Each process in the communicator sends one element of type \c T to each
      /// process (including itself) and receives one element of type \c T from each process.
      /// The i-th element in the array \c send_data is sent to the i-th process.  When the
      /// message transfer has finished, the i-th element in the array \c recv_data was received
      /// from the i-th process.
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T>
      irequest ialltoall(const T *send_data, T *recv_data) const {
        MPI_Request req;
        MPI_Ialltoall(send_data, 1, detail::datatype_traits<T>::get_datatype(), recv_data, 1,
                      detail::datatype_traits<T>::get_datatype(), comm_, &req);
        return base_irequest{req};
      }

      /// Sends messages to all processes and receives messages from all processes in a
      /// non-blocking manner.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param send_data pointer to continuous storage for outgoing messages
      /// \param sendl memory layouts of the data to send
      /// \param recv_data pointer to continuous storage for incoming messages
      /// \param recvl memory layouts of the data to receive
      /// \return request representing the ongoing message transfer
      /// \details Each process in the communicator sends elements of type \c T to each process
      /// (including itself) and receives elements of type \c T from each process. The memory
      /// layouts of the incoming and the outgoing messages are described by \c sendl and
      /// \c recvl. Both layouts might differ but must be compatible, i.e., must hold the same
      /// number of elements of type \c T.  The i-th memory block with the layout \c sendl in
      /// the array \c send_data is sent to the i-th process.  When the message transfer has
      /// finished, the i-th memory block with the layout \c recvl in the array \c recv_data was
      /// received from the i-th process.
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T>
      irequest ialltoall(const T *send_data, const layout<T> &sendl, T *recv_data,
                         const layout<T> &recvl) const {
        MPI_Request req;
        MPI_Ialltoall(send_data, 1, detail::datatype_traits<layout<T>>::get_datatype(sendl),
                      recv_data, 1, detail::datatype_traits<layout<T>>::get_datatype(recvl),
                      comm_, &req);
        return base_irequest{req};
      }

      // === each rank sends a varying number of values to each rank with possibly different
      // layouts
      // --- blocking all-to-all ---
      /// Sends messages with a variable amount of data to all processes and receives
      /// messages with a variable amount of data from all processes.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param send_data pointer to continuous storage for outgoing messages
      /// \param sendls memory layouts of the data to send
      /// \param senddispls displacements of the data to send
      /// \param recv_data pointer to continuous storage for incoming messages
      /// \param recvls memory layouts of the data to receive
      /// \param recvdispls displacements of the data to receive
      /// \details Each process in the communicator sends elements of type \c T to each process
      /// (including itself) and receives elements of type \c T from each process.  Send- and
      /// receive-data are stored in consecutive blocks of variable size in the buffers
      /// \c send_data and \c recv_data, respectively. The i-th memory block with the layout
      /// <tt>sendls[i]</tt> in the array \c send_data starts \c senddispls[i] bytes after the address
      /// given in send_data. The i-th memory block is sent to the i-th process. The i-th memory
      /// block with the layout <tt>recvls[i]</tt> in the array recv_data starts \c recvdispls[i]
      /// bytes after the address given in \c recv_data.  When the function has finished, the
      /// i-th memory block in the array \c recv_data was received from the i-th process.
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T>
      void alltoallv(const T *send_data, const layouts<T> &sendls,
                     const displacements &senddispls, T *recv_data, const layouts<T> &recvls,
                     const displacements &recvdispls) const {
        check_size(senddispls);
        check_size(sendls);
        check_size(recvdispls);
        check_size(recvls);
        const std::vector<int> counts(recvls.size(), 1);
        const std::vector<int> senddispls_int(senddispls.begin(), senddispls.end());
        const std::vector<int> recvdispls_int(recvdispls.begin(), recvdispls.end());
        static_assert(
            sizeof(decltype(*sendls())) == sizeof(MPI_Datatype),
            "compiler adds some unexpected padding, reinterpret cast will yield wrong results");
        MPI_Alltoallw(send_data, counts.data(), senddispls_int.data(),
                      reinterpret_cast<const MPI_Datatype *>(sendls()), recv_data,
                      counts.data(), recvdispls_int.data(),
                      reinterpret_cast<const MPI_Datatype *>(recvls()), comm_);
      }

      /// Sends messages with a variable amount of data to all processes and receives
      /// messages with a variable amount of data from all processes.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param send_data pointer to continuous storage for outgoing messages
      /// \param sendls memory layouts of the data to send
      /// \param recv_data pointer to continuous storage for incoming messages
      /// \param recvls memory layouts of the data to receive
      /// \details Each process in the communicator sends elements of type \c T to each process
      /// (including itself) and receives elements of type \c T from each process.  Send- and
      /// receive-data are stored in consecutive blocks of variable size in the buffers
      /// \c send_data and \c recv_data, respectively. The i-th memory block with the layout
      /// <tt>sendls[i]</tt> in the array \c send_data starts at the address given in \c send_data.
      /// The i-th memory block is sent to the i-th process. The i-th memory block with the
      /// \c layout recvls[i] in the array \c recv_data starts at the address given in
      /// \c recv_data.  Note that the memory layouts need to include appropriate holes at the
      /// beginning in order to avoid overlapping send- or receive blocks. When the function has
      /// finished, the i-th memory block in the array \c recv_data was received from the i-th
      /// process.
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T>
      void alltoallv(const T *send_data, const layouts<T> &sendls, T *recv_data,
                     const layouts<T> &recvls) const {
        const displacements sendrecvdispls(size());
        alltoallv(send_data, sendls, sendrecvdispls, recv_data, recvls, sendrecvdispls);
      }

      // --- non-blocking all-to-all ---
    protected:
      template<typename T>
      struct ialltoallv_state {
        MPI_Request req{};
        layouts<T> sendl;
        layouts<T> recvl;
        std::vector<int> counts;
        std::vector<int> senddispls_int;
        std::vector<int> recvdispls_int;
        MPI_Status status{};
        ialltoallv_state(const layouts<T> &sendl, const layouts<T> &recvl,
                         std::vector<int> &&counts, std::vector<int> &&senddispls_int,
                         std::vector<int> &&recvdispls_int)
            : sendl{sendl},
              recvl{recvl},
              counts{std::move(counts)},
              senddispls_int{std::move(senddispls_int)},
              recvdispls_int{std::move(recvdispls_int)} {
        }
        ialltoallv_state(const layouts<T> &recvl, std::vector<int> &&counts,
                         std::vector<int> &&recvdispls_int)
            : sendl{},
              recvl{recvl},
              counts{std::move(counts)},
              senddispls_int{},
              recvdispls_int{std::move(recvdispls_int)} {
        }
      };

      template<typename T>
      static int ialltoallv_query(void *state, MPI_Status *s) {
        auto *sendrecv_state{static_cast<ialltoallv_state<T> *>(state)};
        const int error_backup{s->MPI_ERROR};
        *s = sendrecv_state->status;
        s->MPI_ERROR = error_backup;
        return MPI_SUCCESS;
      }

      template<typename T>
      static int ialltoallv_free(void *state) {
        auto *sendrecv_state{static_cast<ialltoallv_state<T> *>(state)};
        delete sendrecv_state;
        return MPI_SUCCESS;
      }

      static int ialltoallv_cancel([[maybe_unused]] void *state,
                                   [[maybe_unused]] int complete) {
        return MPI_SUCCESS;
      }

      template<typename T>
      void ialltoallv_task(const T *send_data, T *recv_data, ialltoallv_state<T> *state) const {
        MPI_Request req;
        static_assert(
            sizeof(decltype(*state->sendl())) == sizeof(MPI_Datatype),
            "compiler adds some unexpected padding, reinterpret cast will yield wrong results");
        if (send_data != nullptr)
          MPI_Ialltoallw(send_data, state->counts.data(), state->senddispls_int.data(),
                         reinterpret_cast<const MPI_Datatype *>(state->sendl()), recv_data,
                         state->counts.data(), state->recvdispls_int.data(),
                         reinterpret_cast<const MPI_Datatype *>(state->recvl()), comm_, &req);
        else
          MPI_Ialltoallw(MPI_IN_PLACE, 0, 0, 0, recv_data, state->counts.data(),
                         state->recvdispls_int.data(),
                         reinterpret_cast<const MPI_Datatype *>(state->recvl()), comm_, &req);
        MPI_Status s;
        MPI_Wait(&req, &s);
        state->status = s;
        MPI_Grequest_complete(state->req);
      }

    public:
      /// Sends messages with a variable amount of data to all processes and receives
      /// messages with a variable amount of data from all processes in a non-blocking manner.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param send_data pointer to continuous storage for outgoing messages
      /// \param sendls memory layouts of the data to send
      /// \param senddispls displacements of the data to send
      /// \param recv_data pointer to continuous storage for incoming messages
      /// \param recvls memory layouts of the data to receive
      /// \param recvdispls displacements of the data to receive
      /// \return request representing the ongoing message transfer
      /// \details Each process in the communicator sends elements of type \c T to each process
      /// (including itself) and receives elements of type \c T from each process.  Send- and
      /// receive-data are stored in consecutive blocks of variable size in the buffers
      /// \c send_data and \c recv_data, respectively. The i-th memory block with the layout
      /// <tt>sendls[i]</tt> in the array \c send_data starts \c senddispls[i] bytes after the address
      /// given in send_data. The i-th memory block is sent to the i-th process. The i-th memory
      /// block with the layout <tt>recvls[i]</tt> in the array \c recv_data starts \c recvdispls[i]
      /// bytes after the address given in \c recv_data.  When the function has finished, the
      /// i-th memory block in the array \c recv_data was received from the i-th process.
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T>
      irequest ialltoallv(const T *send_data, const layouts<T> &sendls,
                          const displacements &senddispls, T *recv_data,
                          const layouts<T> &recvls, const displacements &recvdispls) const {
        check_size(senddispls);
        check_size(sendls);
        check_size(recvdispls);
        check_size(recvls);
        auto *state{
            new ialltoallv_state<T>(sendls, recvls, std::vector<int>(recvls.size(), 1),
                                    std::vector<int>(senddispls.begin(), senddispls.end()),
                                    std::vector<int>(recvdispls.begin(), recvdispls.end()))};
        MPI_Request req;
        MPI_Grequest_start(ialltoallv_query<T>, ialltoallv_free<T>, ialltoallv_cancel, state,
                           &req);
        state->req = req;
        std::thread thread([this, send_data, recv_data, state]() {
          ialltoallv_task(send_data, recv_data, state);
        });
        thread.detach();
        return base_irequest{req};
      }

      /// Sends messages with a variable amount of data to all processes and receives
      /// messages with a variable amount of data from all processes in a non-blocking manner.
      /// \tparam T type of the data to send, must meet the requirements as described in the
      /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
      /// \param send_data pointer to continuous storage for outgoing messages
      /// \param sendls memory layouts of the data to send
      /// \param recv_data pointer to continuous storage for incoming messages
      /// \param recvls memory layouts of the data to receive
      /// \return request representing the ongoing message transfer
      /// \details Each process in the communicator sends elements of type \c T to each process
      /// (including itself) and receives elements of type \c T from each process.  Send- and
      /// receive-data are stored in consecutive blocks of variable size in the buffers
      /// \c send_data and \c recv_data, respectively. The i-th memory block with the layout
      /// <tt>sendls[i]</tt> in the array \c send_data starts at the address given in \c send_data.
      /// The i-th memory block is sent to the i-th process. The i-th memory block with the
      /// layout <tt>recvls[i]</tt> in the array \c recv_data starts at the address given in
      /// \c recv_data.  Note that the memory layouts need to include appropriate holes at the
      /// beginning in order to avoid overlapping send- or receive blocks. When the function has
      /// finished, the i-th memory block in the array \c recv_data was received from the i-th
      /// process.
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T>
      irequest ialltoallv(const T *send_data, const layouts<T> &sendls, T *recv_data,
                          const layouts<T> &recvls) const {
        const displacements sendrecvdispls(size());
        return ialltoallv(send_data, sendls, sendrecvdispls, recv_data, recvls, sendrecvdispls);
      }

      // === reduce ===
      // --- blocking reduce ---
      /// Performs a reduction operation over all processes.
      /// \tparam F type representing the reduction operation, reduction operation is performed
      /// on data of type \c T
      /// \tparam T type of input and output data of the reduction operation, must meet the
      /// requirements as described in the \verbatim embed:rst:inline :doc:`data_types` \endverbatim
      /// section
      /// \param f reduction operation
      /// \param root_rank rank of the process that will receive the reduction result
      /// \param send_data input data for the reduction operation
      /// \param recv_data will hold the result of the reduction operation if rank equals root_rank
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T, typename F>
      void reduce(F f, int root_rank, const T &send_data, T &recv_data) const {
        check_root(root_rank);
        MPI_Reduce(&send_data, &recv_data, 1, detail::datatype_traits<T>::get_datatype(),
                   detail::get_op<T, F>(f).mpi_op, root_rank, comm_);
      }

      /// Performs a reduction operation over all processes.
      /// \tparam F type representing the element-wise reduction operation, reduction operation is
      /// performed on data of type \c T
      /// \tparam T type of input and output data of the reduction operation, must meet the
      /// requirements as described in the \verbatim embed:rst:inline :doc:`data_types` \endverbatim
      /// section
      /// \param f reduction operation
      /// \param root_rank rank of the process that will receive the reduction result
      /// \param send_data input buffer for the reduction operation
      /// \param recv_data will hold the results of the reduction operation if rank equals
      /// root_rank, may be nullptr if rank does no equal to root_rank
      /// \param l memory layouts of the data to send and to receive
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      /// \anchor communicator_reduce_contiguous_layout
      template<typename T, typename F>
      void reduce(F f, int root_rank, const T *send_data, T *recv_data,
                  const contiguous_layout<T> &l) const {
        check_root(root_rank);
        MPI_Reduce(send_data, recv_data, l.size(), detail::datatype_traits<T>::get_datatype(),
                   detail::get_op<T, F>(f).mpi_op, root_rank, comm_);
      }

      // --- non-blocking reduce ---
      /// Performs a reduction operation over all processes in a non-blocking manner.
      /// \tparam F type representing the reduction operation, reduction operation is performed
      /// on data of type \c T
      /// \tparam T type of input and output data of the reduction operation, must meet the
      /// requirements as described in the \verbatim embed:rst:inline :doc:`data_types` \endverbatim
      /// section
      /// \param f reduction operation
      /// \param root_rank rank of the process that will receive the reduction result
      /// \param send_data input data for the reduction operation
      /// \param recv_data will hold the result of the reduction operation if rank equals root_rank
      /// \return request representing the ongoing reduction operation
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T, typename F>
      irequest ireduce(F f, int root_rank, const T &send_data, T &recv_data) const {
        check_root(root_rank);
        MPI_Request req;
        MPI_Ireduce(&send_data, &recv_data, 1, detail::datatype_traits<T>::get_datatype(),
                    detail::get_op<T, F>(f).mpi_op, root_rank, comm_, &req);
        return base_irequest{req};
      }

      /// Performs a reduction operation over all processes in a non-blocking manner.
      /// \tparam F type representing the element-wise reduction operation, reduction operation is
      /// performed on data of type \c T
      /// \tparam T type of input and output data of the reduction operation, must meet the
      /// requirements as described in the \verbatim embed:rst:inline :doc:`data_types` \endverbatim
      /// section
      /// \param f reduction operation
      /// \param root_rank rank of the process that will receive the reduction result
      /// \param send_data input buffer for the reduction operation
      /// \param recv_data will hold the results of the reduction operation if rank equals
      /// root_rank, may be nullptr if rank does no equal to root_rank
      /// \param l memory layouts of the data to send and to receive
      /// \return request representing the ongoing reduction operation
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T, typename F>
      irequest ireduce(F f, int root_rank, const T *send_data, T *recv_data,
                       const contiguous_layout<T> &l) const {
        check_root(root_rank);
        MPI_Request req;
        MPI_Ireduce(send_data, recv_data, l.size(), detail::datatype_traits<T>::get_datatype(),
                    detail::get_op<T, F>(f).mpi_op, root_rank, comm_, &req);
        return base_irequest{req};
      }

      // === all-reduce ===
      // --- blocking all-reduce ---
      /// Performs a reduction operation over all processes and broadcasts the result.
      /// \tparam F type representing the reduction operation, reduction operation is performed
      /// on data of type \c T
      /// \tparam T type of input and output data of the reduction operation, must meet the
      /// requirements as described in the \verbatim embed:rst:inline :doc:`data_types` \endverbatim
      /// section
      /// \param f reduction operation
      /// \param send_data input data for the reduction operation
      /// \param recv_data will hold the result of the reduction operation
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T, typename F>
      void allreduce(F f, const T &send_data, T &recv_data) const {
        MPI_Allreduce(&send_data, &recv_data, 1, detail::datatype_traits<T>::get_datatype(),
                      detail::get_op<T, F>(f).mpi_op, comm_);
      }

      /// Performs a reduction operation over all processes and broadcasts the result.
      /// \tparam F type representing the element-wise reduction operation, reduction operation is
      /// performed on data of type \c T
      /// \tparam T type of input and output data of the reduction operation, must meet the
      /// requirements as described in the \verbatim embed:rst:inline :doc:`data_types` \endverbatim
      /// section
      /// \param f reduction operation
      /// \param send_data input buffer for the reduction operation
      /// \param recv_data will hold the results of the reduction operation
      /// \param l memory layouts of the data to send and to receive
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T, typename F>
      void allreduce(F f, const T *send_data, T *recv_data,
                     const contiguous_layout<T> &l) const {
        MPI_Allreduce(send_data, recv_data, l.size(),
                      detail::datatype_traits<T>::get_datatype(),
                      detail::get_op<T, F>(f).mpi_op, comm_);
      }

      // --- non-blocking all-reduce ---
      /// Performs a reduction operation over all processes and broadcasts the result in a
      /// non-blocking manner.
      /// \tparam F type representing the reduction operation, reduction operation is performed
      /// on data of type \c T
      /// \tparam T type of input and output data of the reduction operation, must meet the
      /// requirements as described in the \verbatim embed:rst:inline :doc:`data_types` \endverbatim
      /// section
      /// \param f reduction operation
      /// \param send_data input data for the reduction operation
      /// \param recv_data will hold the result of the reduction operation
      /// \return request representing the ongoing reduction operation
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T, typename F>
      irequest iallreduce(F f, const T &send_data, T &recv_data) const {
        MPI_Request req;
        MPI_Iallreduce(&send_data, &recv_data, 1, detail::datatype_traits<T>::get_datatype(),
                       detail::get_op<T, F>(f).mpi_op, comm_, &req);
        return base_irequest{req};
      }

      /// Performs a reduction operation over all processes and broadcasts the result in a
      /// non-blocking manner.
      /// \tparam F type representing the element-wise reduction operation, reduction operation is
      /// performed on data of type \c T
      /// \tparam T type of input and output data of the reduction operation, must meet the
      /// requirements as described in the \verbatim embed:rst:inline :doc:`data_types` \endverbatim
      /// section
      /// \param f reduction operation
      /// \param send_data input buffer for the reduction operation
      /// \param recv_data will hold the results of the reduction operation
      /// \param l memory layouts of the data to send and to receive
      /// \return request representing the ongoing reduction operation
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T, typename F>
      irequest iallreduce(F f, const T *send_data, T *recv_data,
                          const contiguous_layout<T> &l) const {
        MPI_Request req;
        MPI_Iallreduce(send_data, recv_data, l.size(),
                       detail::datatype_traits<T>::get_datatype(),
                       detail::get_op<T, F>(f).mpi_op, comm_, &req);
        return base_irequest{req};
      }

      // === reduce-scatter-block ===
      // --- blocking reduce-scatter-block ---
      /// Performs a reduction operation over all processes and scatters the result.
      /// \tparam F type representing the reduction operation, reduction operation is performed
      /// on data of type \c T
      /// \tparam T type of input and output data of the reduction operation, must meet the
      /// requirements as described in the \verbatim embed:rst:inline :doc:`data_types` \endverbatim
      /// section
      /// \param f reduction operation
      /// \param send_data input data for the reduction operation, number of elements in buffer
      /// send_data must equal the size of the communicator
      /// \param recv_data will hold the result of the reduction operation
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T, typename F>
      void reduce_scatter_block(F f, const T *send_data, T &recv_data) const {
        MPI_Reduce_scatter_block(send_data, &recv_data, 1,
                                 detail::datatype_traits<T>::get_datatype(),
                                 detail::get_op<T, F>(f).mpi_op, comm_);
      }

      /// Performs a reduction operation over all processes and scatters the result.
      /// \tparam F type representing the element-wise reduction operation, reduction operation is
      /// performed on data of type \c T
      /// \tparam T type of input and output data of the reduction operation, must meet the
      /// requirements as described in the \verbatim embed:rst:inline :doc:`data_types` \endverbatim
      /// section
      /// \param f reduction operation
      /// \param send_data input data for the reduction operation, number of elements in buffer
      /// send_data must equal the size of the communicator times the number of elements given by
      /// the layout parameter
      /// \param recv_data will hold the results of the reduction operation
      /// \param recvcount memory layouts of the data to send and to receive
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T, typename F>
      void reduce_scatter_block(F f, const T *send_data, T *recv_data,
                                const contiguous_layout<T> &recvcount) const {
        MPI_Reduce_scatter_block(send_data, recv_data, recvcount.size(),
                                 detail::datatype_traits<T>::get_datatype(),
                                 detail::get_op<T, F>(f).mpi_op, comm_);
      }

      // --- non-blocking reduce-scatter-block ---
      /// Performs a reduction operation over all processes and scatters the result in a
      /// non-blocking manner.
      /// \tparam F type representing the reduction operation, reduction operation is performed
      /// on data of type \c T
      /// \tparam T type of input and output data of the reduction operation, must meet the
      /// requirements as described in the \verbatim embed:rst:inline :doc:`data_types` \endverbatim
      /// section
      /// \param f reduction operation
      /// \param send_data input buffer for the reduction operation, number of elements in buffer
      /// send_data must equal the size of the communicator
      /// \param recv_data will hold the result of the reduction operation
      /// \return request representing the ongoing reduction operation
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T, typename F>
      irequest ireduce_scatter_block(F f, const T *send_data, T &recv_data) const {
        MPI_Request req;
        MPI_Ireduce_scatter_block(send_data, &recv_data, 1,
                                  detail::datatype_traits<T>::get_datatype(),
                                  detail::get_op<T, F>(f).mpi_op, comm_, &req);
        return base_irequest{req};
      }

      /// Performs a reduction operation over all processes and scatters the result in a
      /// non-blocking manner.
      /// \tparam F type representing the element-wise reduction operation, reduction operation is
      /// performed on data of type \c T
      /// \tparam T type of input and output data of the reduction operation, must meet the
      /// requirements as described in the \verbatim embed:rst:inline :doc:`data_types` \endverbatim
      /// section
      /// \param f reduction operation
      /// \param send_data input buffer for the reduction operation, number of elements in buffer
      /// send_data must equal the size of the communicator times the number of elements given by
      /// the layout parameter
      /// \param recv_data will hold the results of the reduction operation
      /// \param recvcount memory layouts of the data to send and to receive
      /// \return request representing the ongoing reduction operation
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T, typename F>
      irequest ireduce_scatter_block(F f, const T *send_data, T *recv_data,
                                     const contiguous_layout<T> &recvcount) const {
        MPI_Request req;
        MPI_Ireduce_scatter_block(send_data, recv_data, recvcount.size(),
                                  detail::datatype_traits<T>::get_datatype(),
                                  detail::get_op<T, F>(f).mpi_op, comm_, &req);
        return base_irequest{req};
      }

      // === reduce-scatter ===
      // --- blocking reduce-scatter ---
      /// Performs a reduction operation over all processes and scatters the result.
      /// \tparam F type representing the element-wise reduction operation, reduction operation is
      /// performed on data of type \c T
      /// \tparam T type of input and output data of the reduction operation, must meet the
      /// requirements as described in the \verbatim embed:rst:inline :doc:`data_types` \endverbatim
      /// section
      /// \param f reduction operation
      /// \param send_data input data for the reduction operation, number of elements in buffer
      /// send_data must equal the sum of the number of elements given by the collection of layout
      /// parameters
      /// \param recv_data will hold the results of the reduction operation
      /// \param recvcounts memory layouts of the data to send and to receive
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T, typename F>
      void reduce_scatter(F f, const T *send_data, T *recv_data,
                          const contiguous_layouts<T> &recvcounts) const {
        MPI_Reduce_scatter(send_data, recv_data, recvcounts.sizes(),
                           detail::datatype_traits<T>::get_datatype(),
                           detail::get_op<T, F>(f).mpi_op, comm_);
      }

      // --- non-blocking reduce-scatter ---
      /// Performs a reduction operation over all processes and scatters the result in a
      /// non-blocking manner.
      /// \tparam F type representing the element-wise reduction operation, reduction operation is
      /// performed on data of type \c T
      /// \tparam T type of input and output data of the reduction operation, must meet the
      /// requirements as described in the \verbatim embed:rst:inline :doc:`data_types` \endverbatim
      /// section
      /// \param f reduction operation
      /// \param send_data input data for the reduction operation, number of elements in buffer
      /// send_data must equal the sum of the number of elements given by the collection of layout
      /// parameters
      /// \param recv_data will hold the results of the reduction operation
      /// \param recvcounts memory layouts of the data to send and to receive
      /// \return request representing the ongoing reduction operation
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T, typename F>
      irequest ireduce_scatter(F f, const T *send_data, T *recv_data,
                               contiguous_layouts<T> &recvcounts) const {
        MPI_Request req;
        MPI_Ireduce_scatter(send_data, recv_data, recvcounts.sizes(),
                            detail::datatype_traits<T>::get_datatype(),
                            detail::get_op<T, F>(f).mpi_op, comm_, &req);
        return base_irequest{req};
      }

      // === scan ===
      // --- blocking scan ---
      /// Performs partial reduction operation (scan) over all processes.
      /// \tparam F type representing the reduction operation, reduction operation is performed
      /// on data of type \c T
      /// \tparam T type of input and output data of the reduction operation, must meet the
      /// requirements as described in the \verbatim embed:rst:inline :doc:`data_types` \endverbatim
      /// section
      /// \param f reduction operation
      /// \param send_data input data for the reduction operation
      /// \param recv_data will hold the result of the reduction operation
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T, typename F>
      void scan(F f, const T &send_data, T &recv_data) const {
        MPI_Scan(&send_data, &recv_data, 1, detail::datatype_traits<T>::get_datatype(),
                 detail::get_op<T, F>(f).mpi_op, comm_);
      }

      /// Performs a partial reduction operation (scan) over all processes.
      /// \tparam F type representing the element-wise reduction operation, reduction operation is
      /// performed on data of type \c T
      /// \tparam T type of input and output data of the reduction operation, must meet the
      /// requirements as described in the \verbatim embed:rst:inline :doc:`data_types` \endverbatim
      /// section
      /// \param f reduction operation
      /// \param send_data input buffer for the reduction operation
      /// \param recv_data will hold the results of the reduction operation
      /// \param l memory layouts of the data to send and to receive
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T, typename F>
      void scan(F f, const T *send_data, T *recv_data, const contiguous_layout<T> &l) const {
        MPI_Scan(send_data, recv_data, l.size(), detail::datatype_traits<T>::get_datatype(),
                 detail::get_op<T, F>(f).mpi_op, comm_);
      }

      // --- non-blocking scan ---
      /// Performs a partial reduction operation (scan) over all processes in a
      /// non-blocking manner.
      /// \tparam F type representing the reduction operation, reduction operation is performed
      /// on data of type \c T
      /// \tparam T type of input and output data of the reduction operation, must meet the
      /// requirements as described in the \verbatim embed:rst:inline :doc:`data_types` \endverbatim
      /// section
      /// \param f reduction operation
      /// \param send_data input data for the reduction operation
      /// \param recv_data will hold the result of the reduction operation
      /// \return request representing the ongoing reduction operation
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T, typename F>
      irequest iscan(F f, const T &send_data, T &recv_data) const {
        MPI_Request req;
        MPI_Iscan(&send_data, &recv_data, 1, detail::datatype_traits<T>::get_datatype(),
                  detail::get_op<T, F>(f).mpi_op, comm_, &req);
        return base_irequest{req};
      }

      /// Performs a partial reduction operation (scan) over all processes in a
      /// non-blocking manner.
      /// \tparam F type representing the element-wise reduction operation, reduction operation is
      /// performed on data of type \c T
      /// \tparam T type of input and output data of the reduction operation, must meet the
      /// requirements as described in the \verbatim embed:rst:inline :doc:`data_types` \endverbatim
      /// section
      /// \param f reduction operation
      /// \param send_data input buffer for the reduction operation
      /// \param recv_data will hold the results of the reduction operation
      /// \param l memory layouts of the data to send and to receive
      /// \return request representing the ongoing reduction operation
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T, typename F>
      irequest iscan(F f, const T *send_data, T *recv_data,
                     const contiguous_layout<T> &l) const {
        MPI_Request req;
        MPI_Iscan(send_data, recv_data, l.size(), detail::datatype_traits<T>::get_datatype(),
                  detail::get_op<T, F>(f).mpi_op, comm_, &req);
        return base_irequest{req};
      }

      // === exscan ===
      // --- blocking exscan ---
      /// Performs partial reduction operation (exclusive scan) over all processes.
      /// \tparam F type representing the reduction operation, reduction operation is performed
      /// on data of type \c T
      /// \tparam T type of input and output data of the reduction operation, must meet the
      /// requirements as described in the \verbatim embed:rst:inline :doc:`data_types` \endverbatim
      /// section
      /// \param f reduction operation
      /// \param send_data input data for the reduction operation
      /// \param recv_data will hold the result of the reduction operation
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T, typename F>
      void exscan(F f, const T &send_data, T &recv_data) const {
        MPI_Exscan(&send_data, &recv_data, 1, detail::datatype_traits<T>::get_datatype(),
                   detail::get_op<T, F>(f).mpi_op, comm_);
      }

      /// Performs a partial reduction operation (exclusive scan) over all processes.
      /// \tparam F type representing the element-wise reduction operation, reduction operation is
      /// performed on data of type \c T
      /// \tparam T type of input and output data of the reduction operation, must meet the
      /// requirements as described in the \verbatim embed:rst:inline :doc:`data_types` \endverbatim
      /// section
      /// \param f reduction operation
      /// \param send_data input buffer for the reduction operation
      /// \param recv_data will hold the results of the reduction operation
      /// \param l memory layouts of the data to send and to receive
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T, typename F>
      void exscan(F f, const T *send_data, T *recv_data, const contiguous_layout<T> &l) const {
        MPI_Exscan(send_data, recv_data, l.size(), detail::datatype_traits<T>::get_datatype(),
                   detail::get_op<T, F>(f).mpi_op, comm_);
      }

      // --- non-blocking exscan ---
      /// Performs a partial reduction operation (exclusive scan) over all processes in a
      /// non-blocking manner.
      /// \tparam F type representing the reduction operation, reduction operation is performed
      /// on data of type \c T
      /// \tparam T type of input and output data of the reduction operation, must meet the
      /// requirements as described in the \verbatim embed:rst:inline :doc:`data_types` \endverbatim
      /// section
      /// \param f reduction operation
      /// \param send_data input data for the reduction operation
      /// \param recv_data will hold the result of the reduction operation
      /// \return request representing the ongoing reduction operation
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T, typename F>
      irequest iexscan(F f, const T &send_data, T &recv_data) const {
        MPI_Request req;
        MPI_Iexscan(&send_data, &recv_data, 1, detail::datatype_traits<T>::get_datatype(),
                    detail::get_op<T, F>(f).mpi_op, comm_, &req);
        return base_irequest{req};
      }

      /// Performs a partial reduction operation (exclusive scan) over all processes in a
      /// non-blocking manner.
      /// \tparam F type representing the element-wise reduction operation, reduction operation is
      /// performed on data of type \c T
      /// \tparam T type of input and output data of the reduction operation, must meet the
      /// requirements as described in the \verbatim embed:rst:inline :doc:`data_types` \endverbatim
      /// section
      /// \param f reduction operation
      /// \param send_data input buffer for the reduction operation
      /// \param recv_data will hold the results of the reduction operation
      /// \param l memory layouts of the data to send and to receive
      /// \return request representing the ongoing reduction operation
      /// \note This is a collective operation and must be called (possibly by utilizing another
      /// overload) by all processes in the communicator.
      template<typename T, typename F>
      irequest iexscan(F f, const T *send_data, T *recv_data,
                       const contiguous_layout<T> &l) const {
        MPI_Request req;
        MPI_Iexscan(send_data, recv_data, l.size(), detail::datatype_traits<T>::get_datatype(),
                    detail::get_op<T, F>(f).mpi_op, comm_, &req);
        return base_irequest{req};
      }
    };
  }  // namespace impl

  //--------------------------------------------------------------------

  /// Specifies the communication context for a communication operation.
  class communicator : public impl::base_communicator {
    using base = impl::base_communicator;

  protected:
    explicit communicator(MPI_Comm comm) : base{comm} {
    }

  public:
    /// Creates an empty communicator with no associated process.
    communicator() = default;

    /// Creates a new communicator which is equivalent to an existing one.
    /// \param other the other communicator to copy from
    /// \note This is a collective operation that needs to be carried out by all processes of
    /// the communicator \c other. Communicators should not be copied unless a new independent
    /// communicator is wanted. Communicators should be passed via references to functions to
    /// avoid unnecessary copying.
    communicator(const communicator &other) : base{} {
      MPI_Comm_dup(other.comm_, &comm_);
    }

    /// Move-constructs a communicator.
    /// \param other the other communicator to move from
    communicator(communicator &&other) noexcept : base{other.comm_} {
      other.comm_ = MPI_COMM_NULL;
    }

    /// Specifies the process order when merging the local and the remote groups of an
    /// inter-communicator into a communicator.
    enum class merge_order_type {
      /// when merging the local and the remote groups of an inter-communicator put processes
      /// of this group before processes that belong to the other group
      order_low,
      /// when merging the local and the remote groups of an inter-communicator put processes
      /// of this group after processes that belong to the other group
      order_high
    };

    /// indicates that when merging the local and the remote groups of an inter-communicator put
    /// processes of this group before processes that belong to the other group
    static constexpr merge_order_type order_low = merge_order_type::order_low;
    /// indicates that when merging the local and the remote groups of an inter-communicator put
    /// processes of this group after processes that belong to the other group
    static constexpr merge_order_type order_high = merge_order_type::order_high;

    /// Creates a new communicator by merging the local and the remote groups of an
    /// inter-communicator.
    /// \param other the inter-communicator to merge
    /// \param order affects the process ordering in the new communicator
    /// \note This is a collective operation that needs to be carried out by all processes of
    /// the local and the remote groups of  the inter-communicator \c other. The order parameter
    /// must be the same for all process within in the local group as well as within the remote
    /// group.  It should differ for both groups.
    explicit communicator(const inter_communicator &other, merge_order_type order);

    /// Constructs a new communicator from an existing one with a specified communication
    /// group.
    /// \param comm_collective tag to indicate the mode of construction
    /// \param other the communicator
    /// \param gr the group that determines the new communicator's structure
    /// \note This is a collective operation that needs to be carried out by all processes of
    /// the communicator \c other.
    explicit communicator([[maybe_unused]] comm_collective_tag comm_collective,
                          const communicator &other, const group &gr) {
      MPI_Comm_create(other.comm_, gr.gr_, &comm_);
    }

    /// Constructs a new communicator from an existing one with a specified communication
    /// group.
    /// \param group_collective tag to indicate the mode of construction
    /// \param other the communicator
    /// \param gr the group that determines the new communicator's structure
    /// \param t tag to distinguish between different parallel operations in different threads
    /// \note This is a collective operation that needs to be carried out by all processes of
    /// the given group.
    explicit communicator([[maybe_unused]] group_collective_tag group_collective,
                          const communicator &other, const group &gr, tag_t t = tag_t(0)) {
      MPI_Comm_create_group(other.comm_, gr.gr_, static_cast<int>(t), &comm_);
    }

    /// Constructs a new communicator from an existing one with a specified communication
    /// group.
    /// \tparam color_type color type, must be integral type
    /// \tparam key_type key type, must be integral type
    /// \param split tag to indicate the mode of construction
    /// \param other the communicator
    /// \param color control of subset assignment
    /// \param key control of rank assignment
    /// \note This is a collective operation that needs to be carried out by all processes of
    /// the communicator \c other.
    template<typename color_type, typename key_type = int>
    explicit communicator([[maybe_unused]] split_tag split, const communicator &other,
                          color_type color, key_type key = 0) {
      static_assert(detail::is_valid_color_v<color_type>,
                    "not an enumeration type or underlying enumeration type too large");
      static_assert(detail::is_valid_key_v<key_type>,
                    "not an enumeration type or underlying enumeration type too large");
      MPI_Comm_split(other.comm_, detail::underlying_type<color_type>::value(color),
                     detail::underlying_type<key_type>::value(key), &comm_);
    }

    /// Constructs a new communicator from an existing one by spitting the communicator
    /// into disjoint subgroups each of which can create a shared memory region.
    /// \tparam color_type color type, must be integral type
    /// \param split_shared_memory tag to indicate the mode of construction
    /// \param other the communicator
    /// \param key control of rank assignment
    /// \note This is a collective operation that needs to be carried out by all processes of
    /// the communicator \c other.
    template<typename key_type = int>
    explicit communicator([[maybe_unused]] split_shared_memory_tag split_shared_memory,
                          const communicator &other, key_type key = 0) {
      static_assert(detail::is_valid_tag_v<key_type>,
                    "not an enumeration type or underlying enumeration type too large");
      MPI_Comm_split_type(other.comm_, MPI_COMM_TYPE_SHARED,
                          detail::underlying_type<key_type>::value(key), MPI_INFO_NULL, &comm_);
    }

    /// Copy-assigns and creates a new communicator which is equivalent to an existing
    /// one.
    /// \param other the other communicator to copy from
    /// \return this communicator
    /// \note This is a collective operation that needs to be carried out by all processes of
    /// the communicator \c other. Communicators should not be copied unless a new independent
    /// communicator is wanted. Communicators should be passed via references to functions to
    /// avoid unnecessary copying.
    communicator &operator=(const communicator &other) noexcept {
      if (this != &other)
        base::operator=(other);
      return *this;
    }

    /// Move-assigns a communicator.
    /// \param other the other communicator to move from
    /// \return this communicator
    /// \note This is a collective operation that needs to be carried out by all processes of
    /// the communicator \c other.
    communicator &operator=(communicator &&other) noexcept {
      if (this != &other)
        base::operator=(static_cast<base &&>(other));
      return *this;
    }

    /// Determines the total number of processes in a communicator.
    /// \return number of processes
    [[nodiscard]] int size() const {
      return base::size();
    }

    /// Determines the rank within a communicator.
    /// \return the rank of the calling process in the communicator
    [[nodiscard]] int rank() const {
      return base::rank();
    }

    /// Updates the hints of the communicator.
    /// \param i info object with new hints
    void info(const mpl::info &i) const {
      base::info(i);
    }

    /// Get the the hints of the communicator.
    /// \return hints of the communicator
    [[nodiscard]] mpl::info info() const {
      return base::info();
    }

    /// Tests for identity of communicators.
    /// \param other communicator to compare with
    /// \return true if identical
    bool operator==(const communicator &other) const {
      return base::operator==(other);
    }

    /// Tests for identity of communicators.
    /// \param other communicator to compare with
    /// \return true if not identical
    bool operator!=(const communicator &other) const {
      return base::operator!=(other);
    }

    /// Equality types for communicator comparison.
    enum class equality_type {
      /// communicators are identical, i.e., communicators represent the same communication
      /// context
      identical = MPI_IDENT,
      /// communicators are identical, i.e., communicators have the same members in same rank
      /// order but a different context
      congruent = MPI_CONGRUENT,
      /// communicators are similar, i.e., communicators have same the members in different rank
      /// order
      similar = MPI_SIMILAR,
      /// communicators are unequal, i.e., communicators have different sets of members
      unequal = MPI_UNEQUAL
    };

    /// indicates that communicators are identical, i.e., communicators represent the same
    /// communication context
    static constexpr equality_type identical = equality_type::identical;
    /// indicates that communicators are identical, i.e., communicators have same the members in
    /// same rank order but different context
    static constexpr equality_type congruent = equality_type::congruent;
    /// indicates that communicators are similar, i.e., communicators have same tha members in
    /// different rank order
    static constexpr equality_type similar = equality_type::similar;
    /// indicates that communicators are unequal, i.e., communicators have different sets of
    /// members
    static constexpr equality_type unequal = equality_type::unequal;

    /// Compares to another communicator.
    /// \param other communicator to compare with
    /// \return equality type
    [[nodiscard]] equality_type compare(const communicator &other) const {
      int result;
      MPI_Comm_compare(comm_, other.comm_, &result);
      return static_cast<equality_type>(result);
    }

    // === all-to-all ===
    // === each rank sends a single value to each rank
    using base::alltoall;
    using base::ialltoall;

    // --- blocking all-to-all, in place ---
    /// Sends messages to all processes and receives messages from all processes,
    /// in-place version.
    /// \tparam T type of the data to send, must meet the requirements as described in the
    /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
    /// \param sendrecv_data pointer to continuous storage for outgoing messages and for incoming
    /// messages
    /// \details Each process in the communicator sends one element of type \c T to each process
    /// (including itself) and receives one element of type \c T from each process.  The i-th
    /// element in the array \c sendrecv_data is sent to the i-th process.  When the function
    /// has finished, the i-th element in the array \c sendrecv_data was received from the i-th
    /// process.
    /// \note This is a collective operation and must be called (possibly by utilizing another
    /// overload) by all processes in the communicator.
    template<typename T>
    void alltoall(T *sendrecv_data) const {
      MPI_Alltoall(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, sendrecv_data, 1,
                   detail::datatype_traits<T>::get_datatype(), comm_);
    }

    /// Sends messages to all processes and receives messages from all processes,
    /// in-place version.
    /// \tparam T type of the data to send, must meet the requirements as described in the
    /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
    /// \param sendrecv_data pointer to continuous storage for outgoing messages and for incoming
    /// messages
    /// \param sendrecvl memory layouts of the data to send and to receive
    /// \details Each process in the communicator sends elements of type \c T to each process
    /// (including itself) and receives elements of type \c T from each process. The memory
    /// layouts of the incoming and the outgoing messages are described by \c sendrecvl.
    /// The i-th memory block with the layout \c sendrecvl in the array \c sendrecv_data
    /// is sent to the i-th process.  When the function has finished, the i-th memory block with
    /// the layout \c sendrecvl in the array \c sendrecv_data was received from the i-th
    /// process.
    /// \note This is a collective operation and must be called (possibly by utilizing another
    /// overload) by all processes in the communicator.
    template<typename T>
    void alltoall(T *sendrecv_data, const layout<T> &sendrecvl) const {
      MPI_Alltoall(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, sendrecv_data, 1,
                   detail::datatype_traits<layout<T>>::get_datatype(sendrecvl), comm_);
    }

    // --- non-blocking all-to-all, in place ---
    /// Sends messages to all processes and receives messages from all processes in a
    /// non-blocking manner, in-place version.
    /// \tparam T type of the data to send, must meet the requirements as described in the
    /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
    /// \param sendrecv_data pointer to continuous storage for outgoing messages and for incoming
    /// messages
    /// \return request representing the ongoing message transfer
    /// \details Each process in the communicator sends one element of type \c T to each process
    /// (including itself) and receives one element of type \c T from each process.  The i-th
    /// element in the array \c sendrecv_data is sent to the i-th process.  When the message
    /// transfer has finished, the i-th element in the array \c sendrecv_data was received from
    /// the i-th process.
    /// \note This is a collective operation and must be called (possibly by utilizing another
    /// overload) by all processes in the communicator.
    template<typename T>
    irequest ialltoall(T *sendrecv_data) const {
      MPI_Request req;
      MPI_Ialltoall(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, sendrecv_data, 1,
                    detail::datatype_traits<T>::get_datatype(), comm_, &req);
      return impl::base_irequest{req};
    }

    /// Sends messages to all processes and receives messages from all processes in a
    /// non-blocking manner, in-place version.
    /// \tparam T type of the data to send, must meet the requirements as described in the
    /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
    /// \param sendrecv_data pointer to continuous storage for outgoing messages and for incoming
    /// messages
    /// \param sendrecvl memory layouts of the data to send and to receive
    /// \return request representing the ongoing message transfer
    /// \details Each process in the communicator sends elements of type \c T to each process
    /// (including itself) and receives elements of type \c T from each process. The memory
    /// layouts of the incoming and the outgoing messages are described by \c sendrecvl.
    /// The i-th memory block with the layout \c sendrecvl in the array \c sendrecv_data
    /// is sent to the i-th process.  When the message transfer has finished, the i-th memory
    /// block with the layout \c sendrecvl in the array \c sendrecv_data was received from the
    /// i-th process.
    /// \note This is a collective operation and must be called (possibly by utilizing another
    /// overload) by all processes in the communicator.
    template<typename T>
    irequest ialltoall(T *sendrecv_data, const layout<T> &sendrecvl) const {
      MPI_Request req;
      MPI_Ialltoall(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, sendrecv_data, 1,
                    detail::datatype_traits<layout<T>>::get_datatype(sendrecvl), comm_, &req);
      return impl::base_irequest{req};
    }

    // === each rank sends a varying number of values to each rank with possibly different
    // layouts
    using base::alltoallv;
    using base::ialltoallv;

    // --- blocking all-to-all, in place ---
    /// Sends messages with a variable amount of data to all processes and receives
    /// messages with a variable amount of data from all processes, in-place variant.
    /// \tparam T type of the data to send, must meet the requirements as described in the
    /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
    /// \param sendrecv_data pointer to continuous storage for outgoing and incoming messages
    /// \param sendrecvls memory layouts of the data to send and to receive
    /// \param sendrecvdispls displacements of the data to send and to receive
    /// \details Each process in the communicator sends elements of type \c T to each process
    /// (including itself) and receives elements of type \c T from each process.  Send- and
    /// receive-data are stored in consecutive blocks of variable size in the buffer
    /// \c sendecvdata. The i-th memory block with the layout <tt>sendlrecvs[i]</tt> in the
    /// array \c sendrecv_data starts <tt>sendrecvdispls[i]</tt> bytes after the address given
    /// in \c sendrecv_data. The i-th memory block is sent to the i-th process. When the
    /// function has finished, the i-th memory block in the array \c sendrecv_data was received
    /// from the i-th process.
    /// \note This is a collective operation and must be called (possibly by utilizing another
    /// overload) by all processes in the communicator.
    template<typename T>
    void alltoallv(T *sendrecv_data, const layouts<T> &sendrecvls,
                   const displacements &sendrecvdispls) const {
      check_size(sendrecvdispls);
      check_size(sendrecvls);
      const std::vector<int> counts(sendrecvls.size(), 1);
      const std::vector<int> sendrecvdispls_int(sendrecvdispls.begin(), sendrecvdispls.end());
      MPI_Alltoallw(MPI_IN_PLACE, 0, 0, 0, sendrecv_data, counts.data(),
                    sendrecvdispls_int.data(),
                    reinterpret_cast<const MPI_Datatype *>(sendrecvls()), comm_);
    }

    /// Sends messages with a variable amount of data to all processes and receives
    /// messages with a variable amount of data from all processes, in-place variant.
    /// \tparam T type of the data to send, must meet the requirements as described in the
    /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
    /// \param sendrecv_data pointer to continuous storage for incoming and outgoing messages
    /// \param sendrecvls memory layouts of the data to send and to receive
    /// \details Each process in the communicator sends elements of type \c T to each process
    /// (including itself) and receives elements of type \c T from each process.  Send- and
    /// receive-data are stored in consecutive blocks of variable size in the buffer
    /// \c sendrecv_data. The i-th memory block with the layout <tt>sendrecvls[i]</tt> in the
    /// array \c sendrecv_data starts at the address given in \c sendrecv_data. The i-th memory
    /// block is sent to the i-th process. Note that the memory layouts need to include
    /// appropriate holes at the beginning in order to avoid overlapping send-receive blocks.
    /// When the function has finished, the i-th memory block in the array \c sendrecv_data was
    /// received from the i-th process.
    /// \note This is a collective operation and must be called (possibly by utilizing another
    /// overload) by all processes in the communicator.
    template<typename T>
    void alltoallv(T *sendrecv_data, const layouts<T> &sendrecvls) const {
      alltoallv(sendrecv_data, sendrecvls, displacements(size()));
    }

    // --- non-blocking all-to-all, in place ---
    /// Sends messages with a variable amount of data to all processes and receives
    /// messages with a variable amount of data from all processes in a non-blocking manner,
    /// in-place variant.
    /// \tparam T type of the data to send, must meet the requirements as described in the
    /// \verbatim embed:rst:inline :doc:`data_types` \endverbatim section
    /// \param sendrecv_data pointer to continuous storage for outgoing and incoming messages
    /// \param sendrecvls memory layouts of the data to send and to receive
    /// \param sendrecvdispls displacements of the data to send and to receive
    /// \return request representing the ongoing message transfer
    /// \details Each process in the communicator sends elements of type \c T to each process
    /// (including itself) and receives elements of type \c T from each process.  Send- and
    /// receive-data are stored in consecutive blocks of variable size in the buffer
    /// \c sendecvdata. The i-th memory block with the layout <tt>sendlrecvs[i]</tt> in the
    /// array \c sendrecv_data starts <tt>sendrecvdispls[i]</tt> bytes after the address given
    /// in sendrecv_data. The i-th memory block is sent to the i-th process. When the function
    /// has finished, the i-th memory block in the array \c sendrecv_data was received from the
    /// i-th process.
    /// \note This is a collective operation and must be called (possibly by utilizing another
    /// overload) by all processes in the communicator.
    template<typename T>
    irequest ialltoallv(T *sendrecv_data, const layouts<T> &sendrecvls,
                        const displacements &sendrecvdispls) const {
      check_size(sendrecvdispls);
      check_size(sendrecvls);
      ialltoallv_state<T> *state{new ialltoallv_state<T>(
          sendrecvls, std::vector<int>(sendrecvls.size(), 1),
          std::vector<int>(sendrecvdispls.begin(), sendrecvdispls.end()))};
      MPI_Request req;
      MPI_Grequest_start(ialltoallv_query<T>, ialltoallv_free<T>, ialltoallv_cancel, state,
                         &req);
      state->req = req;
      std::thread thread([this, sendrecv_data, state]() {
        ialltoallv_task(static_cast<T *>(nullptr), sendrecv_data, state);
      });
      thread.detach();
      return impl::base_irequest{req};
    }

    /// Sends messages with a variable amount of data to all processes and receives
    /// messages with a variable amount of data from all processes in a non-blocking manner,
    /// in-place variant.
    /// \tparam T type of the data to send, must meet the requirements as described in the
    /// \param sendrecv_data pointer to continuous storage for incoming and outgoing messages
    /// \param sendrecvls memory layouts of the data to send and to receive
    /// \return request representing the ongoing message transfer
    /// \details Each process in the communicator sends elements of type \c T to each process
    /// (including itself) and receives elements of type \c T from each process.  Send- and
    /// receive-data are stored in consecutive blocks of variable size in the buffer
    /// \c sendrecv_data. The i-th memory block with the layout <tt>sendrecvls[i]</tt> in the
    /// array \c sendrecv_data starts at the address given in \c sendrecv_data. The i-th memory
    /// block is sent to the i-th process. Note that the memory layouts need to include
    /// appropriate holes at the beginning in order to avoid overlapping send-receive blocks.
    /// When the function has finished, the i-th memory block in the array \c sendrecv_data was
    /// received from the i-th process.
    /// \note This is a collective operation and must be called (possibly by utilizing another
    /// overload) by all processes in the communicator.
    template<typename T>
    irequest ialltoallv(T *sendrecv_data, const layouts<T> &sendrecvls) const {
      return ialltoallv(sendrecv_data, sendrecvls, displacements(size()));
    }

    // === reduce ===
    using base::reduce;
    using base::ireduce;

    // --- blocking reduce, in place ---
    /// Performs a reduction operation over all processes, in-place variant.
    /// \tparam F type representing the reduction operation, reduction operation is performed
    /// on data of type \c T
    /// \tparam T type of input and output data of the reduction operation, must meet the
    /// requirements as described in the \verbatim embed:rst:inline :doc:`data_types` \endverbatim  section
    /// \param f reduction operation
    /// \param root_rank rank of the process that will receive the reduction result
    /// \param sendrecv_data input data for the reduction operation, will hold the result of the
    /// reduction operation if rank equals root_rank
    /// \note This is a collective operation and must be called (possibly by utilizing another
    /// overload) by all processes in the communicator.
    template<typename T, typename F>
    void reduce(F f, int root_rank, T &sendrecv_data) const {
      check_root(root_rank);
      if (rank() == root_rank)
        MPI_Reduce(MPI_IN_PLACE, &sendrecv_data, 1, detail::datatype_traits<T>::get_datatype(),
                   detail::get_op<T, F>(f).mpi_op, root_rank, comm_);
      else
        MPI_Reduce(&sendrecv_data, nullptr, 1, detail::datatype_traits<T>::get_datatype(),
                   detail::get_op<T, F>(f).mpi_op, root_rank, comm_);
    }

    /// Performs a reduction operation over all processes, non-root in-place variant.
    /// \tparam F type representing the reduction operation, reduction operation is performed
    /// on data of type \c T
    /// \tparam T type of input and output data of the reduction operation, must meet the
    /// requirements as described in the \verbatim embed:rst:inline :doc:`data_types` \endverbatim
    /// section
    /// \param f reduction operation
    /// \param root_rank rank of the process that will receive the reduction result, must be
    /// different from the rank of the calling process
    /// \param send_data input data for the reduction operation
    /// \note This is a collective operation and must be called (possibly by utilizing another
    /// overload) by all processes in the communicator.
    template<typename T, typename F>
    void reduce(F f, int root_rank, const T &send_data) const {
      check_nonroot(root_rank);
      MPI_Reduce(&send_data, nullptr, 1, detail::datatype_traits<T>::get_datatype(),
                 detail::get_op<T, F>(f).mpi_op, root_rank, comm_);
    }

    /// Performs a reduction operation over all processes, in-place variant.
    /// \tparam F type representing the element-wise reduction operation, reduction operation is
    /// performed on data of type \c T
    /// \tparam T type of input and output data of the reduction operation, must meet the
    /// requirements as described in the \verbatim embed:rst:inline :doc:`data_types` \endverbatim
    /// section
    /// \param f reduction operation
    /// \param root_rank rank of the process that will receive the reduction result
    /// \param sendrecv_data input buffer for the reduction operation, will hold the results of
    /// the reduction operation if rank equals \c root_rank
    /// \param l memory layouts of the data to send and to receive
    /// \note This is a collective operation and must be called (possibly by utilizing another
    /// overload) by all processes in the communicator.
    template<typename T, typename F>
    void reduce(F f, int root_rank, T *sendrecv_data, const contiguous_layout<T> &l) const {
      if (rank() == root_rank)
        MPI_Reduce(MPI_IN_PLACE, sendrecv_data, l.size(),
                   detail::datatype_traits<T>::get_datatype(), detail::get_op<T, F>(f).mpi_op,
                   root_rank, comm_);
      else
        MPI_Reduce(sendrecv_data, nullptr, l.size(), detail::datatype_traits<T>::get_datatype(),
                   detail::get_op<T, F>(f).mpi_op, root_rank, comm_);
    }

    /// Performs a reduction operation over all processes, non-root in-place variant.
    /// \tparam F type representing the element-wise reduction operation, reduction operation is
    /// performed on data of type \c T
    /// \tparam T type of input and output data of the reduction operation, must meet the
    /// requirements as described in the \verbatim embed:rst:inline :doc:`data_types` \endverbatim
    /// section
    /// \param f reduction operation
    /// \param root_rank rank of the process that will receive the reduction result, must be
    /// different from the rank of the calling process
    /// \param send_data input buffer for the reduction operation
    /// \param l memory layouts of the data to send and to receive
    /// \note This is a collective operation and must be called (possibly by utilizing another
    /// overload) by all processes in the communicator.
    template<typename T, typename F>
    void reduce(F f, int root_rank, const T *send_data, const contiguous_layout<T> &l) const {
      check_nonroot(root_rank);
      MPI_Reduce(send_data, nullptr, l.size(), detail::datatype_traits<T>::get_datatype(),
                 detail::get_op<T, F>(f).mpi_op, root_rank, comm_);
    }

    // --- non-blocking reduce, in place ---
    /// Performs a reduction operation over all processes in a non-blocking manner,
    /// in-place variant.
    /// \tparam F type representing the reduction operation, reduction operation is performed
    /// on data of type \c T
    /// \tparam T type of input and output data of the reduction operation, must meet the
    /// requirements as described in the \verbatim embed:rst:inline :doc:`data_types` \endverbatim
    /// section
    /// \param f reduction operation
    /// \param root_rank rank of the process that will receive the reduction result
    /// \param sendrecv_data input data for the reduction operation, will hold the result of the
    /// reduction operation if rank equals \c root_rank
    /// \return request representing the ongoing reduction operation
    /// \note This is a collective operation and must be called (possibly by utilizing another
    /// overload) by all processes in the communicator.
    template<typename T, typename F>
    irequest ireduce(F f, int root_rank, T &sendrecv_data) const {
      check_root(root_rank);
      MPI_Request req;
      if (rank() == root_rank)
        MPI_Ireduce(MPI_IN_PLACE, &sendrecv_data, 1, detail::datatype_traits<T>::get_datatype(),
                    detail::get_op<T, F>(f).mpi_op, root_rank, comm_, &req);
      else
        MPI_Ireduce(&sendrecv_data, nullptr, 1, detail::datatype_traits<T>::get_datatype(),
                    detail::get_op<T, F>(f).mpi_op, root_rank, comm_, &req);
      return impl::base_irequest{req};
    }

    /// Performs a reduction operation over all processes in a non-blocking manner,
    /// non-root in-place variant.
    /// \tparam F type representing the reduction operation, reduction operation is performed
    /// on data of type \c T
    /// \tparam T type of input and output data of the reduction operation, must meet the
    /// requirements as described in the \verbatim embed:rst:inline :doc:`data_types` \endverbatim  section
    /// \param f reduction operation
    /// \param root_rank rank of the process that will receive the reduction result, must be
    /// different from the rank of the calling process
    /// \param send_data input data for the reduction operation
    /// \return request representing the ongoing reduction operation
    /// \note This is a collective operation and must be called (possibly by utilizing another
    /// overload) by all processes in the communicator.
    template<typename T, typename F>
    irequest ireduce(F f, int root_rank, const T &send_data) const {
      check_nonroot(root_rank);
      MPI_Request req;
      MPI_Ireduce(&send_data, nullptr, 1, detail::datatype_traits<T>::get_datatype(),
                  detail::get_op<T, F>(f).mpi_op, root_rank, comm_, &req);
      return impl::base_irequest{req};
    }

    /// Performs a reduction operation over all processes in non-blocking manner,
    /// in-place variant.
    /// \tparam F type representing the element-wise reduction operation, reduction operation is
    /// performed on data of type \c T
    /// \tparam T type of input and output data of the reduction operation, must meet the
    /// requirements as described in the \verbatim embed:rst:inline :doc:`data_types` \endverbatim
    /// section
    /// \param f reduction operation
    /// \param root_rank rank of the process that will receive the reduction result
    /// \param sendrecv_data input buffer for the reduction operation, will hold the results of
    /// the reduction operation if rank equals \c root_rank
    /// \param l memory layouts of the data to send and to receive
    /// \return request representing the ongoing reduction operation
    /// \note This is a collective operation and must be called (possibly by utilizing another
    /// overload) by all processes in the communicator.
    template<typename T, typename F>
    irequest ireduce(F f, int root_rank, T *sendrecv_data,
                     const contiguous_layout<T> &l) const {
      check_root(root_rank);
      MPI_Request req;
      if (rank() == root_rank)
        MPI_Ireduce(MPI_IN_PLACE, sendrecv_data, l.size(),
                    detail::datatype_traits<T>::get_datatype(), detail::get_op<T, F>(f).mpi_op,
                    root_rank, comm_, &req);
      else
        MPI_Ireduce(sendrecv_data, nullptr, l.size(),
                    detail::datatype_traits<T>::get_datatype(), detail::get_op<T, F>(f).mpi_op,
                    root_rank, comm_, &req);
      return impl::base_irequest{req};
    }

    /// Performs a reduction operation over all processes in a non-blocking manner,
    /// non-root in-place variant.
    /// \tparam F type representing the element-wise reduction operation, reduction operation is
    /// performed on data of type \c T
    /// \tparam T type of input and output data of the reduction operation, must meet the
    /// requirements as described in the \verbatim embed:rst:inline :doc:`data_types` \endverbatim
    /// section
    /// \param f reduction operation
    /// \param root_rank rank of the process that will receive the reduction result, must be
    /// different from the rank of the calling process
    /// \param send_data input buffer for the reduction operation
    /// \param l memory layouts of the data to send and to receive
    /// \return request representing the ongoing reduction operation
    /// \note This is a collective operation and must be called (possibly by utilizing another
    /// overload) by all processes in the communicator.
    template<typename T, typename F>
    irequest ireduce(F f, int root_rank, const T *send_data,
                     const contiguous_layout<T> &l) const {
      check_nonroot(root_rank);
      MPI_Request req;
      MPI_Ireduce(send_data, nullptr, l.size(), detail::datatype_traits<T>::get_datatype(),
                  detail::get_op<T, F>(f).mpi_op, root_rank, comm_, &req);
      return impl::base_irequest{req};
    }

    // === all-reduce ===
    using base::allreduce;
    using base::iallreduce;

    // --- blocking all-reduce, in place ---
    /// Performs a reduction operation over all processes and broadcasts the result,
    /// in-place variant.
    /// \tparam F type representing the reduction operation, reduction operation is performed
    /// on data of type \c T
    /// \tparam T type of input and output data of the reduction operation, must meet the
    /// requirements as described in the \verbatim embed:rst:inline :doc:`data_types` \endverbatim
    /// section
    /// \param f reduction operation
    /// \param sendrecv_data input data for the reduction operation
    /// \note This is a collective operation and must be called (possibly by utilizing another
    /// overload) by all processes in the communicator.
    template<typename T, typename F>
    void allreduce(F f, T &sendrecv_data) const {
      MPI_Allreduce(MPI_IN_PLACE, &sendrecv_data, 1, detail::datatype_traits<T>::get_datatype(),
                    detail::get_op<T, F>(f).mpi_op, comm_);
    }

    /// Performs a reduction operation over all processes and broadcasts the result,
    /// in-place variant.
    /// \tparam F type representing the element-wise reduction operation, reduction operation is
    /// performed on data of type \c T
    /// \tparam T type of input and output data of the reduction operation, must meet the
    /// requirements as described in the \verbatim embed:rst:inline :doc:`data_types` \endverbatim
    /// section
    /// \param f reduction operation
    /// \param sendrecv_data input buffer for the reduction operation
    /// \param l memory layouts of the data to send and to receive
    /// \note This is a collective operation and must be called (possibly by utilizing another
    /// overload) by all processes in the communicator.
    template<typename T, typename F>
    void allreduce(F f, T *sendrecv_data, const contiguous_layout<T> &l) const {
      MPI_Allreduce(MPI_IN_PLACE, sendrecv_data, l.size(),
                    detail::datatype_traits<T>::get_datatype(), detail::get_op<T, F>(f).mpi_op,
                    comm_);
    }

    // --- non-blocking all-reduce, in place ---
    /// Performs a reduction operation over all processes and broadcasts the result in a
    /// non-blocking manner, in-place variant.
    /// \tparam F type representing the reduction operation, reduction operation is performed
    /// on data of type \c T
    /// \tparam T type of input and output data of the reduction operation, must meet the
    /// requirements as described in the \verbatim embed:rst:inline :doc:`data_types` \endverbatim
    /// section
    /// \param f reduction operation
    /// \param sendrecv_data input data for the reduction operation
    /// \return request representing the ongoing reduction operation
    /// \note This is a collective operation and must be called (possibly by utilizing another
    /// overload) by all processes in the communicator.
    template<typename T, typename F>
    irequest iallreduce(F f, T &sendrecv_data) const {
      MPI_Request req;
      MPI_Iallreduce(MPI_IN_PLACE, &sendrecv_data, 1,
                     detail::datatype_traits<T>::get_datatype(), detail::get_op<T, F>(f).mpi_op,
                     comm_, &req);
      return impl::base_irequest{req};
    }

    /// Performs a reduction operation over all processes and broadcasts the result in
    /// non-blocking manner, in-place variant.
    /// \tparam F type representing the element-wise reduction operation, reduction operation is
    /// performed on data of type \c T
    /// \tparam T type of input and output data of the reduction operation, must meet the
    /// requirements as described in the \verbatim embed:rst:inline :doc:`data_types` \endverbatim
    /// section
    /// \param f reduction operation
    /// \param sendrecv_data input buffer for the reduction operation
    /// \param l memory layouts of the data to send and to receive
    /// \return request representing the ongoing reduction operation
    /// \note This is a collective operation and must be called (possibly by utilizing another
    /// overload) by all processes in the communicator.
    template<typename T, typename F>
    irequest iallreduce(F f, T *sendrecv_data, const contiguous_layout<T> &l) const {
      MPI_Request req;
      MPI_Iallreduce(MPI_IN_PLACE, sendrecv_data, l.size(),
                     detail::datatype_traits<T>::get_datatype(), detail::get_op<T, F>(f).mpi_op,
                     comm_, &req);
      return impl::base_irequest{req};
    }

    // === scan ===
    using base::scan;
    using base::iscan;

    // --- blocking scan, in place ---
    /// Performs a partial reduction operation (scan) over all processes, in-place
    /// variant.
    /// \tparam F type representing the reduction operation, reduction operation is performed
    /// on data of type \c T
    /// \tparam T type of input and output data of the reduction operation, must meet the
    /// requirements as described in the \verbatim embed:rst:inline :doc:`data_types` \endverbatim
    /// section
    /// \param f reduction operation
    /// \param sendrecv_data input data for the reduction operation
    /// \note This is a collective operation and must be called (possibly by utilizing another
    /// overload) by all processes in the communicator.
    template<typename T, typename F>
    void scan(F f, T &sendrecv_data) const {
      MPI_Scan(MPI_IN_PLACE, &sendrecv_data, 1, detail::datatype_traits<T>::get_datatype(),
               detail::get_op<T, F>(f).mpi_op, comm_);
    }

    /// Performs a partial reduction operation (scan) over all processes, in-place
    /// variant.
    /// \tparam F type representing the element-wise reduction operation, reduction operation is
    /// performed on data of type \c T
    /// \tparam T type of input and output data of the reduction operation, must meet the
    /// requirements as described in the \verbatim embed:rst:inline :doc:`data_types` \endverbatim
    /// section
    /// \param f reduction operation
    /// \param sendrecv_data input buffer for the reduction operation
    /// \param l memory layouts of the data to send and to receive
    /// \note This is a collective operation and must be called (possibly by utilizing another
    /// overload) by all processes in the communicator.
    template<typename T, typename F>
    void scan(F f, T *sendrecv_data, const contiguous_layout<T> &l) const {
      MPI_Scan(MPI_IN_PLACE, sendrecv_data, l.size(),
               detail::datatype_traits<T>::get_datatype(), detail::get_op<T, F>(f).mpi_op,
               comm_);
    }

    // --- non-blocking scan, in place ---
    /// Performs a partial reduction operation (scan) over all processes in a
    /// non-blocking manner, in-place variant.
    /// \tparam F type representing the reduction operation, reduction operation is performed
    /// on data of type \c T
    /// \tparam T type of input and output data of the reduction operation, must meet the
    /// requirements as described in the \verbatim embed:rst:inline :doc:`data_types` \endverbatim
    /// section
    /// \param f reduction operation
    /// \param sendrecv_data input data for the reduction operation
    /// \return request representing the ongoing reduction operation
    /// \note This is a collective operation and must be called (possibly by utilizing another
    /// overload) by all processes in the communicator.
    template<typename T, typename F>
    irequest iscan(F f, T &sendrecv_data) const {
      MPI_Request req;
      MPI_Iscan(MPI_IN_PLACE, &sendrecv_data, 1, detail::datatype_traits<T>::get_datatype(),
                detail::get_op<T, F>(f).mpi_op, comm_, &req);
      return impl::base_irequest{req};
    }

    /// Performs a partial reduction (scan) operation over all processes in a
    /// non-blocking manner, in-place variant.
    /// \tparam F type representing the element-wise reduction operation, reduction operation is
    /// performed on data of type \c T
    /// \tparam T type of input and output data of the reduction operation, must meet the
    /// requirements as described in the \verbatim embed:rst:inline :doc:`data_types` \endverbatim
    /// section
    /// \param f reduction operation
    /// \param sendrecv_data input buffer for the reduction operation
    /// \param l memory layouts of the data to send and to receive
    /// \return request representing the ongoing reduction operation
    /// \note This is a collective operation and must be called (possibly by utilizing another
    /// overload) by all processes in the communicator.
    template<typename T, typename F>
    irequest iscan(F f, T *sendrecv_data, const contiguous_layout<T> &l) const {
      MPI_Request req;
      MPI_Iscan(MPI_IN_PLACE, sendrecv_data, l.size(),
                detail::datatype_traits<T>::get_datatype(), detail::get_op<T, F>(f).mpi_op,
                comm_, &req);
      return impl::base_irequest{req};
    }

    // === exscan ===
    using base::exscan;
    using base::iexscan;

    // --- blocking exscan, in place ---
    /// Performs a partial reduction operation (exclusive scan) over all processes,
    /// in-place variant.
    /// \tparam F type representing the reduction operation, reduction operation is performed
    /// on data of type \c T
    /// \tparam T type of input and output data of the reduction operation, must meet the
    /// requirements as described in the \verbatim embed:rst:inline :doc:`data_types` \endverbatim
    /// section
    /// \param f reduction operation
    /// \param sendrecv_data input data for the reduction operation
    /// \note This is a collective operation and must be called (possibly by utilizing another
    /// overload) by all processes in the communicator.
    template<typename T, typename F>
    void exscan(F f, T &sendrecv_data) const {
      MPI_Exscan(MPI_IN_PLACE, &sendrecv_data, 1, detail::datatype_traits<T>::get_datatype(),
                 detail::get_op<T, F>(f).mpi_op, comm_);
    }

    /// Performs a partial reduction operation (exclusive scan) over all processes,
    /// in-place variant.
    /// \tparam F type representing the element-wise reduction operation, reduction operation is
    /// performed on data of type \c T
    /// \tparam T type of input and output data of the reduction operation, must meet the
    /// requirements as described in the \verbatim embed:rst:inline :doc:`data_types` \endverbatim
    /// section
    /// \param f reduction operation
    /// \param sendrecv_data input buffer for the reduction operation
    /// \param l memory layouts of the data to send and to receive
    /// \note This is a collective operation and must be called (possibly by utilizing another
    /// overload) by all processes in the communicator.
    template<typename T, typename F>
    void exscan(F f, T *sendrecv_data, const contiguous_layout<T> &l) const {
      MPI_Exscan(MPI_IN_PLACE, sendrecv_data, l.size(),
                 detail::datatype_traits<T>::get_datatype(), detail::get_op<T, F>(f).mpi_op,
                 comm_);
    }

    // --- non-blocking exscan, in place ---
    /// Performs a partial reduction operation (exclusive scan) over all processes in a
    /// non-blocking manner, in-place variant.
    /// \tparam F type representing the reduction operation, reduction operation is performed
    /// on data of type \c T
    /// \tparam T type of input and output data of the reduction operation, must meet the
    /// requirements as described in the \verbatim embed:rst:inline :doc:`data_types` \endverbatim
    /// section
    /// \param f reduction operation
    /// \param sendrecv_data input data for the reduction operation
    /// \return request representing the ongoing reduction operation
    /// \note This is a collective operation and must be called (possibly by utilizing another
    /// overload) by all processes in the communicator.
    template<typename T, typename F>
    irequest iexscan(F f, T &sendrecv_data) const {
      MPI_Request req;
      MPI_Iexscan(MPI_IN_PLACE, &sendrecv_data, 1, detail::datatype_traits<T>::get_datatype(),
                  detail::get_op<T, F>(f).mpi_op, comm_, &req);
      return impl::base_irequest{req};
    }

    /// Performs a partial reduction operation (exclusive scan) over all processes in a
    /// non-blocking manner, in-place variant.
    /// \tparam F type representing the element-wise reduction operation, reduction operation is
    /// performed on data of type \c T
    /// \tparam T type of input and output data of the reduction operation, must meet the
    /// requirements as described in the \verbatim embed:rst:inline :doc:`data_types` \endverbatim
    /// section
    /// \param f reduction operation
    /// \param sendrecv_data input buffer for the reduction operation
    /// \param l memory layouts of the data to send and to receive
    /// \return request representing the ongoing reduction operation
    /// \note This is a collective operation and must be called (possibly by utilizing another
    /// overload) by all processes in the communicator.
    template<typename T, typename F>
    irequest iexscan(F f, T *sendrecv_data, const contiguous_layout<T> &l) const {
      MPI_Request req;
      MPI_Iexscan(MPI_IN_PLACE, sendrecv_data, l.size(),
                  detail::datatype_traits<T>::get_datatype(), detail::get_op<T, F>(f).mpi_op,
                  comm_, &req);
      return impl::base_irequest{req};
    }

    /// Spawns new processes and establishes communication.
    /// \param root_rank the root process, following arguments are ignored on non-root ranks
    /// \param max_procs number of processes to span
    /// \param command command and command-line options to the processes that are spawned
    /// \return inter-communicator that establishes a communication channel between the
    /// processes of this communicator and the new spawned processes
    /// \note This is a collective operation and must be called (possibly by utilizing another
    /// overload) by all processes in the communicator.
    inter_communicator spawn(int root_rank, int max_procs, const command_line &command) const;

    /// Spawns new processes and establishes communication.
    /// \param root_rank the root process, following arguments are ignored on non-root ranks
    /// \param max_procs number of processes to span
    /// \param command command and command-line options to the processes that are spawned
    /// \param i info object telling the underlying MPI runtime how to spawn the new processes
    /// \return inter-communicator that establishes a communication channel between the
    /// processes of this communicator and the new spawned processes
    /// \note This is a collective operation and must be called (possibly by utilizing another
    /// overload) by all processes in the communicator.
    inter_communicator spawn(int root_rank, int max_procs, const command_line &command,
                             const mpl::info &i) const;

    /// Spawns new processes and establishes communication, non-root variant.
    /// \param root_rank the root process
    /// \return inter-communicator that establishes a communication channel between the
    /// processes of this communicator and the new spawned processes
    /// \note This is a collective operation and must be called (possibly by utilizing another
    /// overload) by all processes in the communicator.
    inter_communicator spawn(int root_rank) const;

    /// Spawns new processes and establishes communication.
    /// \param root_rank the root process, following arguments are ignored on non-root ranks
    /// \param commands command and command-line options to the processes that are spawned
    /// \return inter-communicator that establishes a communication channel between the
    /// processes of this communicator and the new spawned processes
    /// \note This is a collective operation and must be called (possibly by utilizing another
    /// overload) by all processes in the communicator.
    inter_communicator spawn_multiple(int root_rank, const command_lines &commands) const;

    /// Spawns new processes and establishes communication.
    /// \param root_rank the root process, following arguments are ignored on non-root ranks
    /// \param commands command and command-line options to the processes that are spawned
    /// \param i list of info object telling the underlying MPI runtime how to spawn the new
    /// processes
    /// \return inter-communicator that establishes a communication channel between the
    /// processes of this communicator and the new spawned processes
    /// \note This is a collective operation and must be called (possibly by utilizing another
    /// overload) by all processes in the communicator.
    inter_communicator spawn_multiple(int root_rank, const command_lines &commands,
                                      const mpl::infos &i) const;

    /// Spawns new processes and establishes communication, non-root variant.
    /// \param root_rank the root process
    /// \return inter-communicator that establishes a communication channel between the
    /// processes of this communicator and the new spawned processes
    /// \note This is a collective operation and must be called (possibly by utilizing another
    /// overload) by all processes in the communicator.
    inter_communicator spawn_multiple(int root_rank) const;

    friend class group;

    friend class cartesian_communicator;

    friend class graph_communicator;

    friend class distributed_graph_communicator;

    friend class inter_communicator;

    friend class environment::detail::env;

    friend class file;
  };

  //--------------------------------------------------------------------

  /// Specifies the communication context for a communication operation between two
  /// non-overlapping groups.
  class inter_communicator : public impl::base_communicator {
    using base = impl::base_communicator;

    explicit inter_communicator(MPI_Comm comm) : base{comm} {
    }

  public:
    /// Creates a new inter-communicator from two existing communicators.
    /// \param local_communicator communicator that contains the local group of the new
    /// inter-communicator
    /// \param local_leader rank of the local group leader within the communicator
    /// local_communicator
    /// \param peer_communicator peer communicator to which the local and the remote
    /// leaders belong
    /// \param remote_leader rank of the remote group leader within the peer communicator
    /// \param t tag associated to this operation
    /// \note It is a collective operation over the union of the processes with the local
    /// communicator and the peer communicator.
    explicit inter_communicator(const communicator &local_communicator, int local_leader,
                                const communicator &peer_communicator, int remote_leader,
                                tag_t t = tag_t(0))
        : base{} {
      MPI_Intercomm_create(local_communicator.comm_, local_leader, peer_communicator.comm_,
                           remote_leader, static_cast<int>(t), &comm_);
    }

    /// Creates a new inter-communicator which is equivalent to an existing one.
    /// \param other the other inter-communicator to copy from
    /// \note This is a collective operation that needs to be carried out by all local and
    /// remote processes of the inter-communicator \c other.  Inter-communicators should not be
    /// copied unless a new independent communicator is wanted.  Inter-Communicators should be
    /// passed via references to functions to avoid unnecessary copying.
    inter_communicator(const inter_communicator &other) : base{} {
      MPI_Comm_dup(other.comm_, &comm_);
    }

    /// Move-constructs an inter-communicator.
    /// \param other the other inter-communicator to move from
    inter_communicator(inter_communicator &&other) noexcept : base{other.comm_} {
      other.comm_ = MPI_COMM_NULL;
    }

    /// Get the parent inter-communicator of the current process, which is created when the
    /// process was spawned.
    /// \return inter-communicator that establishes a communication channel between the
    /// spawning process group and the new spawned processes
    static const inter_communicator &parent() {
      static auto get_parent = []() {
        MPI_Comm comm;
        MPI_Comm_get_parent(&comm);
        return comm;
      };
      static inter_communicator s_parent{get_parent()};
      return s_parent;
    }

    /// Copy-assigns and creates a new inter-communicator which is equivalent to an
    /// existing one.
    /// \param other the other inter-communicator to copy from
    /// \return this inter-communicator
    /// \note This is a collective operation that needs to be carried out by all local and
    /// remote processes of the communicator \c other. Inter-communicators should not be copied
    /// unless a new independent inter-communicator is wanted. Inter-communicators should be
    /// passed via references to functions to avoid unnecessary copying.
    inter_communicator &operator=(const inter_communicator &other) noexcept {
      if (this != &other)
        base::operator=(other);
      return *this;
    }

    /// Move-assigns an inter-communicator.
    /// \param other the other inter-communicator to move from
    /// \return this communicator
    /// \note This is a collective operation that needs to be carried out by all processes local
    /// and remote processes of the inter-communicator \c other.
    inter_communicator &operator=(inter_communicator &&other) noexcept {
      if (this != &other)
        base::operator=(static_cast<base &&>(other));
      return *this;
    }

    /// Determines the total number of processes in the local group of an
    /// inter-communicator.
    /// \return number of processes
    [[nodiscard]] int size() const {
      return base::size();
    }

    /// Determines the rank within the local group of an inter-communicator.
    /// \return the rank of the calling process in the inter-communicator
    [[nodiscard]] int rank() const {
      return base::rank();
    }

    /// Determines the total number of processes in the remote group of an
    /// inter-communicator.
    /// \return number of processes
    [[nodiscard]] int remote_size() const {
      int result;
      MPI_Comm_remote_size(comm_, &result);
      return result;
    }

    /// Tests for identity of inter-communicators.
    /// \param other inter-communicator to compare with
    /// \return true if identical
    bool operator==(const communicator &other) const {
      return base::operator==(other);
    }

    /// Tests for identity of inter-communicators.
    /// \param other inter-communicator to compare with
    /// \return true if not identical
    bool operator!=(const communicator &other) const {
      return base::operator!=(other);
    }

    /// Equality types for inter-communicator comparison.
    enum class equality_type {
      /// inter-communicators are identical, i.e., inter-communicators represent the same
      /// inter-communication context with the identical local and remote groups
      identical = MPI_IDENT,
      /// inter-communicators are identical, i.e., inter-communicators have local and remote
      /// groups with the same members in same rank order but a different context
      congruent = MPI_CONGRUENT,
      /// inter-communicators are similar, i.e., inter-communicators have local and remote
      /// groups with the same members in different rank order
      similar = MPI_SIMILAR,
      /// inter-communicators are unequal, i.e., inter-communicators have different local and
      /// remote groups
      unequal = MPI_UNEQUAL
    };

    /// indicates that inter-communicators are identical, i.e., inter-communicators represent
    /// the same inter-communication context with the identical local and remote groups
    static constexpr equality_type identical = equality_type::identical;
    /// indicates that inter-communicators are identical, i.e., inter-communicators have local
    /// and remote groups with the same members in same rank order but a different context
    static constexpr equality_type congruent = equality_type::congruent;
    /// indicates that inter-communicators are similar, i.e., inter-communicators have local
    /// and remote groups with the same members in different rank order
    static constexpr equality_type similar = equality_type::similar;
    /// inter-communicators are unequal, i.e., inter-communicators have different local and
    /// remote groups
    static constexpr equality_type unequal = equality_type::unequal;

    /// Compares to another inter-communicator.
    /// \param other inter-communicator to compare with
    /// \return equality type
    [[nodiscard]] equality_type compare(const communicator &other) const {
      int result;
      MPI_Comm_compare(comm_, other.comm_, &result);
      return static_cast<equality_type>(result);
    }

    friend class group;
    friend class communicator;
  };

  //--------------------------------------------------------------------

  inline communicator::communicator(const inter_communicator &other, merge_order_type order)
      : base{} {
    const int high{order == merge_order_type::order_high};
    MPI_Intercomm_merge(other.comm_, high, &comm_);
  }

  inline inter_communicator communicator::spawn(int root_rank, int max_procs,
                                                const command_line &command) const {
    check_root(root_rank);
    MPI_Comm comm;
    if (root_rank == rank()) {
#if defined MPL_DEBUG
      if (command.size() < 1)
        throw invalid_argument();
#endif
      // performing some deep copies in order to avoid const_cast
      std::vector<std::vector<char>> args;
      args.reserve(command.size() - 1);
      for (command_line::size_type i{1}; i < command.size(); ++i)
        args.push_back(std::vector<char>(command[i].begin(), command[i].end()));
      std::vector<char *> args_pointers;
      args_pointers.reserve(args.size() + 1);
      for (auto &arg : args)
        args_pointers.push_back(arg.data());
      args_pointers.push_back(nullptr);
      MPI_Comm_spawn(command[0].c_str(), args_pointers.data(), max_procs, MPI_INFO_NULL,
                     root_rank, comm_, &comm, MPI_ERRCODES_IGNORE);
    } else
      MPI_Comm_spawn(nullptr, MPI_ARGV_NULL, 0, MPI_INFO_NULL, root_rank, comm_, &comm,
                     MPI_ERRCODES_IGNORE);
    return inter_communicator{comm};
  }

  inline inter_communicator communicator::spawn(int root_rank, int max_procs,
                                                const command_line &command,
                                                const mpl::info &i) const {
    check_root(root_rank);
    MPI_Comm comm;
    if (root_rank == rank()) {
#if defined MPL_DEBUG
      if (command.size() < 1)
        throw invalid_argument();
#endif
      // performing some deep copies in order to avoid const_cast
      std::vector<std::vector<char>> args;
      args.reserve(command.size() - 1);
      for (command_line::size_type i{1}; i < command.size(); ++i)
        args.push_back(std::vector<char>(command[i].begin(), command[i].end()));
      std::vector<char *> args_pointers;
      args_pointers.reserve(args.size() + 1);
      for (auto &arg : args)
        args_pointers.push_back(arg.data());
      args_pointers.push_back(nullptr);
      MPI_Comm_spawn(command[0].c_str(), args_pointers.data(), max_procs, i.info_, root_rank,
                     comm_, &comm, MPI_ERRCODES_IGNORE);
    } else
      MPI_Comm_spawn(nullptr, MPI_ARGV_NULL, 0, MPI_INFO_NULL, root_rank, comm_, &comm,
                     MPI_ERRCODES_IGNORE);
    return inter_communicator{comm};
  }

  inline inter_communicator communicator::spawn(int root_rank) const {
    check_nonroot(root_rank);
    MPI_Comm comm;
    MPI_Comm_spawn(nullptr, MPI_ARGV_NULL, 0, MPI_INFO_NULL, root_rank, comm_, &comm,
                   MPI_ERRCODES_IGNORE);
    return inter_communicator{comm};
  }

  inline inter_communicator communicator::spawn_multiple(int root_rank,
                                                         const command_lines &commands) const {
    check_root(root_rank);
    MPI_Comm comm;
    if (root_rank == rank()) {
      int count{0};
      std::vector<std::vector<char>> vector_of_commands;
      std::vector<char *> vector_of_commands_ptr;
      std::vector<std::vector<std::vector<char>>> vector_of_args;
      std::vector<std::vector<char *>> vector_of_args_ptr;
      std::vector<char **> vector_of_args_ptr_ptr;
      std::vector<int> vector_of_maxprocs;
      std::vector<MPI_Info> vector_of_info;
      vector_of_commands.reserve(commands.size());
      vector_of_commands_ptr.reserve(commands.size());
      vector_of_args.reserve(commands.size());
      vector_of_args_ptr.reserve(commands.size());
      vector_of_args_ptr_ptr.reserve(commands.size());
      vector_of_maxprocs.reserve(commands.size());
      for (const auto &command : commands) {
        ++count;
#if defined MPL_DEBUG
        if (command.size() < 1)
          throw invalid_argument();
#endif
        vector_of_commands.push_back(std::vector<char>(command[0].begin(), command[0].end()));
        vector_of_commands_ptr.push_back(vector_of_commands.back().data());
        {
          std::vector<std::vector<char>> args;
          args.reserve(command.size() - 1);
          for (command_line::size_type i{1}; i < command.size(); ++i)
            args.push_back(std::vector<char>(command[i].begin(), command[i].end()));
          vector_of_args.push_back(std::move(args));
        }
        std::vector<char *> args_pointers;
        args_pointers.reserve(vector_of_args.back().size() + 1);
        for (auto &arg : vector_of_args.back())
          args_pointers.push_back(arg.data());
        args_pointers.push_back(nullptr);
        vector_of_args_ptr.push_back(std::move(args_pointers));
        vector_of_args_ptr_ptr.push_back(vector_of_args_ptr.back().data());
        vector_of_maxprocs.push_back(1);
        vector_of_info.push_back(MPI_INFO_NULL);
      }
      MPI_Comm_spawn_multiple(count, vector_of_commands_ptr.data(),
                              vector_of_args_ptr_ptr.data(), vector_of_maxprocs.data(),
                              vector_of_info.data(), root_rank, comm_, &comm,
                              MPI_ERRCODES_IGNORE);
    } else
      MPI_Comm_spawn_multiple(0, nullptr, MPI_ARGVS_NULL, nullptr, nullptr, root_rank, comm_,
                              &comm, MPI_ERRCODES_IGNORE);
    return inter_communicator{comm};
  }

  inline inter_communicator communicator::spawn_multiple(int root_rank,
                                                         const command_lines &commands,
                                                         const mpl::infos &i) const {
    check_root(root_rank);
#if defined MPL_DEBUG
    if (commands.size() != i.size())
      throw invalid_argument();
#endif
    MPI_Comm comm;
    if (root_rank == rank()) {
      int count{0};
      std::vector<std::vector<char>> vector_of_commands;
      std::vector<char *> vector_of_commands_ptr;
      std::vector<std::vector<std::vector<char>>> vector_of_args;
      std::vector<std::vector<char *>> vector_of_args_ptr;
      std::vector<char **> vector_of_args_ptr_ptr;
      std::vector<int> vector_of_maxprocs;
      std::vector<MPI_Info> vector_of_info;
      vector_of_commands.reserve(commands.size());
      vector_of_commands_ptr.reserve(commands.size());
      vector_of_args.reserve(commands.size());
      vector_of_args_ptr.reserve(commands.size());
      vector_of_args_ptr_ptr.reserve(commands.size());
      vector_of_maxprocs.reserve(commands.size());
      for (const auto &command : commands) {
        ++count;
#if defined MPL_DEBUG
        if (command.size() < 1)
          throw invalid_argument();
#endif
        vector_of_commands.push_back(std::vector<char>(command[0].begin(), command[0].end()));
        vector_of_commands_ptr.push_back(vector_of_commands.back().data());
        {
          std::vector<std::vector<char>> args;
          args.reserve(command.size() - 1);
          for (command_line::size_type i{1}; i < command.size(); ++i)
            args.push_back(std::vector<char>(command[i].begin(), command[i].end()));
          vector_of_args.push_back(std::move(args));
        }
        std::vector<char *> args_pointers;
        args_pointers.reserve(vector_of_args.back().size() + 1);
        for (auto &arg : vector_of_args.back())
          args_pointers.push_back(arg.data());
        args_pointers.push_back(nullptr);
        vector_of_args_ptr.push_back(std::move(args_pointers));
        vector_of_args_ptr_ptr.push_back(vector_of_args_ptr.back().data());
        vector_of_maxprocs.push_back(1);
      }
      for (const auto &info : i)
        vector_of_info.push_back(info.info_);
      MPI_Comm_spawn_multiple(count, vector_of_commands_ptr.data(),
                              vector_of_args_ptr_ptr.data(), vector_of_maxprocs.data(),
                              vector_of_info.data(), root_rank, comm_, &comm,
                              MPI_ERRCODES_IGNORE);
    } else
      MPI_Comm_spawn_multiple(0, nullptr, MPI_ARGVS_NULL, nullptr, nullptr, root_rank, comm_,
                              &comm, MPI_ERRCODES_IGNORE);
    return inter_communicator{comm};
  }

  inline inter_communicator communicator::spawn_multiple(int root_rank) const {
    check_nonroot(root_rank);
    MPI_Comm comm;
    MPI_Comm_spawn_multiple(0, nullptr, MPI_ARGVS_NULL, nullptr, nullptr, root_rank, comm_,
                            &comm, MPI_ERRCODES_IGNORE);
    return inter_communicator{comm};
  }

  //--------------------------------------------------------------------

  inline group::group(const group &other) {
    MPI_Group_excl(other.gr_, 0, nullptr, &gr_);
  }

  inline group::group(const communicator &comm) {
    MPI_Comm_group(comm.comm_, &gr_);
  }

  inline group::group(const inter_communicator &comm) {
    MPI_Comm_group(comm.comm_, &gr_);
  }

  inline group::group(group::Union_tag, const group &other_1, const group &other_2) {
    MPI_Group_union(other_1.gr_, other_2.gr_, &gr_);
  }

  inline group::group(group::intersection_tag, const group &other_1, const group &other_2) {
    MPI_Group_intersection(other_1.gr_, other_2.gr_, &gr_);
  }

  inline group::group(group::difference_tag, const group &other_1, const group &other_2) {
    MPI_Group_difference(other_1.gr_, other_2.gr_, &gr_);
  }

  inline group::group(group::include_tag, const group &other, const ranks &rank) {
    MPI_Group_incl(other.gr_, rank.size(), rank(), &gr_);
  }

  inline group::group(group::exclude_tag, const group &other, const ranks &rank) {
    MPI_Group_excl(other.gr_, rank.size(), rank(), &gr_);
  }

}  // namespace mpl

#endif
