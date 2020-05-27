#if !(defined MPL_COMM_GROUP_HPP)

#define MPL_COMM_GROUP_HPP

#include <mpi.h>
#include <type_traits>
#include <tuple>
#include <thread>
#include <mpl/layout.hpp>
#include <mpl/vector.hpp>

namespace mpl {

  class group;

  class communicator;

  class cart_communicator;

  class graph_communicator;

  namespace environment {

    namespace detail {

      class env;

    }

  }  // namespace environment

  //--------------------------------------------------------------------

  class group {
    MPI_Group gr{MPI_GROUP_EMPTY};

  public:
    enum class equality_type {
      ident = MPI_IDENT,
      similar = MPI_SIMILAR,
      unequal = MPI_UNEQUAL
    };

    class Union {};

    class intersection {};

    class difference {};

    class incl {};

    class excl {};

    group() = default;

    explicit group(const communicator &comm);  // define later
    group(group &&other) noexcept {
      gr = other.gr;
      other.gr = MPI_GROUP_EMPTY;
    }

    explicit group(Union, const group &other_1, const group &other_2);         // define later
    explicit group(intersection, const group &other_1, const group &other_2);  // define later
    explicit group(difference, const group &other_1, const group &other_2);    // define later
    explicit group(incl, const group &other, const ranks &rank);               // define later
    explicit group(excl, const group &other, const ranks &rank);               // define later
    ~group() {
      int result;
      MPI_Group_compare(gr, MPI_GROUP_EMPTY, &result);
      if (result != MPI_IDENT)
        MPI_Group_free(&gr);
    }

    void operator=(const group &) = delete;

    group &operator=(group &&other) noexcept {
      if (this != &other) {
        int result;
        MPI_Group_compare(gr, MPI_GROUP_EMPTY, &result);
        if (result != MPI_IDENT)
          MPI_Group_free(&gr);
        gr = other.gr;
        other.gr = MPI_GROUP_EMPTY;
      }
      return *this;
    }

    int size() const {
      int result;
      MPI_Group_size(gr, &result);
      return result;
    }

    int rank() const {
      int result;
      MPI_Group_rank(gr, &result);
      return result;
    }

    int translate(int rank, const group &other) const {
      int other_rank;
      MPI_Group_translate_ranks(gr, 1, &rank, other.gr, &other_rank);
      return other_rank;
    }

    ranks translate(const ranks &rank, const group &other) const {
      ranks other_rank;
      MPI_Group_translate_ranks(gr, static_cast<int>(rank.size()), rank(), other.gr,
                                other_rank());
      return other_rank;
    }

    bool operator==(const group &other) const {
      int result;
      MPI_Group_compare(gr, other.gr, &result);
      return result == MPI_IDENT;
    }

    bool operator!=(const group &other) const {
      int result;
      MPI_Group_compare(gr, other.gr, &result);
      return result != MPI_IDENT;
    }

    equality_type compare(const group &other) const {
      int result;
      MPI_Group_compare(gr, other.gr, &result);
      return static_cast<equality_type>(result);
    }

    friend class communicator;
  };

  //--------------------------------------------------------------------

  class communicator {
    struct isend_irecv_state {
      MPI_Request req{};
      int source{MPI_ANY_SOURCE};
      int tag{MPI_ANY_TAG};
      MPI_Datatype datatype{MPI_DATATYPE_NULL};
      int count{MPI_UNDEFINED};
    };

    static int isend_irecv_query(void *state, MPI_Status *s) {
      isend_irecv_state *sendrecv_state = reinterpret_cast<isend_irecv_state *>(state);
      MPI_Status_set_elements(s, sendrecv_state->datatype, sendrecv_state->count);
      MPI_Status_set_cancelled(s, 0);
      s->MPI_SOURCE = sendrecv_state->source;
      s->MPI_TAG = sendrecv_state->tag;
      return MPI_SUCCESS;
    }

    static int isend_irecv_free(void *state) {
      isend_irecv_state *sendrecv_state = reinterpret_cast<isend_irecv_state *>(state);
      delete sendrecv_state;
      return MPI_SUCCESS;
    }

    static int isend_irecv_cancel(void *state, int complete) { return MPI_SUCCESS; }

  protected:
    MPI_Comm comm{MPI_COMM_NULL};

  public:
    enum class equality_type {
      ident = MPI_IDENT,
      congruent = MPI_CONGRUENT,
      similar = MPI_SIMILAR,
      unequal = MPI_UNEQUAL
    };

    static constexpr equality_type ident = equality_type::ident;
    static constexpr equality_type congruent = equality_type::congruent;
    static constexpr equality_type similar = equality_type::similar;
    static constexpr equality_type unequal = equality_type::unequal;

    class comm_collective {};

    class group_collective {};

    class split {};

    class split_shared {};

  private:
    void check_dest(int dest) const {
#if defined MPL_DEBUG
      if (dest != proc_null and (dest < 0 or dest >= size()))
        throw invalid_rank();
#endif
    }

    void check_source(int source) const {
#if defined MPL_DEBUG
      if (source != proc_null and source != any_source and (source < 0 or source >= size()))
        throw invalid_rank();
#endif
    }

    void check_send_tag(tag t) const {
#if defined MPL_DEBUG
      if (static_cast<int>(t) < 0 or static_cast<int>(t) > static_cast<int>(tag::up()))
        throw invalid_tag();
#endif
    }

    void check_recv_tag(tag t) const {
#if defined MPL_DEBUG
      if (static_cast<int>(t) != static_cast<int>(tag::any()) and
          (static_cast<int>(t) < 0 or static_cast<int>(t) > static_cast<int>(tag::up())))
        throw invalid_tag();
#endif
    }

    void check_root(int root) const {
#if defined MPL_DEBUG
      if (root < 0 or root >= size())
        throw invalid_rank();
#endif
    }

    void check_nonroot(int root) const {
#if defined MPL_DEBUG
      if (root < 0 or root >= size() or root == rank())
        throw invalid_rank();
#endif
    }

    template<typename T>
    void check_size(const layouts<T> &l) const {
#if defined MPL_DEBUG
      if (static_cast<int>(l.size()) > size())
        throw invalid_size();
#endif
    }

    void check_size(const displacements &d) const {
#if defined MPL_DEBUG
      if (static_cast<int>(d.size()) > size())
        throw invalid_size();
#endif
    }

    void check_count(int count) const {
#if defined MPL_DEBUG
      if (count == MPI_UNDEFINED)
        throw invalid_count();
#endif
    }

    template<typename T>
    void check_container_size(const T &container, detail::basic_or_fixed_size_type) const {}

    template<typename T>
    void check_container_size(const T &container, detail::stl_container) const {
#if defined MPL_DEBUG
      if (container.size() > std::numeric_limits<int>::max())
        throw invalid_count();
#endif
    }

    template<typename T>
    void check_container_size(const T &container) const {
      check_container_size(container, typename datatype_traits<T>::data_type_category{});
    }

  protected:
    explicit communicator(MPI_Comm comm) : comm(comm) {}

  public:
    communicator() = default;

    communicator(const communicator &other) { MPI_Comm_dup(other.comm, &comm); }

    communicator(communicator &&other) noexcept {
      comm = other.comm;
      other.comm = MPI_COMM_NULL;
    }

    explicit communicator(comm_collective, const communicator &other, const group &gr) {
      MPI_Comm_create(other.comm, gr.gr, &comm);
    }

    explicit communicator(group_collective, const communicator &other, const group &gr,
                          tag t = tag(0)) {
      MPI_Comm_create_group(other.comm, gr.gr, static_cast<int>(t), &comm);
    }

    template<typename color_type, typename key_type = int>
    explicit communicator(split, const communicator &other, color_type color,
                          key_type key = 0) {
      static_assert(detail::is_valid_color<color_type>::value,
                    "not an enumeration type or underlying enumeration type too large");
      static_assert(detail::is_valid_key<key_type>::value,
                    "not an enumeration type or underlying enumeration type too large");
      MPI_Comm_split(other.comm, detail::underlying_type<color_type>::value(color),
                     detail::underlying_type<key_type>::value(key), &comm);
    }

    template<typename key_type = int>
    explicit communicator(split_shared, const communicator &other, key_type key = 0) {
      static_assert(detail::is_valid_tag<key_type>::value,
                    "not an enumeration type or underlying enumeration type too large");
      MPI_Comm_split_type(other.comm, MPI_COMM_TYPE_SHARED,
                          detail::underlying_type<key_type>::value(key), MPI_INFO_NULL, &comm);
    }

    ~communicator() {
      if (is_valid()) {
        int result1, result2;
        MPI_Comm_compare(comm, MPI_COMM_WORLD, &result1);
        MPI_Comm_compare(comm, MPI_COMM_SELF, &result2);
        if (result1 != MPI_IDENT and result2 != MPI_IDENT)
          MPI_Comm_free(&comm);
      }
    }

    void operator=(const communicator &) = delete;

    communicator &operator=(communicator &&other) noexcept {
      if (this != &other) {
        if (is_valid()) {
          int result1, result2;
          MPI_Comm_compare(comm, MPI_COMM_WORLD, &result1);
          MPI_Comm_compare(comm, MPI_COMM_SELF, &result2);
          if (result1 != MPI_IDENT and result2 != MPI_IDENT)
            MPI_Comm_free(&comm);
        }
        comm = other.comm;
        other.comm = MPI_COMM_NULL;
      }
      return *this;
    }

    int size() const {
      int result;
      MPI_Comm_size(comm, &result);
      return result;
    }

    int rank() const {
      int result;
      MPI_Comm_rank(comm, &result);
      return result;
    }

    bool operator==(const communicator &other) const {
      int result;
      MPI_Comm_compare(comm, other.comm, &result);
      return result == MPI_IDENT;
    }

    bool operator!=(const communicator &other) const {
      int result;
      MPI_Comm_compare(comm, other.comm, &result);
      return result != MPI_IDENT;
    }

    equality_type compare(const communicator &other) const {
      int result;
      MPI_Comm_compare(comm, other.comm, &result);
      return static_cast<equality_type>(result);
    }

    bool is_valid() const { return comm != MPI_COMM_NULL; }

    void abort(int err) const { MPI_Abort(comm, err); }

    friend class group;

    friend class cart_communicator;

    friend class graph_communicator;

    friend class dist_graph_communicator;

    friend class environment::detail::env;

    // === point to point ==============================================

    // === standard send ===
    // --- blocking standard send ---
  private:
    template<typename T>
    void send(const T &data, int dest, tag t, detail::basic_or_fixed_size_type) const {
      MPI_Send(&data, 1, datatype_traits<T>::get_datatype(), dest, static_cast<int>(t), comm);
    }

    template<typename T>
    void send(const T &data, int dest, tag t, detail::contigous_const_stl_container) const {
      using value_type = typename T::value_type;
      vector_layout<value_type> l(data.size());
      send(data.size() > 0 ? &data[0] : nullptr, l, dest, t);
    }

    template<typename T>
    void send(const T &data, int dest, tag t, detail::stl_container) const {
      using value_type =
          typename detail::remove_const_from_members<typename T::value_type>::type;
      detail::vector<value_type> serial_data(data.size(), std::begin(data));
      vector_layout<value_type> l(serial_data.size());
      send(serial_data.data(), l, dest, t);
    }

  public:
    template<typename T>
    void send(const T &data, int dest, tag t = tag(0)) const {
      check_dest(dest);
      check_send_tag(t);
      check_container_size(data);
      send(data, dest, t, typename datatype_traits<T>::data_type_category{});
    }

    template<typename T>
    void send(const T *data, const layout<T> &l, int dest, tag t = tag(0)) const {
      check_dest(dest);
      check_send_tag(t);
      MPI_Send(data, 1, datatype_traits<layout<T>>::get_datatype(l), dest, static_cast<int>(t),
               comm);
    }

    template<typename iterT>
    void send(iterT begin, iterT end, int dest, tag t = tag(0)) const {
      using value_type = typename std::iterator_traits<iterT>::value_type;
      if (detail::is_contiguous_iterator<iterT>::value) {
        vector_layout<value_type> l(std::distance(begin, end));
        send(&(*begin), l, dest, t);
      } else {
        iterator_layout<value_type> l(begin, end);
        send(&(*begin), l, dest, t);
      }
    }

    // --- nonblocking standard send ---
  private:
    template<typename T>
    irequest isend(const T &data, int dest, tag t, detail::basic_or_fixed_size_type) const {
      MPI_Request req;
      MPI_Isend(&data, 1, datatype_traits<T>::get_datatype(), dest, static_cast<int>(t), comm,
                &req);
      return detail::irequest(req);
    }

    template<typename T>
    void isend(const T &data, int dest, tag t, isend_irecv_state *isend_state,
               detail::contigous_const_stl_container) const {
      using value_type = typename T::value_type;
      const int count(data.size());
      MPI_Datatype datatype{datatype_traits<value_type>::get_datatype()};
      MPI_Request req;
      MPI_Isend(data.size() > 0 ? &data[0] : nullptr, count, datatype, dest,
                static_cast<int>(t), comm, &req);
      MPI_Status s;
      MPI_Wait(&req, &s);
      isend_state->source = s.MPI_SOURCE;
      isend_state->tag = s.MPI_TAG;
      isend_state->datatype = datatype;
      isend_state->count = 0;
      MPI_Grequest_complete(isend_state->req);
    }

    template<typename T>
    void isend(const T &data, int dest, tag t, isend_irecv_state *isend_state,
               detail::stl_container) const {
      using value_type =
          typename detail::remove_const_from_members<typename T::value_type>::type;
      detail::vector<value_type> serial_data(data.size(), std::begin(data));
      isend(serial_data, dest, t, isend_state, detail::contigous_const_stl_container{});
    }

    template<typename T, typename C>
    irequest isend(const T &data, int dest, tag t, C) const {
      isend_irecv_state *send_state = new isend_irecv_state();
      MPI_Request req;
      MPI_Grequest_start(isend_irecv_query, isend_irecv_free, isend_irecv_cancel, send_state,
                         &req);
      send_state->req = req;
      std::thread thread(
          [this, &data, dest, t, send_state]() { isend(data, dest, t, send_state, C{}); });
      thread.detach();
      return detail::irequest(req);
    }

  public:
    template<typename T>
    irequest isend(const T &data, int dest, tag t = tag(0)) const {
      check_dest(dest);
      check_send_tag(t);
      check_container_size(data);
      return isend(data, dest, t, typename datatype_traits<T>::data_type_category{});
    }

    template<typename T>
    irequest isend(const T *data, const layout<T> &l, int dest, tag t = tag(0)) const {
      check_dest(dest);
      check_send_tag(t);
      MPI_Request req;
      MPI_Isend(data, 1, datatype_traits<layout<T>>::get_datatype(l), dest, static_cast<int>(t),
                comm, &req);
      return detail::irequest(req);
    }

    template<typename iterT>
    irequest isend(iterT begin, iterT end, int dest, tag t = tag(0)) const {
      using value_type = typename std::iterator_traits<iterT>::value_type;
      if (detail::is_contiguous_iterator<iterT>::value) {
        vector_layout<value_type> l(std::distance(begin, end));
        return isend(&(*begin), l, dest, t);
      } else {
        iterator_layout<value_type> l(begin, end);
        return isend(&(*begin), l, dest, t);
      }
    }

    // --- persistend standard send ---
    template<typename T>
    prequest send_init(const T &data, int dest, tag t = tag(0)) const {
      check_dest(dest);
      check_send_tag(t);
      MPI_Request req;
      MPI_Send_init(&data, 1, datatype_traits<T>::get_datatype(), dest, static_cast<int>(t),
                    comm, &req);
      return prequest(req);
    }

    template<typename T>
    prequest send_init(const T *data, const layout<T> &l, int dest, tag t = tag(0)) const {
      check_dest(dest);
      check_send_tag(t);
      MPI_Request req;
      MPI_Send_init(data, 1, datatype_traits<layout<T>>::get_datatype(l), dest,
                    static_cast<int>(t), comm, &req);
      return prequest(req);
    }

    template<typename iterT>
    prequest send_init(iterT begin, iterT end, int dest, tag t = tag(0)) const {
      using value_type = typename std::iterator_traits<iterT>::value_type;
      if (detail::is_contiguous_iterator<iterT>::value) {
        vector_layout<value_type> l(std::distance(begin, end));
        return send_init(&(*begin), l, dest, t);
      } else {
        iterator_layout<value_type> l(begin, end);
        return send_init(&(*begin), l, dest, t);
      }
    }

    // === buffered send ===
    // --- determine buffer size ---
    template<typename T>
    int bsend_size() const {
      int size;
      MPI_Pack_size(1, datatype_traits<T>::get_datatype(), comm, &size);
      return size + MPI_BSEND_OVERHEAD;
    }

    template<typename T>
    int bsend_size(const layout<T> &l) const {
      int size;
      MPI_Pack_size(1, datatype_traits<layout<T>>::get_datatype(l), comm, &size);
      return size + MPI_BSEND_OVERHEAD;
    }

    // --- blocking buffered send ---
  private:
    template<typename T>
    void bsend(const T &data, int dest, tag t, detail::basic_or_fixed_size_type) const {
      MPI_Bsend(&data, 1, datatype_traits<T>::get_datatype(), dest, static_cast<int>(t), comm);
    }

    template<typename T>
    void bsend(const T &data, int dest, tag t, detail::contigous_const_stl_container) const {
      using value_type = typename T::value_type;
      vector_layout<value_type> l(data.size());
      bsend(data.size() > 0 ? &data[0] : nullptr, l, dest, t);
    }

    template<typename T>
    void bsend(const T &data, int dest, tag t, detail::stl_container) const {
      using value_type =
          typename detail::remove_const_from_members<typename T::value_type>::type;
      detail::vector<value_type> serial_data(data.size(), std::begin(data));
      vector_layout<value_type> l(serial_data.size());
      bsend(serial_data.data(), l, dest, t);
    }

  public:
    template<typename T>
    void bsend(const T &data, int dest, tag t = tag(0)) const {
      check_dest(dest);
      check_send_tag(t);
      check_container_size(data);
      bsend(data, dest, t, typename datatype_traits<T>::data_type_category{});
    }

    template<typename T>
    void bsend(const T *data, const layout<T> &l, int dest, tag t = tag(0)) const {
      check_dest(dest);
      check_send_tag(t);
      MPI_Bsend(data, 1, datatype_traits<layout<T>>::get_datatype(l), dest, static_cast<int>(t),
                comm);
    }

    template<typename iterT>
    void bsend(iterT begin, iterT end, int dest, tag t = tag(0)) const {
      using value_type = typename std::iterator_traits<iterT>::value_type;
      if (detail::is_contiguous_iterator<iterT>::value) {
        vector_layout<value_type> l(std::distance(begin, end));
        bsend(&(*begin), l, dest, t);
      } else {
        iterator_layout<value_type> l(begin, end);
        bsend(&(*begin), l, dest, t);
      }
    }

    // --- nonblocking buffered send ---
  private:
    template<typename T>
    irequest ibsend(const T &data, int dest, tag t, detail::basic_or_fixed_size_type) const {
      MPI_Request req;
      MPI_Ibsend(&data, 1, datatype_traits<T>::get_datatype(), dest, static_cast<int>(t), comm,
                 &req);
      return detail::irequest(req);
    }

    template<typename T>
    void ibsend(const T &data, int dest, tag t, isend_irecv_state *isend_state,
                detail::contigous_const_stl_container) const {
      using value_type = typename T::value_type;
      const int count(data.size());
      MPI_Datatype datatype{datatype_traits<value_type>::get_datatype()};
      MPI_Request req;
      MPI_Ibsend(data.size() > 0 ? &data[0] : nullptr, count, datatype, dest,
                 static_cast<int>(t), comm, &req);
      MPI_Status s;
      MPI_Wait(&req, &s);
      isend_state->source = s.MPI_SOURCE;
      isend_state->tag = s.MPI_TAG;
      isend_state->datatype = datatype;
      isend_state->count = 0;
      MPI_Grequest_complete(isend_state->req);
    }

    template<typename T>
    void ibsend(const T &data, int dest, tag t, isend_irecv_state *isend_state,
                detail::stl_container) const {
      using value_type =
          typename detail::remove_const_from_members<typename T::value_type>::type;
      detail::vector<value_type> serial_data(data.size(), std::begin(data));
      ibsend(serial_data, dest, t, isend_state, detail::contigous_const_stl_container{});
    }

    template<typename T, typename C>
    irequest ibsend(const T &data, int dest, tag t, C) const {
      isend_irecv_state *send_state = new isend_irecv_state();
      MPI_Request req;
      MPI_Grequest_start(isend_irecv_query, isend_irecv_free, isend_irecv_cancel, send_state,
                         &req);
      send_state->req = req;
      std::thread thread(
          [this, &data, dest, t, send_state]() { ibsend(data, dest, t, send_state, C{}); });
      thread.detach();
      return detail::irequest(req);
    }

  public:
    template<typename T>
    irequest ibsend(const T &data, int dest, tag t = tag(0)) const {
      check_dest(dest);
      check_send_tag(t);
      check_container_size(data);
      return ibsend(data, dest, t, typename datatype_traits<T>::data_type_category{});
    }

    template<typename T>
    irequest ibsend(const T *data, const layout<T> &l, int dest, tag t = tag(0)) const {
      check_dest(dest);
      check_send_tag(t);
      MPI_Request req;
      MPI_Ibsend(data, 1, datatype_traits<layout<T>>::get_datatype(l), dest,
                 static_cast<int>(t), comm, &req);
      return detail::irequest(req);
    }

    template<typename iterT>
    irequest ibsend(iterT begin, iterT end, int dest, tag t = tag(0)) const {
      using value_type = typename std::iterator_traits<iterT>::value_type;
      if (detail::is_contiguous_iterator<iterT>::value) {
        vector_layout<value_type> l(std::distance(begin, end));
        return ibsend(&(*begin), l, dest, t);
      } else {
        iterator_layout<value_type> l(begin, end);
        return ibsend(&(*begin), l, dest, t);
      }
    }

    // --- persistent buffered send ---
    template<typename T>
    prequest bsend_init(const T &data, int dest, tag t = tag(0)) const {
      check_dest(dest);
      check_send_tag(t);
      MPI_Request req;
      MPI_Bsend_init(&data, 1, datatype_traits<T>::get_datatype(), dest, static_cast<int>(t),
                     comm, &req);
      return prequest(req);
    }

    template<typename T>
    prequest bsend_init(const T *data, const layout<T> &l, int dest, tag t = tag(0)) const {
      check_dest(dest);
      check_send_tag(t);
      MPI_Request req;
      MPI_Bsend_init(data, 1, datatype_traits<layout<T>>::get_datatype(l), dest,
                     static_cast<int>(t), comm, &req);
      return prequest(req);
    }

    template<typename iterT>
    prequest bsend_init(iterT begin, iterT end, int dest, tag t = tag(0)) const {
      using value_type = typename std::iterator_traits<iterT>::value_type;
      if (detail::is_contiguous_iterator<iterT>::value) {
        vector_layout<value_type> l(std::distance(begin, end));
        return bsend_init(&(*begin), l, dest, t);
      } else {
        iterator_layout<value_type> l(begin, end);
        return bsend_init(&(*begin), l, dest, t);
      }
    }

    // === synchronous send ===
    // --- blocking synchronous send ---
  private:
    template<typename T>
    void ssend(const T &data, int dest, tag t, detail::basic_or_fixed_size_type) const {
      MPI_Ssend(&data, 1, datatype_traits<T>::get_datatype(), dest, static_cast<int>(t), comm);
    }

    template<typename T>
    void ssend(const T &data, int dest, tag t, detail::contigous_const_stl_container) const {
      using value_type = typename T::value_type;
      vector_layout<value_type> l(data.size());
      ssend(data.size() > 0 ? &data[0] : nullptr, l, dest, t);
    }

    template<typename T>
    void ssend(const T &data, int dest, tag t, detail::stl_container) const {
      using value_type =
          typename detail::remove_const_from_members<typename T::value_type>::type;
      detail::vector<value_type> serial_data(data.size(), std::begin(data));
      vector_layout<value_type> l(serial_data.size());
      ssend(serial_data.data(), l, dest, t);
    }

  public:
    template<typename T>
    void ssend(const T &data, int dest, tag t = tag(0)) const {
      check_dest(dest);
      check_send_tag(t);
      ssend(data, dest, t, typename datatype_traits<T>::data_type_category{});
    }

    template<typename T>
    void ssend(const T *data, const layout<T> &l, int dest, tag t = tag(0)) const {
      check_dest(dest);
      check_send_tag(t);
      MPI_Ssend(data, 1, datatype_traits<layout<T>>::get_datatype(l), dest, static_cast<int>(t),
                comm);
    }

    template<typename iterT>
    void ssend(iterT begin, iterT end, int dest, tag t = tag(0)) const {
      using value_type = typename std::iterator_traits<iterT>::value_type;
      if (detail::is_contiguous_iterator<iterT>::value) {
        vector_layout<value_type> l(std::distance(begin, end));
        ssend(&(*begin), l, dest, t);
      } else {
        iterator_layout<value_type> l(begin, end);
        ssend(&(*begin), l, dest, t);
      }
    }

    // --- nonblocking synchronous send ---
  private:
    template<typename T>
    irequest issend(const T &data, int dest, tag t, detail::basic_or_fixed_size_type) const {
      MPI_Request req;
      MPI_Issend(&data, 1, datatype_traits<T>::get_datatype(), dest, static_cast<int>(t), comm,
                 &req);
      return detail::irequest(req);
    }

    template<typename T>
    void issend(const T &data, int dest, tag t, isend_irecv_state *isend_state,
                detail::contigous_const_stl_container) const {
      using value_type = typename T::value_type;
      const int count(data.size());
      MPI_Datatype datatype{datatype_traits<value_type>::get_datatype()};
      MPI_Request req;
      MPI_Issend(data.size() > 0 ? &data[0] : nullptr, count, datatype, dest,
                 static_cast<int>(t), comm, &req);
      MPI_Status s;
      MPI_Wait(&req, &s);
      isend_state->source = s.MPI_SOURCE;
      isend_state->tag = s.MPI_TAG;
      isend_state->datatype = datatype;
      isend_state->count = 0;
      MPI_Grequest_complete(isend_state->req);
    }

    template<typename T>
    void issend(const T &data, int dest, tag t, isend_irecv_state *isend_state,
                detail::stl_container) const {
      using value_type =
          typename detail::remove_const_from_members<typename T::value_type>::type;
      detail::vector<value_type> serial_data(data.size(), std::begin(data));
      issend(serial_data, dest, t, isend_state, detail::contigous_const_stl_container{});
    }

    template<typename T, typename C>
    irequest issend(const T &data, int dest, tag t, C) const {
      isend_irecv_state *send_state = new isend_irecv_state();
      MPI_Request req;
      MPI_Grequest_start(isend_irecv_query, isend_irecv_free, isend_irecv_cancel, send_state,
                         &req);
      send_state->req = req;
      std::thread thread(
          [this, &data, dest, t, send_state]() { issend(data, dest, t, send_state, C{}); });
      thread.detach();
      return detail::irequest(req);
    }

  public:
    template<typename T>
    irequest issend(const T &data, int dest, tag t = tag(0)) const {
      check_dest(dest);
      check_send_tag(t);
      check_container_size(data);
      return issend(data, dest, t, typename datatype_traits<T>::data_type_category{});
    }

    template<typename T>
    irequest issend(const T *data, const layout<T> &l, int dest, tag t = tag(0)) const {
      check_dest(dest);
      check_send_tag(t);
      MPI_Request req;
      MPI_Issend(data, 1, datatype_traits<layout<T>>::get_datatype(l), dest,
                 static_cast<int>(t), comm, &req);
      return detail::irequest(req);
    }

    template<typename iterT>
    irequest issend(iterT begin, iterT end, int dest, tag t = tag(0)) const {
      using value_type = typename std::iterator_traits<iterT>::value_type;
      if (detail::is_contiguous_iterator<iterT>::value) {
        vector_layout<value_type> l(std::distance(begin, end));
        return issend(&(*begin), l, dest, t);
      } else {
        iterator_layout<value_type> l(begin, end);
        return issend(&(*begin), l, dest, t);
      }
    }

    // --- persistent synchronous send ---
    template<typename T>
    prequest ssend_init(const T &data, int dest, tag t = tag(0)) const {
      check_dest(dest);
      check_send_tag(t);
      MPI_Request req;
      MPI_Ssend_init(&data, 1, datatype_traits<T>::get_datatype(), dest, static_cast<int>(t),
                     comm, &req);
      return prequest(req);
    }

    template<typename T>
    prequest ssend_init(const T *data, const layout<T> &l, int dest, tag t = tag(0)) const {
      check_dest(dest);
      check_send_tag(t);
      MPI_Request req;
      MPI_Ssend_init(data, 1, datatype_traits<layout<T>>::get_datatype(l), dest,
                     static_cast<int>(t), comm, &req);
      return prequest(req);
    }

    template<typename iterT>
    prequest ssend_init(iterT begin, iterT end, int dest, tag t = tag(0)) const {
      using value_type = typename std::iterator_traits<iterT>::value_type;
      if (detail::is_contiguous_iterator<iterT>::value) {
        vector_layout<value_type> l(std::distance(begin, end));
        return ssend_init(&(*begin), l, dest, t);
      } else {
        iterator_layout<value_type> l(begin, end);
        return ssend_init(&(*begin), l, dest, t);
      }
    }

    // === ready send ===
    // --- blocking ready send ---
  private:
    template<typename T>
    void rsend(const T &data, int dest, tag t, detail::basic_or_fixed_size_type) const {
      MPI_Rsend(&data, 1, datatype_traits<T>::get_datatype(), dest, static_cast<int>(t), comm);
    }

    template<typename T>
    void rsend(const T &data, int dest, tag t, detail::contigous_const_stl_container) const {
      using value_type = typename T::value_type;
      vector_layout<value_type> l(data.size());
      rsend(data.size() > 0 ? &data[0] : nullptr, l, dest, t);
    }

    template<typename T>
    void rsend(const T &data, int dest, tag t, detail::stl_container) const {
      using value_type =
          typename detail::remove_const_from_members<typename T::value_type>::type;
      detail::vector<value_type> serial_data(data.size(), std::begin(data));
      vector_layout<value_type> l(serial_data.size());
      rsend(serial_data.data(), l, dest, t);
    }

  public:
    template<typename T>
    void rsend(const T &data, int dest, tag t = tag(0)) const {
      check_dest(dest);
      check_send_tag(t);
      check_container_size(data);
      rsend(data, dest, t, typename datatype_traits<T>::data_type_category{});
    }

    template<typename T>
    void rsend(const T *data, const layout<T> &l, int dest, tag t = tag(0)) const {
      check_dest(dest);
      check_send_tag(t);
      MPI_Rsend(data, 1, datatype_traits<layout<T>>::get_datatype(l), dest, static_cast<int>(t),
                comm);
    }

    template<typename iterT>
    void rsend(iterT begin, iterT end, int dest, tag t = tag(0)) const {
      using value_type = typename std::iterator_traits<iterT>::value_type;
      if (detail::is_contiguous_iterator<iterT>::value) {
        vector_layout<value_type> l(std::distance(begin, end));
        rsend(&(*begin), l, dest, t);
      } else {
        iterator_layout<value_type> l(begin, end);
        rsend(&(*begin), l, dest, t);
      }
    }

    // --- nonblocking ready send ---
  private:
    template<typename T>
    irequest irsend(const T &data, int dest, tag t, detail::basic_or_fixed_size_type) const {
      MPI_Request req;
      MPI_Irsend(&data, 1, datatype_traits<T>::get_datatype(), dest, static_cast<int>(t), comm,
                 &req);
      return detail::irequest(req);
    }

    template<typename T>
    void irsend(const T &data, int dest, tag t, isend_irecv_state *isend_state,
                detail::contigous_const_stl_container) const {
      using value_type = typename T::value_type;
      const int count(data.size());
      MPI_Datatype datatype{datatype_traits<value_type>::get_datatype()};
      MPI_Request req;
      MPI_Irsend(data.size() > 0 ? &data[0] : nullptr, count, datatype, dest,
                 static_cast<int>(t), comm, &req);
      MPI_Status s;
      MPI_Wait(&req, &s);
      isend_state->source = s.MPI_SOURCE;
      isend_state->tag = s.MPI_TAG;
      isend_state->datatype = datatype;
      isend_state->count = 0;
      MPI_Grequest_complete(isend_state->req);
    }

    template<typename T>
    void irsend(const T &data, int dest, tag t, isend_irecv_state *isend_state,
                detail::stl_container) const {
      using value_type =
          typename detail::remove_const_from_members<typename T::value_type>::type;
      detail::vector<value_type> serial_data(data.size(), std::begin(data));
      irsend(serial_data, dest, t, isend_state, detail::contigous_const_stl_container{});
    }

    template<typename T, typename C>
    irequest irsend(const T &data, int dest, tag t, C) const {
      isend_irecv_state *send_state = new isend_irecv_state();
      MPI_Request req;
      MPI_Grequest_start(isend_irecv_query, isend_irecv_free, isend_irecv_cancel, send_state,
                         &req);
      send_state->req = req;
      std::thread thread(
          [this, &data, dest, t, send_state]() { irsend(data, dest, t, send_state, C{}); });
      thread.detach();
      return detail::irequest(req);
    }

  public:
    template<typename T>
    irequest irsend(const T &data, int dest, tag t = tag(0)) const {
      check_dest(dest);
      check_send_tag(t);
      check_container_size(data);
      return irsend(data, dest, t, typename datatype_traits<T>::data_type_category{});
    }

    template<typename T>
    irequest irsend(const T *data, const layout<T> &l, int dest, tag t = tag(0)) const {
      check_dest(dest);
      check_send_tag(t);
      MPI_Request req;
      MPI_Irsend(data, 1, datatype_traits<layout<T>>::get_datatype(l), dest,
                 static_cast<int>(t), comm, &req);
      return detail::irequest(req);
    }

    template<typename iterT>
    irequest irsend(iterT begin, iterT end, int dest, tag t = tag(0)) const {
      using value_type = typename std::iterator_traits<iterT>::value_type;
      if (detail::is_contiguous_iterator<iterT>::value) {
        vector_layout<value_type> l(std::distance(begin, end));
        return irsend(&(*begin), l, dest, t);
      } else {
        iterator_layout<value_type> l(begin, end);
        return irsend(&(*begin), l, dest, t);
      }
    }

    // --- persistent ready send ---
    template<typename T>
    prequest rsend_init(const T &data, int dest, tag t = tag(0)) const {
      check_dest(dest);
      check_send_tag(t);
      MPI_Request req;
      MPI_Rsend_init(&data, 1, datatype_traits<T>::get_datatype(), dest, static_cast<int>(t),
                     comm, &req);
      return prequest(req);
    }

    template<typename T>
    prequest rsend_init(const T *data, const layout<T> &l, int dest, tag t = tag(0)) const {
      check_dest(dest);
      check_send_tag(t);
      MPI_Request req;
      MPI_Rsend_init(data, 1, datatype_traits<layout<T>>::get_datatype(l), dest,
                     static_cast<int>(t), comm, &req);
      return prequest(req);
    }

    template<typename iterT>
    prequest rsend_init(iterT begin, iterT end, int dest, tag t = tag(0)) const {
      using value_type = typename std::iterator_traits<iterT>::value_type;
      if (detail::is_contiguous_iterator<iterT>::value) {
        vector_layout<value_type> l(std::distance(begin, end));
        return rsend_init(&(*begin), l, dest, t);
      } else {
        iterator_layout<value_type> l(begin, end);
        return rsend_init(&(*begin), l, dest, t);
      }
    }

    // === receive ===
    // --- blocking receive ---
  private:
    template<typename T>
    status recv(T &data, int source, tag t, detail::basic_or_fixed_size_type) const {
      status s;
      MPI_Recv(&data, 1, datatype_traits<T>::get_datatype(), source, static_cast<int>(t), comm,
               reinterpret_cast<MPI_Status *>(&s));
      return s;
    }

    template<typename T>
    status recv(T &data, int source, tag t, detail::contigous_stl_container) const {
      using value_type = typename T::value_type;
      status s;
      auto *ps{reinterpret_cast<MPI_Status *>(&s)};
      MPI_Message message;
      MPI_Mprobe(source, static_cast<int>(t), comm, &message, ps);
      int count{0};
      MPI_Get_count(ps, datatype_traits<value_type>::get_datatype(), &count);
      check_count(count);
      data.resize(count);
      MPI_Mrecv(data.size() > 0 ? &data[0] : nullptr, count,
                datatype_traits<value_type>::get_datatype(), &message, ps);
      return s;
    }

    template<typename T>
    status recv(T &data, int source, tag t, detail::stl_container) const {
      using value_type =
          typename detail::remove_const_from_members<typename T::value_type>::type;
      status s;
      auto *ps{reinterpret_cast<MPI_Status *>(&s)};
      MPI_Message message;
      MPI_Mprobe(source, static_cast<int>(t), comm, &message, ps);
      int count{0};
      MPI_Get_count(ps, datatype_traits<value_type>::get_datatype(), &count);
      check_count(count);
      detail::vector<value_type> serial_data(count, detail::uninitialized{});
      MPI_Mrecv(serial_data.data(), count, datatype_traits<value_type>::get_datatype(),
                &message, ps);
      T new_data(serial_data.begin(), serial_data.end());
      data.swap(new_data);
      return s;
    }

  public:
    template<typename T>
    status recv(T &data, int source, tag t = tag(0)) const {
      check_source(source);
      check_recv_tag(t);
      return recv(data, source, t, typename datatype_traits<T>::data_type_category{});
    }

    template<typename T>
    status recv(T *data, const layout<T> &l, int source, tag t = tag(0)) const {
      check_source(source);
      check_recv_tag(t);
      status s;
      MPI_Recv(data, 1, datatype_traits<layout<T>>::get_datatype(l), source,
               static_cast<int>(t), comm, reinterpret_cast<MPI_Status *>(&s));
      return s;
    }

    template<typename iterT>
    status recv(iterT begin, iterT end, int source, tag t = tag(0)) const {
      using value_type = typename std::iterator_traits<iterT>::value_type;
      if (detail::is_contiguous_iterator<iterT>::value) {
        vector_layout<value_type> l(std::distance(begin, end));
        return recv(&(*begin), l, source, t);
      } else {
        iterator_layout<value_type> l(begin, end);
        return recv(&(*begin), l, source, t);
      }
    }

    // --- nonblocking receive ---
  private:
    template<typename T>
    irequest irecv(T &data, int source, tag t, detail::basic_or_fixed_size_type) const {
      MPI_Request req;
      MPI_Irecv(&data, 1, datatype_traits<T>::get_datatype(), source, static_cast<int>(t), comm,
                &req);
      return detail::irequest(req);
    }

    template<typename T>
    void irecv(T &data, int source, tag t, isend_irecv_state *irecv_state,
               detail::stl_container) const {
      using value_type =
          typename detail::remove_const_from_members<typename T::value_type>::type;
      const status s{recv(data, source, t)};
      irecv_state->source = s.source();
      irecv_state->tag = static_cast<int>(s.tag());
      irecv_state->datatype = datatype_traits<value_type>::get_datatype();
      irecv_state->count = s.get_count<value_type>();
      MPI_Grequest_complete(irecv_state->req);
    }

    template<typename T, typename C>
    irequest irecv(T &data, int source, tag t, C) const {
      isend_irecv_state *recv_state = new isend_irecv_state();
      MPI_Request req;
      MPI_Grequest_start(isend_irecv_query, isend_irecv_free, isend_irecv_cancel, recv_state,
                         &req);
      recv_state->req = req;
      std::thread thread(
          [this, &data, source, t, recv_state]() { irecv(data, source, t, recv_state, C{}); });
      thread.detach();
      return detail::irequest(req);
    }

  public:
    template<typename T>
    irequest irecv(T &data, int source, tag t = tag(0)) const {
      check_source(source);
      check_recv_tag(t);
      return irecv(data, source, t, typename datatype_traits<T>::data_type_category{});
    }

    template<typename T>
    irequest irecv(T *data, const layout<T> &l, int source, tag t = tag(0)) const {
      check_source(source);
      check_recv_tag(t);
      MPI_Request req;
      MPI_Irecv(data, 1, datatype_traits<layout<T>>::get_datatype(l), source,
                static_cast<int>(t), comm, &req);
      return detail::irequest(req);
    }

    template<typename iterT>
    irequest irecv(iterT begin, iterT end, int source, tag t = tag(0)) const {
      using value_type = typename std::iterator_traits<iterT>::value_type;
      if (detail::is_contiguous_iterator<iterT>::value) {
        vector_layout<value_type> l(std::distance(begin, end));
        return irecv(&(*begin), l, source, t);
      } else {
        iterator_layout<value_type> l(begin, end);
        return irecv(&(*begin), l, source, t);
      }
    }

    // --- persistent receive ---
    template<typename T>
    prequest recv_init(T &data, int source, tag t = tag(0)) const {
      check_source(source);
      check_recv_tag(t);
      MPI_Request req;
      MPI_Recv_init(&data, 1, datatype_traits<T>::get_datatype(), source, static_cast<int>(t),
                    comm, &req);
      return prequest(req);
    }

    template<typename T>
    prequest recv_init(T *data, const layout<T> &l, int source, tag t = tag(0)) const {
      check_source(source);
      check_recv_tag(t);
      MPI_Request req;
      MPI_Recv_init(data, 1, datatype_traits<layout<T>>::get_datatype(l), source,
                    static_cast<int>(t), comm, &req);
      return prequest(req);
    }

    template<typename iterT>
    prequest recv_init(iterT begin, iterT end, int source, tag t = tag(0)) const {
      using value_type = typename std::iterator_traits<iterT>::value_type;
      if (detail::is_contiguous_iterator<iterT>::value) {
        vector_layout<value_type> l(std::distance(begin, end));
        return recv_init(&(*begin), l, source, t);
      } else {
        iterator_layout<value_type> l(begin, end);
        return recv_init(&(*begin), l, source, t);
      }
    }

    // === probe ===
    // --- blocking probe ---
    status probe(int source, tag t = tag(0)) const {
      check_source(source);
      check_recv_tag(t);
      status s;
      MPI_Probe(source, static_cast<int>(t), comm, reinterpret_cast<MPI_Status *>(&s));
      return s;
    }

    // --- nonblocking probe ---
    std::pair<bool, status> iprobe(int source, tag t = tag(0)) const {
      check_source(source);
      check_recv_tag(t);
      int result;
      status s;
      MPI_Iprobe(source, static_cast<int>(t), comm, &result,
                 reinterpret_cast<MPI_Status *>(&s));
      return std::make_pair(static_cast<bool>(result), s);
    }

    // === matching probe ===
    // --- blocking matching probe ---
    std::pair<message, status> mprobe(int source, tag t = tag(0)) const {
      check_source(source);
      check_recv_tag(t);
      status s;
      message m;
      MPI_Mprobe(source, static_cast<int>(t), comm, &m, reinterpret_cast<MPI_Status *>(&s));
      return std::make_pair(m, s);
    }

    // --- nonblocking matching probe ---
    std::tuple<bool, message, status> improbe(int source, tag t = tag(0)) const {
      check_source(source);
      check_recv_tag(t);
      int result;
      status s;
      message m;
      MPI_Improbe(source, static_cast<int>(t), comm, &result, &m,
                  reinterpret_cast<MPI_Status *>(&s));
      return std::make_tuple(static_cast<bool>(result), m, s);
    }

    // === matching receive ===
    // --- blocking matching receive ---

    // --- nonblocking matching receive ---

    // === send and receive ===
    // --- send and receive ---
    template<typename T>
    status sendrecv(const T &senddata, int dest, tag sendtag, T &recvdata, int source,
                    tag recvtag) const {
      check_dest(dest);
      check_source(source);
      check_send_tag(sendtag);
      check_recv_tag(recvtag);
      status s;
      MPI_Sendrecv(&senddata, 1, datatype_traits<T>::get_datatype(), dest,
                   static_cast<int>(sendtag), &recvdata, 1, datatype_traits<T>::get_datatype(),
                   source, static_cast<int>(recvtag), comm, reinterpret_cast<MPI_Status *>(&s));
      return s;
    }

    template<typename T>
    status sendrecv(const T *senddata, const layout<T> &sendl, int dest, tag sendtag,
                    T *recvdata, const layout<T> &recvl, int source, tag recvtag) const {
      check_dest(dest);
      check_source(source);
      check_send_tag(sendtag);
      check_recv_tag(recvtag);
      status s;
      MPI_Sendrecv(senddata, 1, datatype_traits<layout<T>>::get_datatype(sendl), dest,
                   static_cast<int>(sendtag), recvdata, 1,
                   datatype_traits<layout<T>>::get_datatype(recvl), source,
                   static_cast<int>(recvtag), comm, reinterpret_cast<MPI_Status *>(&s));
      return s;
    }

    template<typename iterT1, typename iterT2>
    status sendrecv(iterT1 begin1, iterT1 end1, int dest, tag sendtag, iterT2 begin2,
                    iterT2 end2, int source, tag recvtag) const {
      using value_type1 = typename std::iterator_traits<iterT1>::value_type;
      using value_type2 = typename std::iterator_traits<iterT2>::value_type;
      if (detail::is_contiguous_iterator<iterT1>::value and
          detail::is_contiguous_iterator<iterT2>::value) {
        vector_layout<value_type1> l1(std::distance(begin1, end1));
        vector_layout<value_type2> l2(std::distance(begin2, end2));
        return sendrecv(&(*begin1), l1, dest, sendtag, &(*begin2), l2, source, recvtag);
      } else if (detail::is_contiguous_iterator<iterT1>::value) {
        vector_layout<value_type1> l1(std::distance(begin1, end1));
        iterator_layout<value_type2> l2(begin2, end2);
        return sendrecv(&(*begin1), l1, dest, sendtag, &(*begin2), l2, source, recvtag);
      } else if (detail::is_contiguous_iterator<iterT2>::value) {
        iterator_layout<value_type2> l1(begin1, end1);
        vector_layout<value_type2> l2(std::distance(begin2, end2));
        return sendrecv(&(*begin1), l1, dest, sendtag, &(*begin2), l2, source, recvtag);
      } else {
        iterator_layout<value_type1> l1(begin1, end1);
        iterator_layout<value_type2> l2(begin2, end2);
        return sendrecv(&(*begin1), l1, dest, sendtag, &(*begin2), l2, source, recvtag);
      }
    }

    // --- send, receive and replace ---
    template<typename T>
    status sendrecv_replace(T &data, int dest, tag sendtag, int source, tag recvtag) const {
      check_dest(dest);
      check_source(source);
      check_send_tag(sendtag);
      check_recv_tag(recvtag);
      status s;
      MPI_Sendrecv_replace(&data, 1, datatype_traits<T>::get_datatype(), dest,
                           static_cast<int>(sendtag), source, static_cast<int>(recvtag), comm,
                           reinterpret_cast<MPI_Status *>(&s));
      return s;
    }

    template<typename T>
    status sendrecv_replace(T *data, const layout<T> &l, int dest, tag sendtag, int source,
                            tag recvtag) const {
      check_dest(dest);
      check_source(source);
      check_send_tag(sendtag);
      check_recv_tag(recvtag);
      status s;
      MPI_Sendrecv_replace(data, 1, datatype_traits<layout<T>>::get_datatype(l), dest,
                           static_cast<int>(sendtag), source, static_cast<int>(recvtag), comm,
                           reinterpret_cast<MPI_Status *>(&s));
      return s;
    }

    template<typename iterT>
    status sendrecv_replace(iterT begin, iterT end, int dest, tag sendtag, int source,
                            tag recvtag) const {
      using value_type = typename std::iterator_traits<iterT>::value_type;
      if (detail::is_contiguous_iterator<iterT>::value) {
        vector_layout<value_type> l(std::distance(begin, end));
        return sendrecv_replace(&(*begin), l, dest, sendtag, source, recvtag);
      } else {
        iterator_layout<value_type> l(begin, end);
        return sendrecv_replace(&(*begin), l, dest, sendtag, source, recvtag);
      }
    }

    // === collective ==================================================
    // === barrier ===
    // --- blocking barrier ---
    void barrier() const { MPI_Barrier(comm); }

    // --- nonblocking barrier ---
    irequest ibarrier() const {
      MPI_Request req;
      MPI_Ibarrier(comm, &req);
      return detail::irequest(req);
    }

    // === broadcast ===
    // --- blocking broadcast ---
    template<typename T>
    void bcast(int root, T &data) const {
      check_root(root);
      MPI_Bcast(&data, 1, datatype_traits<T>::get_datatype(), root, comm);
    }

    template<typename T>
    void bcast(int root, T *data, const layout<T> &l) const {
      check_root(root);
      MPI_Bcast(data, 1, datatype_traits<layout<T>>::get_datatype(l), root, comm);
    }

    // --- nonblocking broadcast ---
    template<typename T>
    irequest ibcast(int root, T &data) const {
      check_root(root);
      MPI_Request req;
      MPI_Ibcast(&data, 1, datatype_traits<T>::get_datatype(), root, comm, &req);
      return detail::irequest(req);
    }

    template<typename T>
    irequest ibcast(int root, T *data, const layout<T> &l) const {
      check_root(root);
      MPI_Request req;
      MPI_Ibcast(data, 1, datatype_traits<layout<T>>::get_datatype(l), root, comm, &req);
      return detail::irequest(req);
    }

    // === gather ===
    // === root gets a single value from each rank and stores in contiguous memory
    // --- blocking gather ---
    template<typename T>
    void gather(int root, const T &senddata, T *recvdata) const {
      check_root(root);
      MPI_Gather(&senddata, 1, datatype_traits<T>::get_datatype(), recvdata, 1,
                 datatype_traits<T>::get_datatype(), root, comm);
    }

    template<typename T>
    void gather(int root, const T *senddata, const layout<T> &sendl, T *recvdata,
                const layout<T> &recvl) const {
      check_root(root);
      MPI_Gather(senddata, 1, datatype_traits<layout<T>>::get_datatype(sendl), recvdata, 1,
                 datatype_traits<layout<T>>::get_datatype(recvl), root, comm);
    }

    // --- nonblocking gather ---
    template<typename T>
    irequest igather(int root, const T &senddata, T *recvdata) const {
      check_root(root);
      MPI_Request req;
      MPI_Igather(&senddata, 1, datatype_traits<T>::get_datatype(), recvdata, 1,
                  datatype_traits<T>::get_datatype(), root, comm, &req);
      return detail::irequest(req);
    }

    template<typename T>
    irequest igather(int root, const T *senddata, const layout<T> &sendl, T *recvdata,
                     const layout<T> &recvl) const {
      check_root(root);
      MPI_Request req;
      MPI_Igather(senddata, 1, datatype_traits<layout<T>>::get_datatype(sendl), recvdata, 1,
                  datatype_traits<layout<T>>::get_datatype(recvl), root, comm, &req);
      return detail::irequest(req);
    }

    // --- blocking gather, non-root variant ---
    template<typename T>
    void gather(int root, const T &senddata) const {
      check_nonroot(root);
      MPI_Gather(&senddata, 1, datatype_traits<T>::get_datatype(), 0, 0, MPI_DATATYPE_NULL,
                 root, comm);
    }

    template<typename T>
    void gather(int root, const T *senddata, const layout<T> &sendl) const {
      check_nonroot(root);
      MPI_Gather(senddata, 1, datatype_traits<layout<T>>::get_datatype(sendl), 0, 0,
                 MPI_DATATYPE_NULL, root, comm);
    }

    // --- nonblocking gather, non-root variant ---
    template<typename T>
    irequest igather(int root, const T &senddata) const {
      check_nonroot(root);
      MPI_Request req;
      MPI_Igather(&senddata, 1, datatype_traits<T>::get_datatype(), 0, 0, MPI_DATATYPE_NULL,
                  root, comm, &req);
      return detail::irequest(req);
    }

    template<typename T>
    irequest igather(int root, const T *senddata, const layout<T> &sendl) const {
      check_nonroot(root);
      MPI_Request req;
      MPI_Igather(senddata, 1, datatype_traits<layout<T>>::get_datatype(sendl), 0, 0,
                  MPI_DATATYPE_NULL, root, comm, &req);
      return detail::irequest(req);
    }

    // === root gets varying amount of data from each rank and stores in noncontiguous memory
    // --- blocking gather ---
    template<typename T>
    void gatherv(int root, const T *senddata, const layout<T> &sendl, T *recvdata,
                 const layouts<T> &recvls, const displacements &recvdispls) const {
      check_root(root);
      check_size(recvls);
      check_size(recvdispls);
      int N(size());
      displacements senddispls(N);
      layouts<T> sendls(N);
      sendls[root] = sendl;
      if (rank() == root)
        alltoallv(senddata, sendls, senddispls, recvdata, recvls, recvdispls);
      else
        alltoallv(senddata, sendls, senddispls, recvdata, mpl::layouts<T>(N), recvdispls);
    }

    template<typename T>
    void gatherv(int root, const T *senddata, const layout<T> &sendl, T *recvdata,
                 const layouts<T> &recvls) const {
      gatherv(root, senddata, sendl, recvdata, recvls, displacements(size()));
    }

    // --- nonblocking gather ---
    template<typename T>
    irequest igatherv(int root, const T *senddata, const layout<T> &sendl, T *recvdata,
                      const layouts<T> &recvls, const displacements &recvdispls) const {
      check_root(root);
      check_size(recvls);
      check_size(recvdispls);
      int N(size());
      displacements senddispls(N);
      layouts<T> sendls(N);
      sendls[root] = sendl;
      if (rank() == root)
        return ialltoallv(senddata, sendls, senddispls, recvdata, recvls, recvdispls);
      else
        return ialltoallv(senddata, sendls, senddispls, recvdata, mpl::layouts<T>(N),
                          recvdispls);
    }

    template<typename T>
    irequest igatherv(int root, const T *senddata, const layout<T> &sendl, T *recvdata,
                      const layouts<T> &recvls) const {
      return igatherv(root, senddata, sendl, recvdata, recvls, displacements(size()));
    }

    // --- blocking gather, non-root variant ---
    template<typename T>
    void gatherv(int root, const T *senddata, const layout<T> &sendl) const {
      check_nonroot(root);
      int N(size());
      displacements sendrecvdispls(N);
      layouts<T> sendls(N);
      sendls[root] = sendl;
      alltoallv(senddata, sendls, sendrecvdispls, static_cast<T *>(nullptr), mpl::layouts<T>(N),
                sendrecvdispls);
    }

    // --- nonblocking gather, non-root variant ---
    template<typename T>
    irequest igatherv(int root, const T *senddata, const layout<T> &sendl) const {
      check_nonroot(root);
      int N(size());
      displacements sendrecvdispls(N);
      layouts<T> sendls(N);
      sendls[root] = sendl;
      return ialltoallv(senddata, sendls, sendrecvdispls, static_cast<T *>(nullptr),
                        mpl::layouts<T>(N), sendrecvdispls);
    }

    // === allgather ===
    // === get a single value from each rank and stores in contiguous memory
    // --- blocking allgather ---
    template<typename T>
    void allgather(const T &senddata, T *recvdata) const {
      MPI_Allgather(&senddata, 1, datatype_traits<T>::get_datatype(), recvdata, 1,
                    datatype_traits<T>::get_datatype(), comm);
    }

    template<typename T>
    void allgather(const T *senddata, const layout<T> &sendl, T *recvdata,
                   const layout<T> &recvl) const {
      MPI_Allgather(senddata, 1, datatype_traits<layout<T>>::get_datatype(sendl), recvdata, 1,
                    datatype_traits<layout<T>>::get_datatype(recvl), comm);
    }

    // --- nonblocking allgather ---
    template<typename T>
    irequest iallgather(const T &senddata, T *recvdata) const {
      MPI_Request req;
      MPI_Iallgather(&senddata, 1, datatype_traits<T>::get_datatype(), recvdata, 1,
                     datatype_traits<T>::get_datatype(), comm, &req);
      return detail::irequest(req);
    }

    template<typename T>
    irequest iallgather(const T *senddata, const layout<T> &sendl, T *recvdata,
                        const layout<T> &recvl) const {
      MPI_Request req;
      MPI_Iallgather(senddata, 1, datatype_traits<layout<T>>::get_datatype(sendl), recvdata, 1,
                     datatype_traits<layout<T>>::get_datatype(recvl), comm, &req);
      return detail::irequest(req);
    }

    // === get varying amount of data from each rank and stores in noncontiguous memory
    // --- blocking allgather ---
    template<typename T>
    void allgatherv(const T *senddata, const layout<T> &sendl, T *recvdata,
                    const layouts<T> &recvls, const displacements &recvdispls) const {
      check_size(recvls);
      check_size(recvdispls);
      int N(size());
      displacements senddispls(N);
      layouts<T> sendls(N, sendl);
      alltoallv(senddata, sendls, senddispls, recvdata, recvls, recvdispls);
    }

    template<typename T>
    void allgatherv(const T *senddata, const layout<T> &sendl, T *recvdata,
                    const layouts<T> &recvls) const {
      allgatherv(senddata, sendl, recvdata, recvls, displacements(size()));
    }

    // --- nonblocking allgather ---
    template<typename T>
    irequest iallgatherv(const T *senddata, const layout<T> &sendl, T *recvdata,
                         const layouts<T> &recvls, const displacements &recvdispls) const {
      check_size(recvls);
      check_size(recvdispls);
      int N(size());
      displacements senddispls(N);
      layouts<T> sendls(N, sendl);
      return ialltoallv(senddata, sendls, senddispls, recvdata, recvls, recvdispls);
    }

    template<typename T>
    irequest iallgatherv(const T *senddata, const layout<T> &sendl, T *recvdata,
                         const layouts<T> &recvls) const {
      return iallgatherv(senddata, sendl, recvdata, recvls, displacements(size()));
    }

    // === scatter ===
    // === root sends a single value from contiguous memory to each rank
    // --- blocking scatter ---
    template<typename T>
    void scatter(int root, const T *senddata, T &recvdata) const {
      check_root(root);
      MPI_Scatter(senddata, 1, datatype_traits<T>::get_datatype(), &recvdata, 1,
                  datatype_traits<T>::get_datatype(), root, comm);
    }

    template<typename T>
    void scatter(int root, const T *senddata, const layout<T> &sendl, T *recvdata,
                 const layout<T> &recvl) const {
      check_root(root);
      MPI_Scatter(senddata, 1, datatype_traits<layout<T>>::get_datatype(sendl), recvdata, 1,
                  datatype_traits<layout<T>>::get_datatype(recvl), root, comm);
    }

    // --- nonblocking scatter ---
    template<typename T>
    irequest iscatter(int root, const T *senddata, T &recvdata) const {
      check_root(root);
      MPI_Request req;
      MPI_Iscatter(senddata, 1, datatype_traits<T>::get_datatype(), &recvdata, 1,
                   datatype_traits<T>::get_datatype(), root, comm, &req);
      return detail::irequest(req);
    }

    template<typename T>
    irequest iscatter(int root, const T *senddata, const layout<T> &sendl, T *recvdata,
                      const layout<T> &recvl) const {
      check_root(root);
      MPI_Request req;
      MPI_Iscatter(senddata, 1, datatype_traits<layout<T>>::get_datatype(sendl), recvdata, 1,
                   datatype_traits<layout<T>>::get_datatype(recvl), root, comm, &req);
      return detail::irequest(req);
    }

    // --- blocking scatter, non-root variant ---
    template<typename T>
    void scatter(int root, T &recvdata) const {
      check_nonroot(root);
      MPI_Scatter(0, 0, MPI_DATATYPE_NULL, &recvdata, 1, datatype_traits<T>::get_datatype(),
                  root, comm);
    }

    template<typename T>
    void scatter(int root, T *recvdata, const layout<T> &recvl) const {
      check_root(root);
      MPI_Scatter(0, 0, MPI_DATATYPE_NULL, recvdata, 1,
                  datatype_traits<layout<T>>::get_datatype(recvl), root, comm);
    }

    // --- nonblocking scatter, non-root variant ---
    template<typename T>
    irequest iscatter(int root, T &recvdata) const {
      check_nonroot(root);
      MPI_Request req;
      MPI_Iscatter(0, 0, MPI_DATATYPE_NULL, &recvdata, 1, datatype_traits<T>::get_datatype(),
                   root, comm, &req);
      return detail::irequest(req);
    }

    template<typename T>
    irequest iscatter(int root, T *recvdata, const layout<T> &recvl) const {
      check_nonroot(root);
      MPI_Request req;
      MPI_Iscatter(0, 0, MPI_DATATYPE_NULL, recvdata, 1,
                   datatype_traits<layout<T>>::get_datatype(recvl), root, comm, &req);
      return detail::irequest(req);
    }

    // === root sends varying amount of data from noncontiguous memory to each rank
    // --- blocking scatter ---
    template<typename T>
    void scatterv(int root, const T *senddata, const layouts<T> &sendls,
                  const displacements &senddispls, T *recvdata, const layout<T> &recvl) const {
      check_root(root);
      check_size(sendls);
      check_size(senddispls);
      int N(size());
      displacements recvdispls(N);
      layouts<T> recvls(N);
      recvls[root] = recvl;
      if (rank() == root)
        alltoallv(senddata, sendls, senddispls, recvdata, recvls, recvdispls);
      else
        alltoallv(senddata, sendls, senddispls, recvdata, mpl::layouts<T>(N), recvdispls);
    }

    template<typename T>
    void scatterv(int root, const T *senddata, const layouts<T> &sendls, T *recvdata,
                  const layout<T> &recvl) const {
      scatterv(root, senddata, sendls, displacements(size()), recvdata, recvl);
    }

    // --- nonblocking scatter ---
    template<typename T>
    irequest iscatterv(int root, const T *senddata, const layouts<T> &sendls,
                       const displacements &senddispls, T *recvdata,
                       const layout<T> &recvl) const {
      check_root(root);
      check_size(sendls);
      check_size(senddispls);
      int N(size());
      displacements recvdispls(N);
      layouts<T> recvls(N);
      recvls[root] = recvl;
      if (rank() == root)
        return ialltoallv(senddata, sendls, senddispls, recvdata, recvls, recvdispls);
      else
        return ialltoallv(senddata, sendls, senddispls, recvdata, mpl::layouts<T>(N),
                          recvdispls);
    }

    template<typename T>
    irequest iscatterv(int root, const T *senddata, const layouts<T> &sendls, T *recvdata,
                       const layout<T> &recvl) const {
      return iscatterv(root, senddata, sendls, displacements(size()), recvdata, recvl);
    }

    // --- blocking scatter, non-root variant ---
    template<typename T>
    void scatterv(int root, T *recvdata, const layout<T> &recvl) const {
      check_root(root);
      int N(size());
      displacements sendrecvdispls(N);
      layouts<T> recvls(N);
      recvls[root] = recvl;
      alltoallv(static_cast<const T *>(nullptr), mpl::layouts<T>(N), sendrecvdispls, recvdata,
                recvls, sendrecvdispls);
    }

    // --- nonblocking scatter, non-root variant ---
    template<typename T>
    irequest iscatterv(int root, T *recvdata, const layout<T> &recvl) const {
      check_root(root);
      int N(size());
      displacements sendrecvdispls(N);
      layouts<T> recvls(N);
      recvls[root] = recvl;
      return ialltoallv(static_cast<const T *>(nullptr), mpl::layouts<T>(N), sendrecvdispls,
                        recvdata, recvls, sendrecvdispls);
    }

    // === all-to-all ===
    // === each rank sends a single value to each rank
    // --- blocking all-to-all ---
    template<typename T>
    void alltoall(const T *senddata, T *recvdata) const {
      MPI_Alltoall(senddata, 1, datatype_traits<T>::get_datatype(), recvdata, 1,
                   datatype_traits<T>::get_datatype(), comm);
    }

    template<typename T>
    void alltoall(const T *senddata, const layout<T> &sendl, T *recvdata,
                  const layout<T> &recvl) const {
      MPI_Alltoall(senddata, 1, datatype_traits<layout<T>>::get_datatype(), recvdata, 1,
                   datatype_traits<layout<T>>::get_datatype(), comm);
    }

    // --- nonblocking all-to-all ---
    template<typename T>
    irequest ialltoall(const T *senddata, T *recvdata) const {
      MPI_Request req;
      MPI_Ialltoall(senddata, 1, datatype_traits<T>::get_datatype(), recvdata, 1,
                    datatype_traits<T>::get_datatype(), comm, &req);
      return detail::irequest(req);
    }

    template<typename T>
    irequest ialltoall(const T *senddata, const layout<T> &sendl, T *recvdata,
                       const layout<T> &recvl) const {
      MPI_Request req;
      MPI_Ialltoall(senddata, 1, datatype_traits<layout<T>>::get_datatype(), recvdata, 1,
                    datatype_traits<layout<T>>::get_datatype(), comm, &req);
      return detail::irequest(req);
    }

    // --- blocking all-to-all, in place ---
    template<typename T>
    void alltoall(T *recvdata) const {
      MPI_Alltoall(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, recvdata, 1,
                   datatype_traits<T>::get_datatype(), comm);
    }

    template<typename T>
    void alltoall(T *recvdata, const layout<T> &recvl) const {
      MPI_Alltoall(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, recvdata, 1,
                   datatype_traits<layout<T>>::get_datatype(), comm);
    }

    // --- nonblocking all-to-all, in place ---
    template<typename T>
    irequest ialltoall(T *recvdata) const {
      MPI_Request req;
      MPI_Ialltoall(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, recvdata, 1,
                    datatype_traits<T>::get_datatype(), comm, &req);
      return detail::irequest(req);
    }

    template<typename T>
    irequest ialltoall(T *recvdata, const layout<T> &recvl) const {
      MPI_Request req;
      MPI_Ialltoall(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, recvdata, 1,
                    datatype_traits<layout<T>>::get_datatype(), comm, &req);
      return detail::irequest(req);
    }

    // === each rank sends a varying number of values to each rank with possibly different
    // layouts
    // --- blocking all-to-all ---
    template<typename T>
    void alltoallv(const T *senddata, const layouts<T> &sendl, const displacements &senddispls,
                   T *recvdata, const layouts<T> &recvl,
                   const displacements &recvdispls) const {
      check_size(senddispls);
      check_size(sendl);
      check_size(recvdispls);
      check_size(recvl);
      std::vector<int> counts(recvl.size(), 1);
      std::vector<int> senddispls_int(senddispls.begin(), senddispls.end());
      std::vector<int> recvdispls_int(recvdispls.begin(), recvdispls.end());
      MPI_Alltoallw(senddata, counts.data(), senddispls_int.data(),
                    reinterpret_cast<const MPI_Datatype *>(sendl()), recvdata, counts.data(),
                    recvdispls_int.data(), reinterpret_cast<const MPI_Datatype *>(recvl()),
                    comm);
    }

    template<typename T>
    void alltoallv(const T *senddata, const layouts<T> &sendl, T *recvdata,
                   const layouts<T> &recvl) const {
      displacements sendrecvdispls(size());
      alltoallv(senddata, sendl, sendrecvdispls, recvdata, recvl, sendrecvdispls);
    }

    // --- non-blocking all-to-all ---
    template<typename T>
    irequest ialltoallv(const T *senddata, const layouts<T> &sendl,
                        const displacements &senddispls, T *recvdata, const layouts<T> &recvl,
                        const displacements &recvdispls) const {
      check_size(senddispls);
      check_size(sendl);
      check_size(recvdispls);
      check_size(recvl);
      std::vector<int> counts(recvl.size(), 1);
      std::vector<int> senddispls_int(senddispls.begin(), senddispls.end());
      std::vector<int> recvdispls_int(recvdispls.begin(), recvdispls.end());
      MPI_Request req;
      MPI_Ialltoallw(senddata, counts.data(), senddispls_int.data(),
                     reinterpret_cast<const MPI_Datatype *>(sendl()), recvdata, counts.data(),
                     recvdispls_int.data(), reinterpret_cast<const MPI_Datatype *>(recvl()),
                     comm, &req);
      return detail::irequest(req);
    }

    template<typename T>
    irequest ialltoallv(const T *senddata, const layouts<T> &sendl, T *recvdata,
                        const layouts<T> &recvl) const {
      displacements sendrecvdispls(size());
      return ialltoallv(senddata, sendl, sendrecvdispls, recvdata, recvl, sendrecvdispls);
    }

    // --- blocking all-to-all, in place ---
    template<typename T>
    void alltoallv(T *recvdata, const layouts<T> &recvl,
                   const displacements &recvdispls) const {
      check_size(recvdispls);
      check_size(recvl);
      std::vector<int> counts(recvl.size(), 1);
      std::vector<int> recvdispls_int(recvdispls.begin(), recvdispls.end());
      MPI_Alltoallw(MPI_IN_PLACE, 0, 0, 0, recvdata, counts.data(), recvdispls_int.data(),
                    reinterpret_cast<const MPI_Datatype *>(recvl()), comm);
    }

    template<typename T>
    void alltoallv(T *recvdata, const layouts<T> &recvl) const {
      alltoallv(recvdata, recvl, displacements(size()));
    }

    // --- non-blocking all-to-all, in place ---
    template<typename T>
    irequest ialltoallv(T *recvdata, const layouts<T> &recvl,
                        const displacements &recvdispls) const {
      check_size(recvdispls);
      check_size(recvl);
      std::vector<int> counts(recvl.size(), 1);
      std::vector<int> recvdispls_int(recvdispls.begin(), recvdispls.end());
      MPI_Request req;
      MPI_Ialltoallw(MPI_IN_PLACE, 0, 0, 0, recvdata, counts.data(), recvdispls_int.data(),
                     reinterpret_cast<const MPI_Datatype *>(recvl()), comm, &req);
      return detail::irequest(req);
    }

    template<typename T>
    irequest ialltoallv(T *recvdata, const layouts<T> &recvl) const {
      return ialltoallv(recvdata, recvl, displacements(size()));
    }

    // === reduce ===
    // --- blocking reduce ---
    template<typename T, typename F>
    void reduce(F f, int root, const T &senddata, T &recvdata) const {
      check_root(root);
      MPI_Reduce(&senddata, &recvdata, 1, datatype_traits<T>::get_datatype(),
                 detail::get_op<T, F>(f).mpi_op, root, comm);
    }

    template<typename T, typename F>
    void reduce(F f, int root, const T *senddata, T *recvdata,
                const contiguous_layout<T> &l) const {
      check_root(root);
      MPI_Reduce(senddata, recvdata, l.size(), datatype_traits<T>::get_datatype(),
                 detail::get_op<T, F>(f).mpi_op, root, comm);
    }

    // --- non-blocking reduce ---
    template<typename T, typename F>
    irequest ireduce(F f, int root, const T &senddata, T &recvdata) const {
      check_root(root);
      MPI_Request req;
      MPI_Ireduce(&senddata, &recvdata, 1, datatype_traits<T>::get_datatype(),
                  detail::get_op<T, F>(f).mpi_op, root, comm, &req);
      return detail::irequest(req);
    }

    template<typename T, typename F>
    irequest ireduce(F f, int root, const T *senddata, T *recvdata,
                     const contiguous_layout<T> &l) const {
      check_root(root);
      MPI_Request req;
      MPI_Ireduce(senddata, recvdata, l.size(), datatype_traits<T>::get_datatype(),
                  detail::get_op<T, F>(f).mpi_op, root, comm, &req);
      return detail::irequest(req);
    }

    // --- blocking reduce, in place ---
    template<typename T, typename F>
    void reduce(F f, int root, T &sendrecvdata) const {
      check_root(root);
      if (rank() == root)
        MPI_Reduce(MPI_IN_PLACE, &sendrecvdata, 1, datatype_traits<T>::get_datatype(),
                   detail::get_op<T, F>(f).mpi_op, root, comm);
      else
        MPI_Reduce(&sendrecvdata, nullptr, 1, datatype_traits<T>::get_datatype(),
                   detail::get_op<T, F>(f).mpi_op, root, comm);
    }

    template<typename T, typename F>
    void reduce(F f, int root, const T &senddata) const {
      check_nonroot(root);
      MPI_Reduce(&senddata, nullptr, 1, datatype_traits<T>::get_datatype(),
                 detail::get_op<T, F>(f).mpi_op, root, comm);
    }

    template<typename T, typename F>
    void reduce(F f, int root, T *sendrecvdata, const contiguous_layout<T> &l) const {
      if (rank() == root)
        MPI_Reduce(MPI_IN_PLACE, sendrecvdata, l.size(), datatype_traits<T>::get_datatype(),
                   detail::get_op<T, F>(f).mpi_op, root, comm);
      else
        MPI_Reduce(sendrecvdata, nullptr, l.size(), datatype_traits<T>::get_datatype(),
                   detail::get_op<T, F>(f).mpi_op, root, comm);
    }

    template<typename T, typename F>
    void reduce(F f, int root, const T *sendrecvdata, const contiguous_layout<T> &l) const {
      check_nonroot(root);
      MPI_Reduce(sendrecvdata, nullptr, l.size(), datatype_traits<T>::get_datatype(),
                 detail::get_op<T, F>(f).mpi_op, root, comm);
    }

    // --- non-blocking reduce, in place ---
    template<typename T, typename F>
    irequest ireduce(F f, int root, T &sendrecvdata) const {
      check_root(root);
      MPI_Request req;
      if (rank() == root)
        MPI_Ireduce(MPI_IN_PLACE, &sendrecvdata, 1, datatype_traits<T>::get_datatype(),
                    detail::get_op<T, F>(f).mpi_op, root, comm, &req);
      else
        MPI_Ireduce(&sendrecvdata, nullptr, 1, datatype_traits<T>::get_datatype(),
                    detail::get_op<T, F>(f).mpi_op, root, comm, &req);
      return detail::irequest(req);
    }

    template<typename T, typename F>
    irequest ireduce(F f, int root, const T &sendrecvdata) const {
      check_nonroot(root);
      MPI_Request req;
      MPI_Ireduce(&sendrecvdata, nullptr, 1, datatype_traits<T>::get_datatype(),
                  detail::get_op<T, F>(f).mpi_op, root, comm, &req);
      return detail::irequest(req);
    }

    template<typename T, typename F>
    irequest ireduce(F f, int root, T *sendrecvdata, const contiguous_layout<T> &l) const {
      check_root(root);
      MPI_Request req;
      if (rank() == root)
        MPI_Ireduce(MPI_IN_PLACE, sendrecvdata, l.size(), datatype_traits<T>::get_datatype(),
                    detail::get_op<T, F>(f).mpi_op, root, comm, &req);
      else
        MPI_Ireduce(sendrecvdata, nullptr, l.size(), datatype_traits<T>::get_datatype(),
                    detail::get_op<T, F>(f).mpi_op, root, comm, &req);
      return detail::irequest(req);
    }

    template<typename T, typename F>
    irequest ireduce(F f, int root, const T *sendrecvdata,
                     const contiguous_layout<T> &l) const {
      check_nonroot(root);
      MPI_Request req;
      MPI_Ireduce(sendrecvdata, nullptr, l.size(), datatype_traits<T>::get_datatype(),
                  detail::get_op<T, F>(f).mpi_op, root, comm, &req);
      return detail::irequest(req);
    }

    // === all-reduce ===
    // --- blocking all-reduce ---
    template<typename T, typename F>
    void allreduce(F f, const T &senddata, T &recvdata) const {
      MPI_Allreduce(&senddata, &recvdata, 1, datatype_traits<T>::get_datatype(),
                    detail::get_op<T, F>(f).mpi_op, comm);
    }

    template<typename T, typename F>
    void allreduce(F f, const T *senddata, T *recvdata, const contiguous_layout<T> &l) const {
      MPI_Allreduce(senddata, recvdata, l.size(), datatype_traits<T>::get_datatype(),
                    detail::get_op<T, F>(f).mpi_op, comm);
    }

    // --- non-blocking all-reduce ---
    template<typename T, typename F>
    irequest iallreduce(F f, const T &senddata, T &recvdata) const {
      MPI_Request req;
      MPI_Iallreduce(&senddata, &recvdata, 1, datatype_traits<T>::get_datatype(),
                     detail::get_op<T, F>(f).mpi_op, comm, &req);
      return detail::irequest(req);
    }

    template<typename T, typename F>
    irequest iallreduce(F f, const T *senddata, T *recvdata,
                        const contiguous_layout<T> &l) const {
      MPI_Request req;
      MPI_Iallreduce(senddata, recvdata, l.size(), datatype_traits<T>::get_datatype(),
                     detail::get_op<T, F>(f).mpi_op, comm, &req);
      return detail::irequest(req);
    }

    // --- blocking all-reduce, in place ---
    template<typename T, typename F>
    void allreduce(F f, T &sendrecvdata) const {
      MPI_Allreduce(MPI_IN_PLACE, &sendrecvdata, 1, datatype_traits<T>::get_datatype(),
                    detail::get_op<T, F>(f).mpi_op, comm);
    }

    template<typename T, typename F>
    void allreduce(F f, T *sendrecvdata, const contiguous_layout<T> &l) const {
      MPI_Allreduce(MPI_IN_PLACE, sendrecvdata, l.size(), datatype_traits<T>::get_datatype(),
                    detail::get_op<T, F>(f).mpi_op, comm);
    }

    // --- non-blocking all-reduce, in place ---
    template<typename T, typename F>
    irequest iallreduce(F f, T &sendrecvdata) const {
      MPI_Request req;
      MPI_Iallreduce(MPI_IN_PLACE, &sendrecvdata, 1, datatype_traits<T>::get_datatype(),
                     detail::get_op<T, F>(f).mpi_op, comm, &req);
      return detail::irequest(req);
    }

    template<typename T, typename F>
    irequest iallreduce(F f, T *sendrecvdata, const contiguous_layout<T> &l) const {
      MPI_Request req;
      MPI_Iallreduce(MPI_IN_PLACE, sendrecvdata, l.size(), datatype_traits<T>::get_datatype(),
                     detail::get_op<T, F>(f).mpi_op, comm, &req);
      return detail::irequest(req);
    }

    // === reduce-scatter-block ===
    // --- blocking reduce-scatter-block ---
    template<typename T, typename F>
    void reduce_scatter_block(F f, const T *senddata, T &recvdata) const {
      MPI_Reduce_scatter_block(senddata, &recvdata, 1, datatype_traits<T>::get_datatype(),
                               detail::get_op<T, F>(f).mpi_op, comm);
    }

    template<typename T, typename F>
    void reduce_scatter_block(F f, const T *senddata, T *recvdata,
                              const contiguous_layout<T> &l) const {
      MPI_Reduce_scatter_block(senddata, recvdata, l.size(), datatype_traits<T>::get_datatype(),
                               detail::get_op<T, F>(f).mpi_op, comm);
    }

    // --- non-blocking reduce-scatter-block ---
    template<typename T, typename F>
    irequest ireduce_scatter_block(F f, const T *senddata, T &recvdata) const {
      MPI_Request req;
      MPI_Ireduce_scatter_block(senddata, &recvdata, 1, datatype_traits<T>::get_datatype(),
                                detail::get_op<T, F>(f).mpi_op, comm, &req);
      return detail::irequest(req);
    }

    template<typename T, typename F>
    irequest ireduce_scatter_block(F f, const T *senddata, T *recvdata,
                                   const contiguous_layout<T> &l) const {
      MPI_Request req;
      MPI_Ireduce_scatter_block(senddata, recvdata, l.size(),
                                datatype_traits<T>::get_datatype(),
                                detail::get_op<T, F>(f).mpi_op, comm, &req);
      return detail::irequest(req);
    }

    // === reduce-scatter ===
    // --- blocking reduce-scatter ---
    template<typename T, typename F>
    void reduce_scatter(F f, const T *senddata, T *recvdata,
                        const contiguous_layouts<T> &recvcounts) const {
      MPI_Reduce_scatter(senddata, recvdata, recvcounts.sizes(),
                         datatype_traits<T>::get_datatype(), detail::get_op<T, F>(f).mpi_op,
                         comm);
    }

    // --- non-blocking reduce-scatter ---
    template<typename T, typename F>
    irequest ireduce_scatter(F f, const T *senddata, T *recvdata,
                             contiguous_layouts<T> &recvcounts) const {
      MPI_Request req;
      MPI_Ireduce_scatter(senddata, recvdata, recvcounts.sizes(),
                          datatype_traits<T>::get_datatype(), detail::get_op<T, F>(f).mpi_op,
                          comm, &req);
      return detail::irequest(req);
    }

    // === scan ===
    // --- blocking scan ---
    template<typename T, typename F>
    void scan(F f, const T &senddata, T &recvdata) const {
      MPI_Scan(&senddata, &recvdata, 1, datatype_traits<T>::get_datatype(),
               detail::get_op<T, F>(f).mpi_op, comm);
    }

    template<typename T, typename F>
    void scan(F f, const T *senddata, T *recvdata, const contiguous_layout<T> &l) const {
      MPI_Scan(senddata, recvdata, l.size(), datatype_traits<T>::get_datatype(),
               detail::get_op<T, F>(f).mpi_op, comm);
    }

    // --- non-blocking scan ---
    template<typename T, typename F>
    irequest iscan(F f, const T &senddata, T &recvdata) const {
      MPI_Request req;
      MPI_Iscan(&senddata, &recvdata, 1, datatype_traits<T>::get_datatype(),
                detail::get_op<T, F>(f).mpi_op, comm, &req);
      return detail::irequest(req);
    }

    template<typename T, typename F>
    irequest iscan(F f, const T *senddata, T *recvdata, const contiguous_layout<T> &l) const {
      MPI_Request req;
      MPI_Iscan(senddata, recvdata, l.size(), datatype_traits<T>::get_datatype(),
                detail::get_op<T, F>(f).mpi_op, comm, &req);
      return detail::irequest(req);
    }

    // --- blocking scan, in place ---
    template<typename T, typename F>
    void scan(F f, T &recvdata) const {
      MPI_Scan(MPI_IN_PLACE, &recvdata, 1, datatype_traits<T>::get_datatype(),
               detail::get_op<T, F>(f).mpi_op, comm);
    }

    template<typename T, typename F>
    void scan(F f, T *recvdata, const contiguous_layout<T> &l) const {
      MPI_Scan(MPI_IN_PLACE, recvdata, l.size(), datatype_traits<T>::get_datatype(),
               detail::get_op<T, F>(f).mpi_op, comm);
    }

    // --- non-blocking scan, in place ---
    template<typename T, typename F>
    irequest iscan(F f, T &recvdata) const {
      MPI_Request req;
      MPI_Iscan(MPI_IN_PLACE, &recvdata, 1, datatype_traits<T>::get_datatype(),
                detail::get_op<T, F>(f).mpi_op, comm, &req);
      return detail::irequest(req);
    }

    template<typename T, typename F>
    irequest iscan(F f, T *recvdata, const contiguous_layout<T> &l) const {
      MPI_Request req;
      MPI_Iscan(MPI_IN_PLACE, recvdata, l.size(), datatype_traits<T>::get_datatype(),
                detail::get_op<T, F>(f).mpi_op, comm, &req);
      return detail::irequest(req);
    }

    // === exscan ===
    // --- blocking exscan ---
    template<typename T, typename F>
    void exscan(F f, const T &senddata, T &recvdata) const {
      MPI_Exscan(&senddata, &recvdata, 1, datatype_traits<T>::get_datatype(),
                 detail::get_op<T, F>(f).mpi_op, comm);
    }

    template<typename T, typename F>
    void exscan(F f, const T *senddata, T *recvdata, const contiguous_layout<T> &l) const {
      MPI_Exscan(senddata, recvdata, l.size(), datatype_traits<T>::get_datatype(),
                 detail::get_op<T, F>(f).mpi_op, comm);
    }

    // --- non-blocking exscan ---
    template<typename T, typename F>
    irequest iexscan(F f, const T &senddata, T &recvdata) const {
      MPI_Request req;
      MPI_Iexscan(&senddata, &recvdata, 1, datatype_traits<T>::get_datatype(),
                  detail::get_op<T, F>(f).mpi_op, comm, &req);
      return detail::irequest(req);
    }

    template<typename T, typename F>
    irequest iexscan(F f, const T *senddata, T *recvdata, const contiguous_layout<T> &l) const {
      MPI_Request req;
      MPI_Iexscan(senddata, recvdata, l.size(), datatype_traits<T>::get_datatype(),
                  detail::get_op<T, F>(f).mpi_op, comm, &req);
      return detail::irequest(req);
    }

    // --- blocking exscan, in place ---
    template<typename T, typename F>
    void exscan(F f, T &recvdata) const {
      MPI_Exscan(MPI_IN_PLACE, &recvdata, 1, datatype_traits<T>::get_datatype(),
                 detail::get_op<T, F>(f).mpi_op, comm);
    }

    template<typename T, typename F>
    void exscan(F f, T *recvdata, const contiguous_layout<T> &l) const {
      MPI_Exscan(MPI_IN_PLACE, recvdata, l.size(), datatype_traits<T>::get_datatype(),
                 detail::get_op<T, F>(f).mpi_op, comm);
    }

    // --- non-blocking exscan, in place ---
    template<typename T, typename F>
    irequest iexscan(F f, T &recvdata) const {
      MPI_Request req;
      MPI_Iexscan(MPI_IN_PLACE, &recvdata, 1, datatype_traits<T>::get_datatype(),
                  detail::get_op<T, F>(f).mpi_op, comm, &req);
      return detail::irequest(req);
    }

    template<typename T, typename F>
    irequest iexscan(F f, T *recvdata, const contiguous_layout<T> &l) const {
      MPI_Request req;
      MPI_Iexscan(MPI_IN_PLACE, recvdata, l.size(), datatype_traits<T>::get_datatype(),
                  detail::get_op<T, F>(f).mpi_op, comm, &req);
      return detail::irequest(req);
    }
  };  // namespace mpl

  //--------------------------------------------------------------------

  inline group::group(const communicator &comm) { MPI_Comm_group(comm.comm, &gr); }

  inline group::group(group::Union, const group &other_1, const group &other_2) {
    MPI_Group_union(other_1.gr, other_2.gr, &gr);
  }

  inline group::group(group::intersection, const group &other_1, const group &other_2) {
    MPI_Group_intersection(other_1.gr, other_2.gr, &gr);
  }

  inline group::group(group::difference, const group &other_1, const group &other_2) {
    MPI_Group_difference(other_1.gr, other_2.gr, &gr);
  }

  inline group::group(group::incl, const group &other, const ranks &rank) {
    MPI_Group_incl(other.gr, rank.size(), rank(), &gr);
  }

  inline group::group(group::excl, const group &other, const ranks &rank) {
    MPI_Group_excl(other.gr, rank.size(), rank(), &gr);
  }

}  // namespace mpl

#endif
