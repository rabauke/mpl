#if !(defined MPL_COMM_GROUP_HPP)

#define MPL_COMM_GROUP_HPP

#include <mpi.h>
#include <type_traits>
#include <tuple>

namespace mpl {

  class group;
  class communicator;
  class cart_communicator;

  namespace environment {

    namespace detail {

      class env;

    }

    int tag_up();
    constexpr int any_tag();
    constexpr int any_source();
    constexpr int proc_null();

  }

  //--------------------------------------------------------------------

  class group {
    MPI_Group gr;
  public:
    enum class equality_type { ident=MPI_IDENT, similar=MPI_SIMILAR, unequal=MPI_UNEQUAL };
    struct Union {};
    struct intersection {};
    struct difference {};
    struct incl {};
    struct excl {};
    group() : gr(MPI_GROUP_EMPTY) {
    }
    group(const communicator &comm);  // define later
    group(group &&other) {
      gr=other.gr;
      other.gr=MPI_GROUP_EMPTY;
    }
    group(Union,
	  const group &other_1, const group &other_2);  // define later
    group(intersection,
	  const group &other_1, const group &other_2);  // define later
    group(difference,
	  const group &other_1, const group &other_2);  // define later
    group(incl,
	  const group &other, const ranks &rank);  // define later
    group(excl,
	  const group &other, const ranks &rank);  // define later
    ~group() {
      int result;
      MPI_Group_compare(gr, MPI_GROUP_EMPTY, &result);
      if (result!=MPI_IDENT)
	MPI_Group_free(&gr);
    }
    void operator=(const group &)=delete;
    group & operator=(group &&other) {
      if (this!=&other) {
	int result;
	MPI_Group_compare(gr, MPI_GROUP_EMPTY, &result);
	if (result!=MPI_IDENT)
	  MPI_Group_free(&gr);
	gr=other.gr;
	other.gr=MPI_GROUP_EMPTY;
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
      MPI_Group_translate_ranks(gr, static_cast<int>(rank.size()),
				rank(), other.gr, other_rank());
      return other_rank;
    }
    bool operator==(const group &other) const {
      int result;
      MPI_Group_compare(gr, other.gr, &result);
      return result==MPI_IDENT;
    }
    bool operator!=(const group &other) const {
      int result;
      MPI_Group_compare(gr, other.gr, &result);
      return result!=MPI_IDENT;
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
  protected:
    MPI_Comm comm;
  public:
    enum class equality_type { ident=MPI_IDENT, congruent=MPI_CONGRUENT, similar=MPI_SIMILAR, unequal=MPI_UNEQUAL };
    class comm_collective { };
    class group_collective { };
    class split { };
    class split_shared { };

    void check_dest(int dest) const {
#if defined MPL_DEBUG
      if ((dest)!=environment::proc_null() and
	  ((dest)<0 or (dest)>=size()))
	throw invalid_rank();
#endif
    }

    void check_source(int source) const {
#if defined MPL_DEBUG
      if ((source)!=environment::proc_null() and
	  (source)!=environment::any_source() and
	  ((source)<0 or (source)>=size()))
	throw invalid_rank();
#endif
    }

    void check_stag(int tag) const {
#if defined MPL_DEBUG
      if ((tag)<0 or (tag)>environment::tag_up())
	throw invalid_tag();
#endif
    }
    
    void check_rtag(int tag) const {
#if defined MPL_DEBUG
      if ((tag)!=environment::any_tag() and
	  ((tag)<0 or tag>environment::tag_up()))
	throw invalid_tag();
#endif
    }
    
    void check_root(int root) const {
#if defined MPL_DEBUG
      if ((root)<0 or (root)>=size())
	throw invalid_rank();
#endif
    }

    void check_nonroot(int root) const {
#if defined MPL_DEBUG
      if ((root)<0 or (root)>=size() or (root)==rank())
	throw invalid_rank();
#endif
    }

    template<typename T>
    void check_size(const layouts<T> &l) const {
#if defined MPL_DEBUG
      if (static_cast<int>(l.size())>size())
	throw invalid_size();
#endif
    }
    void check_size(const displacements &d) const {
#if defined MPL_DEBUG
      if (static_cast<int>(d.size())>size())
	throw invalid_size();
#endif
    }

  protected:
    communicator(MPI_Comm comm) : comm(comm) {
    }
  public:
    communicator() : comm(MPI_COMM_NULL) {
    }
    communicator(const communicator &other) {
      MPI_Comm_dup(other.comm, &comm);
    }
    communicator(communicator &&other) {
      comm=other.comm;
      other.comm=MPI_COMM_NULL;
    }
    communicator(comm_collective, const communicator &other, const group &gr) {
      MPI_Comm_create(other.comm, gr.gr, &comm);
    }
    
    template<typename tag_type=int>
    communicator(group_collective, const communicator &other, const group &gr, tag_type tag=0) {
      static_assert(detail::is_valid_tag<tag_type>::value,
		    "not an enumeration type or underlying enumeration type too large");
      MPI_Comm_create_group(other.comm, gr.gr, detail::underlying_type<tag_type>::value(tag), &comm);
    }
    template<typename color_type, typename key_type=int>
    communicator(split, const communicator &other, color_type color, key_type key=0) {
      static_assert(detail::is_valid_tag<color_type>::value,
		    "not an enumeration type or underlying enumeration type too large");
      static_assert(detail::is_valid_tag<key_type>::value,
		    "not an enumeration type or underlying enumeration type too large");
      MPI_Comm_split(other.comm, detail::underlying_type<color_type>::value(color),
		     detail::underlying_type<key_type>::value(key), &comm);
    }
    template<typename key_type=int>
    communicator(split_shared, const communicator &other, key_type key=0) {
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
	if (result1!=MPI_IDENT and result2!=MPI_IDENT)
	  MPI_Comm_free(&comm);
      }
    }
    void operator=(const communicator &)=delete;
    communicator & operator=(communicator &&other) {
      if (this!=&other) {
	if (is_valid()) {
	  int result1, result2;
	  MPI_Comm_compare(comm, MPI_COMM_WORLD, &result1);
	  MPI_Comm_compare(comm, MPI_COMM_SELF, &result2);
	  if (result1!=MPI_IDENT and result2!=MPI_IDENT)
	    MPI_Comm_free(&comm);
	}
	comm=other.comm;
	other.comm=MPI_COMM_NULL;
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
      return result==MPI_IDENT;
    }
    bool operator!=(const communicator &other) const {
      int result;
      MPI_Comm_compare(comm, other.comm, &result);
      return result!=MPI_IDENT;
    }
    equality_type compare(const communicator &other) const {
      int result;
      MPI_Comm_compare(comm, other.comm, &result);
      return static_cast<equality_type>(result);
    }
    bool is_valid() const {
      return comm!=MPI_COMM_NULL;
    }
    void abort(int err) const {
      MPI_Abort(comm, err);
    }

    friend class group;
    friend class cart_communicator;
    friend class environment::detail::env;

    // === point to point ==============================================

    // === standard send ===
    // --- blocking standard send ---
    template<typename T, typename tag_type=int>
    void send(const T &data, int dest, tag_type tag=0) const {
      static_assert(detail::is_valid_tag<tag_type>::value,
		    "not an enumeration type or underlying enumeration type too large");
      check_dest(dest);
      check_stag(detail::underlying_type<tag_type>::value(tag));
      MPI_Send(&data, 1,
	       datatype_traits<T>::get_datatype(),
	       dest, detail::underlying_type<tag_type>::value(tag), comm);
    }
    
    template<typename T, typename tag_type=int>
    void send(const T *data, const layout<T> &l, int dest, tag_type tag=0) const {
      static_assert(detail::is_valid_tag<tag_type>::value,
		    "not an enumeration type or underlying enumeration type too large");
      check_dest(dest);
      check_stag(detail::underlying_type<tag_type>::value(tag));
      MPI_Send(data, 1,
      	       datatype_traits<layout<T>>::get_datatype(l),
      	       dest, detail::underlying_type<tag_type>::value(tag), comm);
    }

    // --- nonblocking standard send ---
    template<typename T, typename tag_type=int>
    detail::irequest isend(const T &data, int dest, tag_type tag=0) const {
      static_assert(detail::is_valid_tag<tag_type>::value,
		    "not an enumeration type or underlying enumeration type too large");
      check_dest(dest);
      check_stag(detail::underlying_type<tag_type>::value(tag));
      MPI_Request req;
      MPI_Isend(&data, 1,
		datatype_traits<T>::get_datatype(),
		dest, detail::underlying_type<tag_type>::value(tag), comm, &req);
      return detail::irequest(req);
    }
    
    template<typename T, typename tag_type=int>
    detail::irequest isend(const T *data, const layout<T> &l, int dest, tag_type tag=0) const {
      static_assert(detail::is_valid_tag<tag_type>::value,
		    "not an enumeration type or underlying enumeration type too large");
      check_dest(dest);
      check_stag(detail::underlying_type<tag_type>::value(tag));
      MPI_Request req;
      MPI_Isend(data, 1,
		datatype_traits<layout<T>>::get_datatype(l),
		dest, detail::underlying_type<tag_type>::value(tag), comm, &req);
      return detail::irequest(req);
    }
    
    // --- persistend standard send ---
    template<typename T, typename tag_type=int>
    detail::prequest send_init(const T &data, int dest, tag_type tag=0) const {
      static_assert(detail::is_valid_tag<tag_type>::value,
		    "not an enumeration type or underlying enumeration type too large");
      check_dest(dest);
      check_stag(detail::underlying_type<tag_type>::value(tag));
      MPI_Request req;
      MPI_send_init(&data, 1,
		    datatype_traits<T>::get_datatype(),
		    dest, detail::underlying_type<tag_type>::value(tag), comm, &req);
      return detail::prequest(req);
    }
    
    template<typename T, typename tag_type=int>
    detail::prequest send_init(const T *data, const layout<T> &l, int dest, tag_type tag=0) const {
      static_assert(detail::is_valid_tag<tag_type>::value,
		    "not an enumeration type or underlying enumeration type too large");
      check_dest(dest);
      check_stag(detail::is_valid_tag<tag_type>::value);
      MPI_Request req;
      MPI_Send_init(data, 1,
		    datatype_traits<layout<T>>::get_datatype(l),
		    dest, detail::is_valid_tag<tag_type>::value, comm, &req);
      return detail::prequest(req);
    }
    
    // === buffered send ===
    // --- determine buffer size ---
    template<typename T>
    int bsend_size() const {
      int size;
      MPI_Pack_size(1,
		    datatype_traits<T>::get_datatype(),
		    comm, &size);
      return size+MPI_BSEND_OVERHEAD;
    }
    
    template<typename T>
    int bsend_size(const layout<T> &l) const {
      int size;
      MPI_Pack_size(1,
		    datatype_traits<layout<T>>::get_datatype(l),
		    comm, &size);
      return size+MPI_BSEND_OVERHEAD;
    }
    
    // --- blocking buffered send ---
    template<typename T, typename tag_type=int>
    void bsend(const T &data, int dest, tag_type tag=0) const {
      static_assert(detail::is_valid_tag<tag_type>::value,
		    "not an enumeration type or underlying enumeration type too large");
      check_dest(dest);
      check_stag(detail::is_valid_tag<tag_type>::value);
      MPI_Bsend(&data, 1,
		datatype_traits<T>::get_datatype(),
		dest, detail::is_valid_tag<tag_type>::value, comm);
    }
    
    template<typename T, typename tag_type=int>
    void bsend(const T *data, const layout<T> &l, int dest, tag_type tag=0) const {
      static_assert(detail::is_valid_tag<tag_type>::value,
		    "not an enumeration type or underlying enumeration type too large");
      check_dest(dest);
      check_stag(detail::is_valid_tag<tag_type>::value);
      MPI_Bsend(data, 1,
		datatype_traits<layout<T>>::get_datatype(l),
		dest, detail::is_valid_tag<tag_type>::value, comm);
    }
    
    // --- nonblocking buffered send ---
    template<typename T, typename tag_type=int>
    detail::irequest ibsend(const T &data, int dest, tag_type tag=0) const {
      static_assert(detail::is_valid_tag<tag_type>::value,
		    "not an enumeration type or underlying enumeration type too large");
      check_dest(dest);
      check_stag(detail::underlying_type<tag_type>::value(tag));
      MPI_Request req;
      MPI_Ibsend(&data, 1,
		 datatype_traits<T>::get_datatype(),
		 dest, detail::underlying_type<tag_type>::value(tag), comm, &req);
      return detail::irequest(req);
    }

    template<typename T, typename tag_type=int>
    detail::irequest ibsend(const T *data, const layout<T> &l, int dest, tag_type tag=0) const {
      static_assert(detail::is_valid_tag<tag_type>::value,
		    "not an enumeration type or underlying enumeration type too large");
      check_dest(dest);
      check_stag(detail::underlying_type<tag_type>::value(tag));
      MPI_Request req;
      MPI_Ibsend(data, 1,
		 datatype_traits<layout<T>>::get_datatype(l),
		 dest, detail::underlying_type<tag_type>::value(tag), comm, &req);
      return detail::irequest(req);
    }

    // --- persistent buffered send ---
    template<typename T, typename tag_type=int>
    detail::prequest bsend_init(const T &data, int dest, tag_type tag=0) const {
      check_dest(dest);
      check_stag(detail::underlying_type<tag_type>::value(tag));
      MPI_Request req;
      MPI_Bsend_init(&data, 1,
		     datatype_traits<T>::get_datatype(),
		     dest, detail::underlying_type<tag_type>::value(tag), comm, &req);
      return detail::prequest(req);
    }

    template<typename T, typename tag_type=int>
    detail::prequest bsend_init(const T *data, const layout<T> &l, int dest, tag_type tag=0) const {
      static_assert(detail::is_valid_tag<tag_type>::value,
		    "not an enumeration type or underlying enumeration type too large");
      check_dest(dest);
      check_stag(detail::underlying_type<tag_type>::value(tag));
      MPI_Request req;
      MPI_Bsend_init(data, 1,
		     datatype_traits<layout<T>>::get_datatype(l),
		     dest, detail::underlying_type<tag_type>::value(tag), comm, &req);
      return detail::prequest(req);
    }

    // === synchronous send ===
    // --- blocking synchronous send ---
    template<typename T, typename tag_type=int>
    void ssend(const T &data, int dest, tag_type tag=0) const {
      static_assert(detail::is_valid_tag<tag_type>::value,
		    "not an enumeration type or underlying enumeration type too large");
      check_dest(dest);
      check_stag(detail::underlying_type<tag_type>::value(tag));
      MPI_Ssend(&data, 1,
		datatype_traits<T>::get_datatype(),
		dest, detail::underlying_type<tag_type>::value(tag), comm);
    }

    template<typename T, typename tag_type=int>
    void ssend(const T *data, const layout<T> &l, int dest, tag_type tag=0) const {
      static_assert(detail::is_valid_tag<tag_type>::value,
		    "not an enumeration type or underlying enumeration type too large");
      check_dest(dest);
      check_stag(detail::underlying_type<tag_type>::value(tag));
      MPI_Ssend(data, 1,
		datatype_traits<layout<T>>::get_datatype(l),
		dest, detail::underlying_type<tag_type>::value(tag), comm);
    }

    // --- nonblocking synchronous send ---
    template<typename T, typename tag_type=int>
    detail::irequest issend(const T &data, int dest, tag_type tag=0) const {
      static_assert(detail::is_valid_tag<tag_type>::value,
		    "not an enumeration type or underlying enumeration type too large");
      check_dest(dest);
      check_stag(detail::underlying_type<tag_type>::value(tag));
      MPI_Request req;
      MPI_Issend(&data, 1,
		 datatype_traits<T>::get_datatype(),
		 dest, detail::underlying_type<tag_type>::value(tag), comm, &req);
      return detail::irequest(req);
    }

    template<typename T, typename tag_type=int>
    detail::irequest issend(const T *data, const layout<T> &l, int dest, tag_type tag=0) const {
      static_assert(detail::is_valid_tag<tag_type>::value,
		    "not an enumeration type or underlying enumeration type too large");
      check_dest(dest);
      check_stag(detail::underlying_type<tag_type>::value(tag));
      MPI_Request req;
      MPI_Issend(data, 1,
		 datatype_traits<layout<T>>::get_datatype(l),
		 dest, detail::underlying_type<tag_type>::value(tag), comm, &req);
      return detail::irequest(req);
    }

    // --- persistent synchronous send ---
    template<typename T, typename tag_type=int>
    detail::prequest ssend_init(const T &data, int dest, tag_type tag=0) const {
      static_assert(detail::is_valid_tag<tag_type>::value,
		    "not an enumeration type or underlying enumeration type too large");
      check_dest(dest);
      check_stag(detail::underlying_type<tag_type>::value(tag));
      MPI_Request req;
      MPI_Ssend_init(&data, 1,
		     datatype_traits<T>::get_datatype(),
		     dest, detail::underlying_type<tag_type>::value(tag), comm, &req);
      return detail::prequest(req);
    }

    template<typename T, typename tag_type=int>
    detail::prequest ssend_init(const T *data, const layout<T> &l, int dest, tag_type tag=0) const {
      static_assert(detail::is_valid_tag<tag_type>::value,
		    "not an enumeration type or underlying enumeration type too large");
      check_dest(dest);
      check_stag(detail::underlying_type<tag_type>::value(tag));
      MPI_Request req;
      MPI_Ssend_init(data, 1,
		     datatype_traits<layout<T>>::get_datatype(l),
		     dest, detail::underlying_type<tag_type>::value(tag), comm, &req);
      return detail::prequest(req);
    }

    // === ready send ===
    // --- blocking ready send ---
    template<typename T, typename tag_type=int>
    void rsend(const T &data, int dest, tag_type tag=0) const {
      static_assert(detail::is_valid_tag<tag_type>::value,
		    "not an enumeration type or underlying enumeration type too large");
      check_dest(dest);
      check_stag(detail::underlying_type<tag_type>::value(tag));
      MPI_Rsend(&data, 1,
		datatype_traits<T>::get_datatype(),
		dest, detail::underlying_type<tag_type>::value(tag), comm);
    }

    template<typename T, typename tag_type=int>
    void rsend(const T *data, const layout<T> &l, int dest, tag_type tag=0) const {
      static_assert(detail::is_valid_tag<tag_type>::value,
		    "not an enumeration type or underlying enumeration type too large");
      check_dest(dest);
      check_stag(detail::underlying_type<tag_type>::value(tag));
      MPI_Rsend(data, 1,
		datatype_traits<layout<T>>::get_datatype(l),
		dest, detail::underlying_type<tag_type>::value(tag), comm);
    }

    // --- nonblocking ready send ---
    template<typename T, typename tag_type=int>
    detail::irequest irsend(const T &data, int dest, tag_type tag=0) const {
      static_assert(detail::is_valid_tag<tag_type>::value,
		    "not an enumeration type or underlying enumeration type too large");
      check_dest(dest);
      check_stag(detail::underlying_type<tag_type>::value(tag));
      MPI_Request req;
      MPI_Irsend(&data, 1,
		 datatype_traits<T>::get_datatype(),
		 dest, detail::underlying_type<tag_type>::value(tag), comm, &req);
      return detail::irequest(req);
    }

    template<typename T, typename tag_type=int>
    detail::irequest irsend(const T *data, const layout<T> &l, int dest, tag_type tag=0) const {
      static_assert(detail::is_valid_tag<tag_type>::value,
		    "not an enumeration type or underlying enumeration type too large");
      check_dest(dest);
      check_stag(detail::underlying_type<tag_type>::value(tag));
      MPI_Request req;
      MPI_Irsend(data, 1,
		 datatype_traits<layout<T>>::get_datatype(l),
		 dest, detail::underlying_type<tag_type>::value(tag), comm, &req);
      return detail::irequest(req);
    }

    // --- persistent ready send ---
    template<typename T, typename tag_type=int>
    detail::prequest rsend_init(const T &data, int dest, tag_type tag=0) const {
      static_assert(detail::is_valid_tag<tag_type>::value,
		    "not an enumeration type or underlying enumeration type too large");
      check_dest(dest);
      check_stag(detail::underlying_type<tag_type>::value(tag));
      MPI_Request req;
      MPI_Rsend_init(&data, 1,
		     datatype_traits<T>::get_datatype(),
		     dest, detail::underlying_type<tag_type>::value(tag), comm, &req);
      return detail::prequest(req);
    }

    template<typename T, typename tag_type=int>
    detail::prequest rsend_init(const T *data, const layout<T> &l, int dest, tag_type tag=0) const {
      static_assert(detail::is_valid_tag<tag_type>::value,
		    "not an enumeration type or underlying enumeration type too large");
      check_dest(dest);
      check_stag(detail::underlying_type<tag_type>::value(tag));
      MPI_Request req;
      MPI_Rsend_init(data, 1,
		     datatype_traits<layout<T>>::get_datatype(l),
		     dest, detail::underlying_type<tag_type>::value(tag), comm, &req);
      return detail::prequest(req);
    }

    // === receive ===
    // --- blocking receive ---
    template<typename T, typename tag_type=int>
    status recv(T &data, int source, tag_type tag=0) const {
      static_assert(detail::is_valid_tag<tag_type>::value,
		    "not an enumeration type or underlying enumeration type too large");
      check_source(source);
      check_rtag(detail::underlying_type<tag_type>::value(tag));
      status s;
      MPI_Recv(&data, 1,
      	       datatype_traits<T>::get_datatype(),
      	       source, detail::underlying_type<tag_type>::value(tag), comm, reinterpret_cast<MPI_Status *>(&s));
      return s;
    }

    template<typename T, typename tag_type=int>
    status recv(T *data, const layout<T> &l, int source, tag_type tag=0) const {
      static_assert(detail::is_valid_tag<tag_type>::value,
		    "not an enumeration type or underlying enumeration type too large");
      check_source(source);
      check_rtag(detail::underlying_type<tag_type>::value(tag));
      status s;
      MPI_Recv(data, 1,
      	       datatype_traits<layout<T>>::get_datatype(l),
      	       source, detail::underlying_type<tag_type>::value(tag), comm, reinterpret_cast<MPI_Status *>(&s));
      return s;
    }
    
    // --- nonblocking receive ---
    template<typename T, typename tag_type=int>
    detail::irequest irecv(T &data, int source, tag_type tag=0) const {
      static_assert(detail::is_valid_tag<tag_type>::value,
		    "not an enumeration type or underlying enumeration type too large");
      check_source(source);
      check_rtag(detail::underlying_type<tag_type>::value(tag));
      MPI_Request req;
      MPI_Irecv(&data, 1,
		datatype_traits<T>::get_datatype(),
		source, detail::underlying_type<tag_type>::value(tag), comm, &req);
      return detail::irequest(req);
    }

    template<typename T, typename tag_type=int>
    detail::irequest irecv(T *data, const layout<T> &l, int source, tag_type tag=0) const {
      static_assert(detail::is_valid_tag<tag_type>::value,
		    "not an enumeration type or underlying enumeration type too large");
      check_source(source);
      check_rtag(detail::underlying_type<tag_type>::value(tag));
      MPI_Request req;
      MPI_Irecv(data, 1,
		datatype_traits<layout<T>>::get_datatype(l),
		source, detail::underlying_type<tag_type>::value(tag), comm, &req);
      return detail::irequest(req);
    }

    // --- persistent receive ---
    template<typename T, typename tag_type=int>
    detail::prequest recv_init(T &data, int source, tag_type tag=0) const {
      static_assert(detail::is_valid_tag<tag_type>::value,
		    "not an enumeration type or underlying enumeration type too large");
      check_source(source);
      check_rtag(detail::underlying_type<tag_type>::value(tag));
      MPI_Request req;
      MPI_Recv_init(&data, 1,
		    datatype_traits<T>::get_datatype(),
		    source, detail::underlying_type<tag_type>::value(tag), comm, &req);
      return detail::prequest(req);
    }

    template<typename T, typename tag_type=int>
    detail::prequest recv_init(T *data, const layout<T> &l, int source, tag_type tag=0) const {
      static_assert(detail::is_valid_tag<tag_type>::value,
		    "not an enumeration type or underlying enumeration type too large");
      check_source(source);
      check_rtag(detail::underlying_type<tag_type>::value(tag));
      MPI_Request req;
      MPI_Recv_init(data, 1,
		    datatype_traits<layout<T>>::get_datatype(l),
		    source, detail::underlying_type<tag_type>::value(tag), comm, &req);
      return detail::prequest(req);
    }

    // === probe ===
    // --- blocking probe ---
    template<typename tag_type=int>
    status probe(int source, tag_type tag=0) const {
      static_assert(detail::is_valid_tag<tag_type>::value,
		    "not an enumeration type or underlying enumeration type too large");
      check_source(source);
      check_rtag(detail::underlying_type<tag_type>::value(tag));
      status s;
      MPI_Probe(source, detail::underlying_type<tag_type>::value(tag),
		comm, reinterpret_cast<MPI_Status *>(&s));
      return s;
    }

    // --- nonblocking probe ---
    template<typename T, typename tag_type=int>
    std::pair<bool, status> iprobe(int source, tag_type tag=0) const {
      static_assert(detail::is_valid_tag<tag_type>::value,
		    "not an enumeration type or underlying enumeration type too large");
      check_source(source);
      check_rtag(detail::underlying_type<tag_type>::value(tag));
      int result;
      status s;
      MPI_Iprobe(source, detail::underlying_type<tag_type>::value(tag),
		 comm, &result, reinterpret_cast<MPI_Status *>(&s));
      return std::make_pair(static_cast<bool>(result), s);
    }
    
    // === matching probe ===
    // --- blocking matching probe ---
    template<typename tag_type=int>
    std::pair<message, status> mprobe(int source, tag_type tag=0) const {
      static_assert(detail::is_valid_tag<tag_type>::value,
		    "not an enumeration type or underlying enumeration type too large");
      check_source(source);
      check_rtag(detail::underlying_type<tag_type>::value(tag));
      status s;
      message m;
      MPI_Mprobe(source, detail::underlying_type<tag_type>::value(tag),
		 comm, &m, reinterpret_cast<MPI_Status *>(&s));
      return std::make_pair(m, s);
    }
    
    // --- nonblocking matching probe ---
    template<typename tag_type=int>
    std::tuple<bool, message, status> improbe(int source, tag_type tag=0) const {
      static_assert(detail::is_valid_tag<tag_type>::value,
		    "not an enumeration type or underlying enumeration type too large");
      check_source(source);
      check_rtag(detail::underlying_type<tag_type>::value(tag));
      int result;
      status s;
      message m;
      MPI_Improbe(source, detail::underlying_type<tag_type>::value(tag),
		  comm, &result, &m, reinterpret_cast<MPI_Status *>(&s));
      return std::make_tuple(static_cast<bool>(result), m, s);
    }

    // === matching receive ===
    // --- blocking matching receive ---

    // --- nonblocking matching receive ---

    // === send and receive ===
    // --- send and receive ---
    template<typename T, typename tag_type=int>
    status sendrecv(const T &senddata, int dest, tag_type sendtag,
		    T &recvdata, int source, tag_type recvtag) const {
      static_assert(detail::is_valid_tag<tag_type>::value,
		    "not an enumeration type or underlying enumeration type too large");
      check_dest(dest);
      check_source(source);
      check_stag(detail::underlying_type<tag_type>::value(sendtag));
      check_rtag(detail::underlying_type<tag_type>::value(recvtag));
      status s;
      MPI_Sendrecv(&senddata, 1,
		   datatype_traits<T>::get_datatype(), dest, detail::underlying_type<tag_type>::value(sendtag),
		   &recvdata, 1,
		   datatype_traits<T>::get_datatype(), source, detail::underlying_type<tag_type>::value(recvtag),
		   comm, reinterpret_cast<MPI_Status *>(&s));
      return s;
    }
    
    template<typename T, typename tag_type=int>
    status sendrecv(const T *senddata, const layout<T> &sendl, int dest, tag_type sendtag,
		    T *recvdata, const layout<T> &recvl, int source, tag_type recvtag) const {
      static_assert(detail::is_valid_tag<tag_type>::value,
		    "not an enumeration type or underlying enumeration type too large");
      check_dest(dest);
      check_source(source);
      check_stag(detail::underlying_type<tag_type>::value(sendtag));
      check_rtag(detail::underlying_type<tag_type>::value(recvtag));
      status s;
      MPI_Sendrecv(senddata, 1,
		   datatype_traits<layout<T>>::get_datatype(sendl), dest, detail::underlying_type<tag_type>::value(sendtag),
		   recvdata, 1,
		   datatype_traits<layout<T>>::get_datatype(recvl), source, detail::underlying_type<tag_type>::value(recvtag),
		   comm, reinterpret_cast<MPI_Status *>(&s));
      return s;
    }
    
    // --- send, receive and replace ---
    template<typename T, typename tag_type=int>
    status sendrecv_replace(T &data,
			    int dest, tag_type sendtag, int source, tag_type recvtag) const {
      static_assert(detail::is_valid_tag<tag_type>::value,
		    "not an enumeration type or underlying enumeration type too large");
      check_dest(dest);
      check_source(source);
      check_stag(detail::underlying_type<tag_type>::value(sendtag));
      check_rtag(detail::underlying_type<tag_type>::value(recvtag));
      status s;
      MPI_Sendrecv_replace(&data, 1,
			   datatype_traits<T>::get_datatype(),
			   dest, detail::underlying_type<tag_type>::value(sendtag),
			   source, detail::underlying_type<tag_type>::value(recvtag),
			   comm, reinterpret_cast<MPI_Status *>(&s));
      return s;
    }
    
    template<typename T, typename tag_type=int>
    status sendrecv_replace(T *data, const layout<T> &l,
			    int dest, tag_type sendtag, int source, tag_type recvtag) const {
      static_assert(detail::is_valid_tag<tag_type>::value,
		    "not an enumeration type or underlying enumeration type too large");
      check_dest(dest);
      check_source(source);
      check_stag(detail::underlying_type<tag_type>::value(sendtag));
      check_rtag(detail::underlying_type<tag_type>::value(recvtag));
      status s;
      MPI_Sendrecv_replace(data, 1,
			   datatype_traits<layout<T>>::get_datatype(l),
			   dest, detail::underlying_type<tag_type>::value(sendtag),
			   source, detail::underlying_type<tag_type>::value(recvtag),
			   comm, reinterpret_cast<MPI_Status *>(&s));
      return s;
    }
    
    // === collective ==================================================
    // === barrier ===
    // --- blocking barrier ---
    void barrier() const {
      MPI_Barrier(comm);
    }
    
    // --- nonblocking barrier ---
    detail::irequest ibarrier() const {
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
    detail::irequest ibcast(int root, T &data) const {
      check_root(root);
      MPI_Request req;
      MPI_Ibcast(&data, 1, datatype_traits<T>::get_datatype(), root, comm, &req);
      return detail::irequest(req);
    }
    
    template<typename T>
    detail::irequest ibcast(int root, T *data, const layout<T> &l) const {
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
      MPI_Gather(&senddata, 1, datatype_traits<T>::get_datatype(),
		 recvdata, 1, datatype_traits<T>::get_datatype(),
		 root, comm);
    }
    
    template<typename T>
    void gather(int root,
		const T *senddata, const layout<T> &sendl,
		T *recvdata, const layout<T> &recvl) const {
      check_root(root);
      MPI_Gather(senddata, 1, datatype_traits<layout<T>>::get_datatype(sendl),
		 recvdata, 1, datatype_traits<layout<T>>::get_datatype(recvl),
		 root, comm);
    }
    
    // --- nonblocking gather ---
    template<typename T>
    detail::irequest igather(int root, const T &senddata, T *recvdata) const {
      check_root(root);
      MPI_Request req;
      MPI_Igather(&senddata, 1, datatype_traits<T>::get_datatype(),
		  recvdata, 1, datatype_traits<T>::get_datatype(),
		  root, comm, &req);
      return detail::irequest(req);
    }
    
    template<typename T>
    detail::irequest igather(int root,
			     const T *senddata, const layout<T> &sendl,
			     T *recvdata, const layout<T> &recvl) const {
      check_root(root);
      MPI_Request req;
      MPI_Igather(senddata, 1, datatype_traits<layout<T>>::get_datatype(sendl),
		  recvdata, 1, datatype_traits<layout<T>>::get_datatype(recvl),
		  root, comm, &req);
      return detail::irequest(req);
    }
    
    // --- blocking gather, non-root variant ---
    template<typename T>
    void gather(int root, const T &senddata) const {
      check_nonroot(root);
      MPI_Gather(&senddata, 1, datatype_traits<T>::get_datatype(),
		 0, 0, MPI_DATATYPE_NULL,
		 root, comm);
    }
    
    template<typename T>
    void gather(int root,
		const T *senddata, const layout<T> &sendl) const {
      check_nonroot(root);
      MPI_Gather(senddata, 1, datatype_traits<layout<T>>::get_datatype(sendl),
		 0, 0, MPI_DATATYPE_NULL,
		 root, comm);
    }
    
    // --- nonblocking gather, non-root variant ---
    template<typename T>
    detail::irequest igather(int root, const T &senddata) const {
      check_nonroot(root);
      MPI_Request req;
      MPI_Igather(&senddata, 1, datatype_traits<T>::get_datatype(),
		  0, 0, MPI_DATATYPE_NULL,
		  root, comm, &req);
      return detail::irequest(req);
    }
    
    template<typename T>
    detail::irequest igather(int root,
			     const T *senddata, const layout<T> &sendl) const {
      check_nonroot(root);
      MPI_Request req;
      MPI_Igather(senddata, 1, datatype_traits<layout<T>>::get_datatype(sendl),
		  0, 0, MPI_DATATYPE_NULL,
		  root, comm, &req);
      return detail::irequest(req);
    }
    
    // === root gets varying amount of data from each rank and stores in noncontiguous memory
    // --- blocking gather ---
    template<typename T>
    void gatherv(int root,
		 const T *senddata, const layout<T> &sendl,
		 T *recvdata, const layouts<T> &recvls, const displacements &recvdispls) const {
      check_root(root);
      check_size(recvls);
      check_size(recvdispls);
      int N(size());
      displacements senddispls(N);
      layouts<T> sendls(N);
      sendls[root]=sendl;
      if (rank()==root)
	alltoallv(senddata, sendls, senddispls,
		  recvdata, recvls, recvdispls);
      else
	alltoallv(senddata, sendls, senddispls,
		  recvdata, mpl::layouts<T>(N), recvdispls);
    }
    
    template<typename T>
    void gatherv(int root,
		 const T *senddata, const layout<T> &sendl,
		 T *recvdata, const layouts<T> &recvls) const {
      gatherv(root, senddata, sendl, recvdata, recvls, displacements(size()));
    }
    
    // --- nonblocking gather ---
    template<typename T>
    detail::irequest igatherv(int root,
			      const T *senddata, const layout<T> &sendl,
			      T *recvdata, const layouts<T> &recvls, const displacements &recvdispls) const {
      check_root(root);
      check_size(recvls);
      check_size(recvdispls);
      int N(size());
      displacements senddispls(N);
      layouts<T> sendls(N);
      sendls[root]=sendl;
      if (rank()==root)
	return ialltoallv(senddata, sendls, senddispls,
			  recvdata, recvls, recvdispls);
      else
	return ialltoallv(senddata, sendls, senddispls,
			  recvdata, mpl::layouts<T>(N), recvdispls);
    }
    
    template<typename T>
    detail::irequest igatherv(int root,
			      const T *senddata, const layout<T> &sendl,
			      T *recvdata, const layouts<T> &recvls) const {
      return igatherv(root, senddata, sendl, recvdata, recvls, displacements(size()));
    }
    
    // --- blocking gather, non-root variant ---
    template<typename T>
    void gatherv(int root,
		 const T *senddata, const layout<T> &sendl) const {
      check_nonroot(root);
      int N(size());
      displacements sendrecvdispls(N);
      layouts<T> sendls(N);
      sendls[root]=sendl;
      alltoallv(senddata, sendls, sendrecvdispls,
		static_cast<T *>(nullptr), mpl::layouts<T>(N), sendrecvdispls);
    }
    
    // --- nonblocking gather, non-root variant ---
    template<typename T>
    detail::irequest igatherv(int root,
			      const T *senddata, const layout<T> &sendl) const {
      check_nonroot(root);
      int N(size());
      displacements sendrecvdispls(N);
      layouts<T> sendls(N);
      sendls[root]=sendl;
      return ialltoallv(senddata, sendls, sendrecvdispls,
			static_cast<T *>(nullptr), mpl::layouts<T>(N), sendrecvdispls);
    }
    
    // === allgather ===
    // === get a single value from each rank and stores in contiguous memory
    // --- blocking allgather ---
    template<typename T>
    void allgather(const T &senddata, T *recvdata) const {
      MPI_Allgather(&senddata, 1, datatype_traits<T>::get_datatype(),
		    recvdata, 1, datatype_traits<T>::get_datatype(),
		    comm);
    }
    
    template<typename T>
    void allgather(const T *senddata, const layout<T> &sendl,
		   T *recvdata, const layout<T> &recvl) const {
      MPI_Allgather(senddata, 1, datatype_traits<layout<T>>::get_datatype(sendl),
		    recvdata, 1, datatype_traits<layout<T>>::get_datatype(recvl),
		    comm);
    }
    
    // --- nonblocking allgather ---
    template<typename T>
    detail::irequest iallgather(const T &senddata, T *recvdata) const {
      MPI_Request req;
      MPI_Iallgather(&senddata, 1, datatype_traits<T>::get_datatype(),
		     recvdata, 1, datatype_traits<T>::get_datatype(),
		     comm, &req);
      return detail::irequest(req);
    }
    
    template<typename T>
    detail::irequest iallgather(const T *senddata, const layout<T> &sendl,
				T *recvdata, const layout<T> &recvl) const {
      MPI_Request req;
      MPI_Iallgather(senddata, 1, datatype_traits<layout<T>>::get_datatype(sendl),
		     recvdata, 1, datatype_traits<layout<T>>::get_datatype(recvl),
		     comm, &req);
      return detail::irequest(req);
    }
    
    // === get varying amount of data from each rank and stores in noncontiguous memory
    // --- blocking allgather ---
    template<typename T>
    void allgatherv(const T *senddata, const layout<T> &sendl,
		    T *recvdata, const layouts<T> &recvls, const displacements &recvdispls) const {
      check_size(recvls);
      check_size(recvdispls);
      int N(size());
      displacements senddispls(N);
      layouts<T> sendls(N, sendl);
      alltoallv(senddata, sendls, senddispls,
		recvdata, recvls, recvdispls);
    }
    
    template<typename T>
    void allgatherv(const T *senddata, const layout<T> &sendl,
		    T *recvdata, const layouts<T> &recvls) const {
      allgatherv(senddata, sendl,
		 recvdata, recvls, displacements(size()));
    }
    
    // --- nonblocking allgather ---
    template<typename T>
    detail::irequest iallgatherv(const T *senddata, const layout<T> &sendl,
				 T *recvdata, const layouts<T> &recvls, const displacements &recvdispls) const {
      check_size(recvls);
      check_size(recvdispls);
      int N(size());
      displacements senddispls(N);
      layouts<T> sendls(N, sendl);
      return ialltoallv(senddata, sendls, senddispls,
			recvdata, recvls, recvdispls);
    }
    
    template<typename T>
    detail::irequest iallgatherv(const T *senddata, const layout<T> &sendl,
				 T *recvdata, const layouts<T> &recvls) const {
      return iallgatherv(senddata, sendl,
			 recvdata, recvls, displacements(size()));
    }
    
    // === scatter ===
    // === root sends a single value from contiguous memory to each rank
    // --- blocking scatter ---
    template<typename T>
    void scatter(int root, const T *senddata, T &recvdata) const {
      check_root(root);
      MPI_Scatter(senddata, 1, datatype_traits<T>::get_datatype(),
		  &recvdata, 1, datatype_traits<T>::get_datatype(),
		  root, comm);
    }
    
    template<typename T>
    void scatter(int root,
		 const T *senddata, const layout<T> &sendl,
		 T *recvdata, const layout<T> &recvl) const {
      check_root(root);
      MPI_Scatter(senddata, 1, datatype_traits<layout<T>>::get_datatype(sendl),
		  recvdata, 1, datatype_traits<layout<T>>::get_datatype(recvl),
		  root, comm);
    }
    
    // --- nonblocking scatter ---
    template<typename T>
    detail::irequest iscatter(int root, const T *senddata, T &recvdata) const {
      check_root(root);
      MPI_Request req;
      MPI_Iscatter(senddata, 1, datatype_traits<T>::get_datatype(),
		   &recvdata, 1, datatype_traits<T>::get_datatype(),
		  root, comm, &req);
      return detail::irequest(req);
    }
    
    template<typename T>
    detail::irequest iscatter(int root,
			      const T *senddata, const layout<T> &sendl,
			      T *recvdata, const layout<T> &recvl) const {
      check_root(root);
      MPI_Request req;
      MPI_Iscatter(senddata, 1, datatype_traits<layout<T>>::get_datatype(sendl),
		   recvdata, 1, datatype_traits<layout<T>>::get_datatype(recvl),
		   root, comm, &req);
      return detail::irequest(req);
    }
    
    // --- blocking scatter, non-root variant ---
    template<typename T>
    void scatter(int root, T &recvdata) const {
      check_nonroot(root);
      MPI_Scatter(0, 0, MPI_DATATYPE_NULL,
		  &recvdata, 1, datatype_traits<T>::get_datatype(),
		  root, comm);
    }
    
    template<typename T>
    void scatter(int root,
		 T *recvdata, const layout<T> &recvl) const {
      check_root(root);
      MPI_Scatter(0, 0, MPI_DATATYPE_NULL,
		  recvdata, 1, datatype_traits<layout<T>>::get_datatype(recvl),
		  root, comm);
    }
    
    // --- nonblocking scatter, non-root variant ---
    template<typename T>
    detail::irequest iscatter(int root, T &recvdata) const {
      check_nonroot(root);
      MPI_Request req;
      MPI_Iscatter(0, 0, MPI_DATATYPE_NULL,
		  &recvdata, 1, datatype_traits<T>::get_datatype(),
		  root, comm, &req);
      return detail::irequest(req);
    }
    
    template<typename T>
    detail::irequest iscatter(int root,
			      T *recvdata, const layout<T> &recvl) const {
      check_nonroot(root);
      MPI_Request req;
      MPI_Iscatter(0, 0, MPI_DATATYPE_NULL,
		   recvdata, 1, datatype_traits<layout<T>>::get_datatype(recvl),
		   root, comm, &req);
      return detail::irequest(req);
    }
    
    // === root sends varying amount of data from noncontiguous memory to each rank
    // --- blocking scatter ---
    template<typename T>
    void scatterv(int root,
		  const T *senddata, const layouts<T> &sendls, const displacements &senddispls,
		  T *recvdata, const layout<T> &recvl) const {
      check_root(root);
      check_size(sendls);
      check_size(senddispls);
      int N(size());
      displacements recvdispls(N);
      layouts<T> recvls(N);
      recvls[root]=recvl;
      if (rank()==root)
	alltoallv(senddata, sendls, senddispls,
		  recvdata, recvls, recvdispls);
      else
	alltoallv(senddata, sendls, senddispls,
		  recvdata, mpl::layouts<T>(N), recvdispls);
    }
    
    template<typename T>
    void scatterv(int root,
		  const T *senddata, const layouts<T> &sendls,
		  T *recvdata, const layout<T> &recvl) const {
      scatterv(root, senddata, sendls, displacements(size()),
	       recvdata, recvl);
    }
    
    // --- nonblocking scatter ---
    template<typename T>
    detail::irequest iscatterv(int root,
			       const T *senddata, const layouts<T> &sendls, const displacements &senddispls,
			       T *recvdata, const layout<T> &recvl) const {
      check_root(root);
      check_size(sendls);
      check_size(senddispls);
      int N(size());
      displacements recvdispls(N);
      layouts<T> recvls(N);
      recvls[root]=recvl;
      if (rank()==root)
	return ialltoallv(senddata, sendls, senddispls,
			  recvdata, recvls, recvdispls);
      else
	return ialltoallv(senddata, sendls, senddispls,
			  recvdata, mpl::layouts<T>(N), recvdispls);
    }
    
    template<typename T>
    detail::irequest iscatterv(int root,
			       const T *senddata, const layouts<T> &sendls,
			       T *recvdata, const layout<T> &recvl) const {
      return iscatterv(root, senddata, sendls, displacements(size()),
		       recvdata, recvl);
    }
    
    // --- blocking scatter, non-root variant ---
    template<typename T>
    void scatterv(int root,
		  T *recvdata, const layout<T> &recvl) const {
      check_root(root);
      int N(size());
      displacements sendrecvdispls(N);
      layouts<T> recvls(N);
      recvls[root]=recvl;
      alltoallv(static_cast<const T *>(nullptr),  mpl::layouts<T>(N), sendrecvdispls,
		recvdata, recvls, sendrecvdispls);
    }
    
    // --- nonblocking scatter, non-root variant ---
    template<typename T>
    detail::irequest iscatterv(int root,
			       T *recvdata, const layout<T> &recvl) const {
      check_root(root);
      int N(size());
      displacements sendrecvdispls(N);
      layouts<T> recvls(N);
      recvls[root]=recvl;
      return ialltoallv(static_cast<const T *>(nullptr),  mpl::layouts<T>(N), sendrecvdispls,
			recvdata, recvls, sendrecvdispls);
    }
    
    // === all-to-all ===
    // === each rank sends a single value to each rank
    // --- blocking all-to-all ---
    template<typename T>
    void alltoall(const T *senddata, T *recvdata) const {
      MPI_Alltoall(senddata, 1, datatype_traits<T>::get_datatype(),
		   recvdata, 1, datatype_traits<T>::get_datatype(),
		   comm);
    }
    
    template<typename T>
    void alltoall(const T *senddata, const layout<T> &sendl,
		  T *recvdata, const layout<T> &recvl) const {
      MPI_Alltoall(senddata, 1, datatype_traits<layout<T>>::get_datatype(),
		   recvdata, 1, datatype_traits<layout<T>>::get_datatype(),
		   comm);
    }
    
    // --- nonblocking all-to-all ---
    template<typename T>
    detail::irequest ialltoall(const T *senddata, T *recvdata) const {
      MPI_Request req;
      MPI_Ialltoall(senddata, 1, datatype_traits<T>::get_datatype(),
		    recvdata, 1, datatype_traits<T>::get_datatype(),
		    comm, &req);
      return detail::irequest(req);
    }
    
    template<typename T>
    detail::irequest ialltoall(const T *senddata, const layout<T> &sendl,
			       T *recvdata, const layout<T> &recvl) const {
      MPI_Request req;
      MPI_Ialltoall(senddata, 1, datatype_traits<layout<T>>::get_datatype(),
		    recvdata, 1, datatype_traits<layout<T>>::get_datatype(),
		    comm, &req);
      return detail::irequest(req);
    }
    
    // --- blocking all-to-all, in place ---
    template<typename T>
    void alltoall(T *recvdata) const {
      MPI_Alltoall(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
		   recvdata, 1, datatype_traits<T>::get_datatype(),
		   comm);
    }
    
    template<typename T>
    void alltoall(T *recvdata, const layout<T> &recvl) const {
      MPI_Alltoall(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
		   recvdata, 1, datatype_traits<layout<T>>::get_datatype(),
		   comm);
    }
    
    // --- nonblocking all-to-all, in place ---
    template<typename T>
    detail::irequest ialltoall(T *recvdata) const {
      MPI_Request req;
      MPI_Ialltoall(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
		    recvdata, 1, datatype_traits<T>::get_datatype(),
		    comm, &req);
      return detail::irequest(req);
    }
    
    template<typename T>
    detail::irequest ialltoall(T *recvdata, const layout<T> &recvl) const {
      MPI_Request req;
      MPI_Ialltoall(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
		    recvdata, 1, datatype_traits<layout<T>>::get_datatype(),
		    comm, &req);
      return detail::irequest(req);
    }

    // === each rank sends a varying number of values to each rank with possibly different layouts
    // --- blocking all-to-all ---
    template<typename T>
    void alltoallv(const T *senddata, const layouts<T> &sendl, const displacements &senddispls,
		   T *recvdata, const layouts<T> &recvl, const displacements &recvdispls) const {
      check_size(senddispls);
      check_size(sendl);
      check_size(recvdispls);
      check_size(recvl);
      std::vector<int> counts(recvl.size(), 1);
      std::vector<int> senddispls_int(senddispls.begin(), senddispls.end());
      std::vector<int> recvdispls_int(recvdispls.begin(), recvdispls.end());
      MPI_Alltoallw(senddata, counts.data(), senddispls_int.data(), reinterpret_cast<const MPI_Datatype *>(sendl()),
	            recvdata, counts.data(), recvdispls_int.data(), reinterpret_cast<const MPI_Datatype *>(recvl()),
	            comm);
    }

    template<typename T>
    void alltoallv(const T *senddata, const layouts<T> &sendl,
		   T *recvdata, const layouts<T> &recvl) const {
      displacements sendrecvdispls(size());
      alltoallv(senddata, sendl, sendrecvdispls,
		recvdata, recvl, sendrecvdispls);
    }

    // --- non-blocking all-to-all ---
    template<typename T>
    detail::irequest ialltoallv(const T *senddata, const layouts<T> &sendl, const displacements &senddispls,
	                        T *recvdata, const layouts<T> &recvl, const displacements &recvdispls) const {
      check_size(senddispls);
      check_size(sendl);
      check_size(recvdispls);
      check_size(recvl);
      std::vector<int> counts(recvl.size(), 1);
      std::vector<int> senddispls_int(senddispls.begin(), senddispls.end());
      std::vector<int> recvdispls_int(recvdispls.begin(), recvdispls.end());
      MPI_Request req;
      MPI_Ialltoallw(senddata, counts.data(), senddispls_int.data(), reinterpret_cast<const MPI_Datatype *>(sendl()),
	 	     recvdata, counts.data(), recvdispls_int.data(), reinterpret_cast<const MPI_Datatype *>(recvl()),
		     comm, &req);
      return detail::irequest(req);
    }

    template<typename T>
    detail::irequest ialltoallv(const T *senddata, const layouts<T> &sendl,
                                T *recvdata, const layouts<T> &recvl) const {
      displacements sendrecvdispls(size());
      return ialltoallv(senddata, sendl, sendrecvdispls,
                        recvdata, recvl, sendrecvdispls);
    }

    // --- blocking all-to-all, in place ---
    template<typename T>
    void alltoallv(T *recvdata, const layouts<T> &recvl, const displacements &recvdispls) const {
      check_size(recvdispls);
      check_size(recvl);
      std::vector<int> counts(recvl.size(), 1);
      std::vector<int> recvdispls_int(recvdispls.begin(), recvdispls.end());
      MPI_Alltoallw(MPI_IN_PLACE, 0, 0, 0,
		    recvdata, counts.data(), recvdispls_int.data(), reinterpret_cast<const MPI_Datatype *>(recvl()),
	            comm);
    }
    
    template<typename T>
    void alltoallv(T *recvdata, const layouts<T> &recvl) const {
      alltoallv(recvdata, recvl, displacements(size()));
    }
    
    // --- non-blocking all-to-all, in place ---
    template<typename T>
    detail::irequest ialltoallv(T *recvdata, const layouts<T> &recvl, const displacements &recvdispls) const {
      check_size(recvdispls);
      check_size(recvl);
      std::vector<int> counts(recvl.size(), 1);
      std::vector<int> recvdispls_int(recvdispls.begin(), recvdispls.end());
      MPI_Request req;
      MPI_Ialltoallw(MPI_IN_PLACE, 0, 0, 0,
		     recvdata, counts.data(), recvdispls_int.data(), reinterpret_cast<const MPI_Datatype *>(recvl()),
		     comm, &req);
      return detail::irequest(req);
    }
    
    template<typename T>
    detail::irequest ialltoallv(T *recvdata, const layouts<T> &recvl) const {
      return ialltoallv(recvdata, recvl, displacements(size()));
    }
    
    // === reduce ===
    // --- blocking reduce ---
    template<typename T, typename F>
    void reduce(F f, int root,
		const T &senddata, T &recvdata) const {
      check_root(root);
      detail::get_op<T, F>().f=&f;
      MPI_Reduce(&senddata, &recvdata, 1,
		 datatype_traits<T>::get_datatype(), detail::get_op<T, F>().mpi_op, root,
		 comm);
    }
    
    template<typename T, typename F>
    void reduce(F f, int root,
		const T *senddata, T *recvdata, const contiguous_layout<T> &l) const {
      check_root(root);
      detail::get_op<T, F>().f=&f;
      MPI_Reduce(senddata, recvdata, l.size(),
		 datatype_traits<T>::get_datatype(), detail::get_op<T, F>().mpi_op, root,
		 comm);
    }
    
    // --- non-blocking reduce ---
    template<typename T, typename F>
    detail::irequest ireduce(F f, int root,
			     const T &senddata, T &recvdata) const {
      check_root(root);
      detail::get_op<T, F>().f=&f;
      MPI_Request req;
      MPI_Ireduce(&senddata, &recvdata, 1,
		  datatype_traits<T>::get_datatype(), detail::get_op<T, F>().mpi_op, root,
		  comm, &req);
      return detail::irequest(req);
    }
    
    template<typename T, typename F>
    detail::irequest ireduce(F f, int root,
			     const T *senddata, T *recvdata, const contiguous_layout<T> &l) const {
      check_root(root);
      detail::get_op<T, F>().f=&f;
      MPI_Request req;
      MPI_Ireduce(senddata, recvdata, l.size(),
		  datatype_traits<T>::get_datatype(), detail::get_op<T, F>().mpi_op, root,
		  comm, &req);
      return detail::irequest(req);
    }
    
    // --- blocking reduce, in place ---
    template<typename T, typename F>
    void reduce(F f, int root,
		T &sendrecvdata) const {
      check_root(root);
      detail::get_op<T, F>().f=&f;
      if (rank()==root)
	MPI_Reduce(MPI_IN_PLACE, &sendrecvdata, 1,
		   datatype_traits<T>::get_datatype(), detail::get_op<T, F>().mpi_op, root,
		   comm);
      else
	MPI_Reduce(&sendrecvdata, nullptr, 1,
		   datatype_traits<T>::get_datatype(), detail::get_op<T, F>().mpi_op, root,
		   comm);
    }
    
    template<typename T, typename F>
    void reduce(F f, int root,
		const T &senddata) const {
      check_nonroot(root);
      detail::get_op<T, F>().f=&f;
      MPI_Reduce(&senddata, nullptr, 1,
		 datatype_traits<T>::get_datatype(), detail::get_op<T, F>().mpi_op, root,
		 comm);
    }
    
    template<typename T, typename F>
    void reduce(F f, int root,
		T *sendrecvdata, const contiguous_layout<T> &l) const {
      detail::get_op<T, F>().f=&f;
      if (rank()==root)
	MPI_Reduce(MPI_IN_PLACE, sendrecvdata, l.size(),
		   datatype_traits<T>::get_datatype(), detail::get_op<T, F>().mpi_op, root,
		   comm);
      else
	MPI_Reduce(sendrecvdata, nullptr, l.size(),
		   datatype_traits<T>::get_datatype(), detail::get_op<T, F>().mpi_op, root,
		   comm);
    }
    
    template<typename T, typename F>
    void reduce(F f, int root,
		const T *sendrecvdata, const contiguous_layout<T> &l) const {
      check_nonroot(root);
      detail::get_op<T, F>().f=&f;
      MPI_Reduce(sendrecvdata, nullptr, l.size(),
	datatype_traits<T>::get_datatype(), detail::get_op<T, F>().mpi_op, root,
	comm);
    }
    
    // --- non-blocking reduce, in place ---
    template<typename T, typename F>
    detail::irequest ireduce(F f, int root,
			     T &sendrecvdata) const {
      check_root(root);
      detail::get_op<T, F>().f=&f;
      MPI_Request req;
      if (rank()==root)
	MPI_Ireduce(MPI_IN_PLACE, &sendrecvdata, 1,
		   datatype_traits<T>::get_datatype(), detail::get_op<T, F>().mpi_op, root,
		    comm, &req);
      else
	MPI_Ireduce(&sendrecvdata, nullptr, 1,
		   datatype_traits<T>::get_datatype(), detail::get_op<T, F>().mpi_op, root,
		   comm, &req);
      return detail::irequest(req);
    }
    
    template<typename T, typename F>
    detail::irequest ireduce(F f, int root,
			     const T &sendrecvdata) const {
      check_nonroot(root);
      detail::get_op<T, F>().f=&f;
      MPI_Request req;
      MPI_Ireduce(&sendrecvdata, nullptr, 1,
		  datatype_traits<T>::get_datatype(), detail::get_op<T, F>().mpi_op, root,
		  comm, &req);
      return detail::irequest(req);
    }
    
    template<typename T, typename F>
    detail::irequest ireduce(F f, int root,
			     T *sendrecvdata, const contiguous_layout<T> &l) const {
      check_root(root);
      detail::get_op<T, F>().f=&f;
      MPI_Request req;
      if (rank()==root)
	MPI_Ireduce(MPI_IN_PLACE, sendrecvdata, l.size(),
		    datatype_traits<T>::get_datatype(), detail::get_op<T, F>().mpi_op, root,
		    comm, &req);
      else
	MPI_Ireduce(sendrecvdata, nullptr, l.size(),
		    datatype_traits<T>::get_datatype(), detail::get_op<T, F>().mpi_op, root,
		    comm, &req);
      return detail::irequest(req);
    }
    
    template<typename T, typename F>
    detail::irequest ireduce(F f, int root,
			     const T *sendrecvdata, const contiguous_layout<T> &l) const {
      check_nonroot(root);
      detail::get_op<T, F>().f=&f;
      MPI_Request req;
      MPI_Ireduce(sendrecvdata, nullptr, l.size(),
	          datatype_traits<T>::get_datatype(), detail::get_op<T, F>().mpi_op, root,
  	          comm, &req);
      return detail::irequest(req);
    }
    
    // === all-reduce ===
    // --- blocking all-reduce ---
    template<typename T, typename F>
    void allreduce(F f,
		   const T &senddata, T &recvdata) const {
      detail::get_op<T, F>().f=&f;
      MPI_Allreduce(&senddata, &recvdata, 1,
		    datatype_traits<T>::get_datatype(), detail::get_op<T, F>().mpi_op,
		    comm);
    }
    
    template<typename T, typename F>
    void allreduce(F f,
		   const T *senddata, T *recvdata, const contiguous_layout<T> &l) const {
      detail::get_op<T, F>().f=&f;
      MPI_Allreduce(senddata, recvdata, l.size(),
		    datatype_traits<T>::get_datatype(), detail::get_op<T, F>().mpi_op,
		    comm);
    }
    
    // --- non-blocking all-reduce ---
    template<typename T, typename F>
    detail::irequest iallreduce(F f,
				const T &senddata, T &recvdata) const {
      detail::get_op<T, F>().f=&f;
      MPI_Request req;
      MPI_Iallreduce(&senddata, &recvdata, 1,
		     datatype_traits<T>::get_datatype(), detail::get_op<T, F>().mpi_op,
		     comm, &req);
      return detail::irequest(req);
    }
    
    template<typename T, typename F>
    detail::irequest iallreduce(F f,
				const T *senddata, T *recvdata, const contiguous_layout<T> &l) const {
      detail::get_op<T, F>().f=&f;
      MPI_Request req;
      MPI_Iallreduce(senddata, recvdata, l.size(),
		     datatype_traits<T>::get_datatype(), detail::get_op<T, F>().mpi_op,
		     comm, &req);
      return detail::irequest(req);
    }
    
    // --- blocking all-reduce, in place ---
    template<typename T, typename F>
    void allreduce(F f,
		   T &sendrecvdata) const {
      detail::get_op<T, F>().f=&f;
      MPI_Allreduce(MPI_IN_PLACE, &sendrecvdata, 1,
		    datatype_traits<T>::get_datatype(), detail::get_op<T, F>().mpi_op,
		    comm);
    }
    
    template<typename T, typename F>
    void allreduce(F f,
		   T *sendrecvdata, const contiguous_layout<T> &l) const {
      detail::get_op<T, F>().f=&f;
      MPI_Allreduce(MPI_IN_PLACE, sendrecvdata, l.size(),
		    datatype_traits<T>::get_datatype(), detail::get_op<T, F>().mpi_op,
		    comm);
    }
    
    // --- non-blocking all-reduce, in place ---
    template<typename T, typename F>
    detail::irequest iallreduce(F f,
				T &sendrecvdata) const {
      detail::get_op<T, F>().f=&f;
      MPI_Request req;
      MPI_Iallreduce(MPI_IN_PLACE, &sendrecvdata, 1,
		     datatype_traits<T>::get_datatype(), detail::get_op<T, F>().mpi_op,
		     comm, &req);
      return detail::irequest(req);
    }
    
    template<typename T, typename F>
    detail::irequest iallreduce(F f,
				T *sendrecvdata, const contiguous_layout<T> &l) const {
      detail::get_op<T, F>().f=&f;
      MPI_Request req;
      MPI_Iallreduce(MPI_IN_PLACE, sendrecvdata, l.size(),
		     datatype_traits<T>::get_datatype(), detail::get_op<T, F>().mpi_op,
		     comm, &req);
      return detail::irequest(req);
    }
    
    // === reduce-scatter-block ===
    // --- blocking reduce-scatter-block ---
    template<typename T, typename F>
    void reduce_scatter_block(F f,
			      const T *senddata, T &recvdata) const {
      detail::get_op<T, F>().f=&f;
      MPI_Reduce_scatter_block(senddata, &recvdata, 1,
			       datatype_traits<T>::get_datatype(), detail::get_op<T, F>().mpi_op,
			       comm);
    }
    
    template<typename T, typename F>
    void reduce_scatter_block(F f,
			      const T *senddata, T *recvdata, const contiguous_layout<T> &l) const {
      detail::get_op<T, F>().f=&f;
      MPI_Reduce_scatter_block(senddata, recvdata, l.size(),
			       datatype_traits<T>::get_datatype(), detail::get_op<T, F>().mpi_op,
			       comm);
    }
    
    // --- non-blocking reduce-scatter-block ---
    template<typename T, typename F>
    detail::irequest ireduce_scatter_block(F f,
					   const T *senddata, T &recvdata) const {
      detail::get_op<T, F>().f=&f;
      MPI_Request req;
      MPI_Ireduce_scatter_block(&senddata, &recvdata, 1,
				datatype_traits<T>::get_datatype(), detail::get_op<T, F>().mpi_op,
				comm, &req);
      return detail::irequest(req);
    }
    
    template<typename T, typename F>
    detail::irequest ireduce_scatter_block(F f,
					   const T *senddata, T *recvdata, const contiguous_layout<T> &l) const {
      detail::get_op<T, F>().f=&f;
      MPI_Request req;
      MPI_Ireduce_scatter_block(senddata, recvdata, l.size(),
				datatype_traits<T>::get_datatype(), detail::get_op<T, F>().mpi_op,
				comm, &req);
      return detail::irequest(req);
    }
    
    // === reduce-scatter ===
    // --- blocking reduce-scatter ---
    template<typename T, typename F>
    void reduce_scatter(F f,
			const T *senddata, T *recvdata, const contiguous_layouts<T> &recvcounts) const {
      detail::get_op<T, F>().f=&f;
      MPI_Reduce_scatter(senddata, recvdata, recvcounts.sizes(),
			 datatype_traits<T>::get_datatype(), detail::get_op<T, F>().mpi_op,
			 comm);
    }
    
    // --- non-blocking reduce-scatter ---
    template<typename T, typename F>
    detail::irequest ireduce_scatter(F f,
				     const T *senddata, T *recvdata, contiguous_layouts<T> &recvcounts) const {
      detail::get_op<T, F>().f=&f;
      MPI_Request req;
      MPI_Ireduce_scatter(senddata, recvdata, recvcounts.sizes(),
			  datatype_traits<T>::get_datatype(), detail::get_op<T, F>().mpi_op,
			  comm, &req);
      return detail::irequest(req);
    }
    
    // --- blocking reduce-scatter, in place ---
    template<typename T, typename F>
    void reduce_scatter(F f,
			T *recvdata, const contiguous_layouts<T> &recvcounts) const {
      detail::get_op<T, F>().f=&f;
      MPI_Reduce_scatter(MPI_IN_PLACE, recvdata, recvcounts.sizes(),
			 datatype_traits<T>::get_datatype(), detail::get_op<T, F>().mpi_op,
			 comm);
    }
    
    // --- non-blocking reduce-scatter, in place ---
    template<typename T, typename F>
    detail::irequest ireduce_scatter(F f,
				     T *recvdata, const contiguous_layouts<T> &recvcounts) const {
      detail::get_op<T, F>().f=&f;
      MPI_Request req;
      MPI_Ireduce_scatter(MPI_IN_PLACE, recvdata, recvcounts.sizes(),
			  datatype_traits<T>::get_datatype(), detail::get_op<T, F>().mpi_op,
			  comm, &req);
      return detail::irequest(req);
    }
    
    // === scan ===
    // --- blocking scan ---
    template<typename T, typename F>
    void scan(F f,
	      const T &senddata, T &recvdata) const {
      detail::get_op<T, F>().f=&f;
      MPI_Scan(&senddata, &recvdata, 1,
	       datatype_traits<T>::get_datatype(), detail::get_op<T, F>().mpi_op,
	       comm);
    }
    
    template<typename T, typename F>
    void scan(F f,
	      const T *senddata, T *recvdata, const contiguous_layout<T> &l) const {
      detail::get_op<T, F>().f=&f;
      MPI_Scan(senddata, recvdata, l.size(),
	       datatype_traits<T>::get_datatype(), detail::get_op<T, F>().mpi_op,
	       comm);
    }
    
    // --- non-blocking scan ---
    template<typename T, typename F>
    detail::irequest iscan(F f,
			   const T &senddata, T &recvdata) const {
      detail::get_op<T, F>().f=&f;
      MPI_Request req;
      MPI_Iscan(&senddata, &recvdata, 1,
		datatype_traits<T>::get_datatype(), detail::get_op<T, F>().mpi_op,
		comm, &req);
      return detail::irequest(req);
    }
    
    template<typename T, typename F>
    detail::irequest iscan(F f,
			   const T *senddata, T *recvdata, const contiguous_layout<T> &l) const {
      detail::get_op<T, F>().f=&f;
      MPI_Request req;
      MPI_Iscan(senddata, recvdata, l.size(),
		datatype_traits<T>::get_datatype(), detail::get_op<T, F>().mpi_op,
		comm, &req);
      return detail::irequest(req);
    }
    
    // --- blocking scan, in place ---
    template<typename T, typename F>
    void scan(F f,
	      T &recvdata) const {
      detail::get_op<T, F>().f=&f;
      MPI_Scan(MPI_IN_PLACE, &recvdata, 1,
	       datatype_traits<T>::get_datatype(), detail::get_op<T, F>().mpi_op,
	       comm);
    }
    
    template<typename T, typename F>
    void scan(F f,
	      T *recvdata, const contiguous_layout<T> &l) const {
      detail::get_op<T, F>().f=&f;
      MPI_Scan(MPI_IN_PLACE, recvdata, l.size(),
	       datatype_traits<T>::get_datatype(), detail::get_op<T, F>().mpi_op,
	       comm);
    }
    
    // --- non-blocking scan, in place ---
    template<typename T, typename F>
    detail::irequest iscan(F f,
                           T &recvdata) const {
      detail::get_op<T, F>().f=&f;
      MPI_Request req;
      MPI_Iscan(MPI_IN_PLACE, &recvdata, 1,
		datatype_traits<T>::get_datatype(), detail::get_op<T, F>().mpi_op,
		comm, &req);
      return detail::irequest(req);
    }
    
    template<typename T, typename F>
    detail::irequest iscan(F f,
			   T *recvdata, const contiguous_layout<T> &l) const {
      detail::get_op<T, F>().f=&f;
      MPI_Request req;
      MPI_Iscan(MPI_IN_PLACE, recvdata, l.size(),
		datatype_traits<T>::get_datatype(), detail::get_op<T, F>().mpi_op,
		comm, &req);
      return detail::irequest(req);
    }
    
    // === exscan ===
    // --- blocking exscan ---
    template<typename T, typename F>
    void exscan(F f,
		const T &senddata, T &recvdata) const {
      detail::get_op<T, F>().f=&f;
      MPI_Exscan(&senddata, &recvdata, 1,
		 datatype_traits<T>::get_datatype(), detail::get_op<T, F>().mpi_op,
		 comm);
    }
    
    template<typename T, typename F>
    void exscan(F f,
		const T *senddata, T *recvdata, const contiguous_layout<T> &l) const {
      detail::get_op<T, F>().f=&f;
      MPI_Exscan(senddata, recvdata, l.size(),
		 datatype_traits<T>::get_datatype(), detail::get_op<T, F>().mpi_op,
		 comm);
    }
    
    // --- non-blocking exscan ---
    template<typename T, typename F>
    detail::irequest iexscan(F f,
			     const T &senddata, T &recvdata) const {
      detail::get_op<T, F>().f=&f;
      MPI_Request req;
      MPI_Iexscan(&senddata, &recvdata, 1,
		  datatype_traits<T>::get_datatype(), detail::get_op<T, F>().mpi_op,
		  comm, &req);
      return detail::irequest(req);
    }
    
    template<typename T, typename F>
    detail::irequest iexscan(F f,
			     const T *senddata, T *recvdata, const contiguous_layout<T> &l) const {
      detail::get_op<T, F>().f=&f;
      MPI_Request req;
      MPI_Iexscan(senddata, recvdata, l.size(),
		  datatype_traits<T>::get_datatype(), detail::get_op<T, F>().mpi_op,
		  comm, &req);
      return detail::irequest(req);
    }
    
    // --- blocking exscan, in place ---
    template<typename T, typename F>
    void exscan(F f,
		T &recvdata) const {
      detail::get_op<T, F>().f=&f;
      MPI_Exscan(MPI_IN_PLACE, &recvdata, 1,
		 datatype_traits<T>::get_datatype(), detail::get_op<T, F>().mpi_op,
		 comm);
    }
    
    template<typename T, typename F>
    void exscan(F f,
		T *recvdata, const contiguous_layout<T> &l) const {
      detail::get_op<T, F>().f=&f;
      MPI_Exscan(MPI_IN_PLACE, recvdata, l.size(),
		 datatype_traits<T>::get_datatype(), detail::get_op<T, F>().mpi_op,
		 comm);
    }
    
    // --- non-blocking exscan, in place ---
    template<typename T, typename F>
    detail::irequest iexscan(F f,
			     T &recvdata) const {
      detail::get_op<T, F>().f=&f;
      MPI_Request req;
      MPI_Iexscan(MPI_IN_PLACE, &recvdata, 1,
		  datatype_traits<T>::get_datatype(), detail::get_op<T, F>().mpi_op,
		  comm, &req);
      return detail::irequest(req);
    }
    
    template<typename T, typename F>
    detail::irequest iexscan(F f,
			     T *recvdata, const contiguous_layout<T> &l) const {
      detail::get_op<T, F>().f=&f;
      MPI_Request req;
      MPI_Iexscan(MPI_IN_PLACE, recvdata, l.size(),
		  datatype_traits<T>::get_datatype(), detail::get_op<T, F>().mpi_op,
		  comm, &req);
      return detail::irequest(req);
    }

  };

  //--------------------------------------------------------------------

  inline group::group(const communicator &comm) {
    MPI_Comm_group(comm.comm, &gr);
  }

  inline group::group(group::Union,
		      const group &other_1, const group &other_2) {
    MPI_Group_union(other_1.gr, other_2.gr, &gr);
  }

  inline group::group(group::intersection,
		      const group &other_1, const group &other_2) {
    MPI_Group_intersection(other_1.gr, other_2.gr, &gr);
  }

  inline group::group(group::difference,
		      const group &other_1, const group &other_2) {
    MPI_Group_difference(other_1.gr, other_2.gr, &gr);
  }

  inline group::group(group::incl,
		      const group &other, const ranks &rank) {
    MPI_Group_incl(other.gr, rank.size(), rank(), &gr);
  }

  inline group::group(group::excl,
		      const group &other, const ranks &rank) {
    MPI_Group_excl(other.gr, rank.size(), rank(), &gr);
  }

}

#endif
