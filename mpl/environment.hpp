#if !(defined MPL_ENVIRONMENT_HPP)

#define MPL_ENVIRONMENT_HPP

#include <string>
#include <memory>
#include <mpi.h>

namespace mpl {
  
  namespace environment {

    namespace detail {

      class env {
	class initializer {
	public:
	  initializer() {
	    MPI_Init(0, 0);
	  }
	  ~initializer() {
	    MPI_Finalize();
	  }
	};
	
	initializer init;
	mpl::communicator comm_world_, comm_self_;
      public:
	env() : 
	  init(), comm_world_(MPI_COMM_WORLD), comm_self_(MPI_COMM_SELF) {
	}
	env(const env &) = delete;
	env& operator=(const env &) = delete;
	int tag_up() const {
	  int flag, tag_up_;
	  MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &tag_up_, &flag);
	  return tag_up_;
	}
	bool wtime_is_global() const {
	  int flag, wtime_is_global_;
	  MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_WTIME_IS_GLOBAL, &wtime_is_global_, &flag);
	  return wtime_is_global_;
	}
	const communicator & comm_world() const {
	  return comm_world_;
	}
	const communicator & comm_self() const {
	  return comm_self_;
	}
	std::string processor_name() const {
	  char name[MPI_MAX_PROCESSOR_NAME];
	  int len;
	  MPI_Get_processor_name(name, &len);
	  return std::string(name);
	}
	double wtime() const {
	  return MPI_Wtime();
	}
	double wtick() const {
	  return MPI_Wtick();
	}
	void buffer_attach(void *buff, int size) const {
	  MPI_Buffer_attach(buff, size);
	}
	std::pair<void *, int> buffer_detach() const {
	  void *buff;
	  int size;
	  MPI_Buffer_detach(&buff, &size);
	  return std::make_pair(buff, size);
	}
      };

      //----------------------------------------------------------------

      const env & get_env() {
	static env the_env;
	return the_env;
      }
      
    }
    
    //------------------------------------------------------------------
    
    int tag_up() {
      return detail::get_env().tag_up();
    }
    
    constexpr int any_tag() {
      return MPI_ANY_TAG;
    }
    
    constexpr int any_source() {
      return MPI_ANY_SOURCE;
    }

    constexpr int proc_null() {
      return MPI_PROC_NULL;
    }

    constexpr int undefined() {
      return MPI_UNDEFINED;
    }

    constexpr int root() {
      return MPI_ROOT;
    }

    constexpr int bsend_overheadroot() {
      return MPI_BSEND_OVERHEAD;
    }

    bool wtime_is_global() {
      return detail::get_env().wtime_is_global();
    }

    const communicator & comm_world() {
      return detail::get_env().comm_world();
    }

    const communicator & comm_self() {
      return detail::get_env().comm_self();
    }

    std::string processor_name() {
      return detail::get_env().processor_name();
    }

    double wtime() {
      return detail::get_env().wtime();
    }

    double wtick() {
      return detail::get_env().wtick();
    }

    void buffer_attach(void *buff, int size) {
      return detail::get_env().buffer_attach(buff, size);
    }

    std::pair<void *, int> buffer_detach() {
      return detail::get_env().buffer_detach();
    }

  }

  //--------------------------------------------------------------------

  template<typename A=std::allocator<char> >
  class bsend_buffer {
    int size;
    A alloc;
    char *buff;
  public:
    bsend_buffer(int size) : size(size), alloc(), buff(alloc.allocate(size)) {
      std::cerr << "allocate " << size << '\n';
      environment::buffer_attach(buff, size);
    }
    bsend_buffer(int size, A alloc) : size(size), alloc(alloc), buff(alloc.allocate(size)) {
      environment::buffer_attach(buff, size);
    }
    ~bsend_buffer() {
      // environment::buffer_detach();
      std::cerr << "deallocate " << environment::buffer_detach().second << '\n';
      alloc.deallocate(buff, size);
    }
  };

}

#endif
