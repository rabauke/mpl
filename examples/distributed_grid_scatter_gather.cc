#include <cstdlib>
#include <iostream>
#include <mpl/mpl.hpp>

template<std::size_t dim, typename T, typename A>
void scatter(const mpl::cart_communicator &C, int root, 
	     const mpl::local_grid<dim, T, A> &L, 
	     mpl::distributed_grid<dim, T, A> &G) {
  mpl::displacements send_recv_d(C.size());
  mpl::layouts<T> recvl;
  for (int i=0; i<C.size(); ++i)
    if (i==root)
      recvl.push_back(G.interior_layout());
    else
      recvl.push_back(mpl::empty_layout<T>());
  if (C.rank()==root)
    C.alltoallw(L.data(), L.sub_layouts(), send_recv_d, 
		G.data(), recvl, send_recv_d);
  else 
    C.alltoallw(L.data(), mpl::layouts<T>(C.size()), send_recv_d, 
		G.data(), recvl, send_recv_d);
}

template<std::size_t dim, typename T, typename A>
void scatter(const mpl::cart_communicator &C, int root, 
	     mpl::distributed_grid<dim, T, A> &G) {
  mpl::displacements send_recv_d(C.size());
  mpl::layouts<T> recvl;
  for (int i=0; i<C.size(); ++i)
    if (i==root)
      recvl.push_back(G.interior_layout());
    else
      recvl.push_back(mpl::empty_layout<T>());
  C.alltoallw(static_cast<const T *>(nullptr), mpl::layouts<T>(C.size()), send_recv_d, 
	      G.data(), recvl, send_recv_d);
}


template<std::size_t dim, typename T, typename A>
void gather(const mpl::cart_communicator &C, int root, 
	    const mpl::distributed_grid<dim, T, A> &G, 
	    mpl::local_grid<dim, T, A> &L) {
  C.gatherv(root, 
	    G.data(), G.interior_layout(), 
	    L.data(), L.sub_layouts(), mpl::displacements(C.size()));
}

template<std::size_t dim, typename T, typename A>
void gather(const mpl::cart_communicator &C, int root, 
	    const mpl::distributed_grid<dim, T, A> &G) {
  // mpl::displacements send_recv_d(C.size());
  // mpl::layouts<T> sendl, recvl;
  // for (int i=0; i<C.size(); ++i)
  //   if (i==root)
  //     sendl.push_back(G.interior_layout());
  //   else
  //     sendl.push_back(mpl::empty_layout<T>());
  // C.alltoallw(G.data(), sendl, send_recv_d, 
  // 	      static_cast<T *>(nullptr), mpl::layouts<T>(C.size()), send_recv_d);
  C.gatherv(root, 
	    G.data(), G.interior_layout());
}


int main() {
  const mpl::communicator & comm_world(mpl::environment::comm_world());
  mpl::cart_communicator::sizes sizes( {{0, true}, {0, false}} );
  int Nx(21), Ny(13);
  mpl::cart_communicator comm_c(comm_world, 
				mpl::dims_create(comm_world.size(), sizes));
  mpl::distributed_grid<2, int> G(comm_c, 
				  {{Nx, 1}, {Ny, 1}});
  for (auto j=G.obegin(1), j_end=G.oend(1); j<j_end; ++j)
    for (auto i=G.obegin(0), i_end=G.oend(0); i<i_end; ++i)
      G(i, j)=comm_c.rank();
  if (comm_world.rank()==0) {
    mpl::local_grid<2, int> Gl(comm_c, {Nx, Ny});
    for (auto j=Gl.begin(1), j_end=Gl.end(1); j<j_end; ++j)
      for (auto i=Gl.begin(0), i_end=Gl.end(0); i<i_end; ++i) 
	Gl(i, j)=0;
    scatter(comm_c, 0, Gl, G);
  } else 
    scatter(comm_c, 0, G);
  for (int i=0; i<comm_c.size(); ++i) {
    if (i==comm_c.rank()) {
      std::cout << std::endl;
      for (auto j=G.obegin(1), j_end=G.oend(1); j<j_end; ++j) {
	for (auto i=G.obegin(0), i_end=G.oend(0); i<i_end; ++i) 
	  std::cout << G(i, j);
	std::cout << std::endl;
      }
    }
    comm_c.barrier();
  }
  for (auto j=G.obegin(1), j_end=G.oend(1); j<j_end; ++j)
    for (auto i=G.obegin(0), i_end=G.oend(0); i<i_end; ++i)
      G(i, j)=comm_c.rank();
  if (comm_world.rank()==0) {
    mpl::local_grid<2, int> Gl(comm_c, {Nx, Ny});
    gather(comm_c, 0, G, Gl);
    std::cout << std::endl;
    for (auto j=Gl.begin(1), j_end=Gl.end(1); j<j_end; ++j) {
      for (auto i=G.begin(0), i_end=Gl.end(0); i<i_end; ++i) 
	std::cout << Gl(i, j);
      std::cout << std::endl;
    }
  } else 
    gather(comm_c, 0, G);
  return EXIT_SUCCESS;
}
