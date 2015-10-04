#include <cstdlib>
#include <iostream>
#include <mpl/mpl.hpp>

template<std::size_t dim, typename T, typename A>
void scatter(const mpl::cart_communicator &C, int root, 
	     const mpl::local_grid<dim, T, A> &L, 
	     mpl::distributed_grid<dim, T, A> &G) {
  mpl::layouts<T> sendl, recvl;
  mpl::displacements sendd, recvd;
  for (int i=0; i<C.size(); ++i) {
    if (i==root)
      recvl.push_back(G.interior_layout());
    else
      recvl.push_back(mpl::empty_layout<T>());
    recvd.push_back(0);
    if (C.rank()==root) {
      sendl.push_back(L.sub_layout(i));
      sendd.push_back(0);
    } else {
      sendl.push_back(mpl::empty_layout<T>());
      sendd.push_back(0);
    }
  }
  C.alltoallw(L.data(), sendl, sendd, 
	      G.data(), recvl, recvd);
}

template<std::size_t dim, typename T, typename A>
void scatter(const mpl::cart_communicator &C, int root, 
	     mpl::distributed_grid<dim, T, A> &G) {
  mpl::layouts<T> sendl, recvl;
  mpl::displacements sendd, recvd;
  for (int i=0; i<C.size(); ++i) {
    if (i==root) 
      recvl.push_back(G.interior_layout());
    else
      recvl.push_back(mpl::empty_layout<T>());
    recvd.push_back(0);
    sendl.push_back(mpl::empty_layout<T>());
    sendd.push_back(0);
  }
  C.alltoallw(static_cast<const T *>(nullptr), sendl, sendd, 
	      G.data(), recvl, recvd);
}


template<std::size_t dim, typename T, typename A>
void gather(const mpl::cart_communicator &C, int root, 
	    const mpl::distributed_grid<dim, T, A> &G, 
	    mpl::local_grid<dim, T, A> &L) {
  mpl::layouts<T> sendl, recvl;
  mpl::displacements sendd, recvd;
  for (int i=0; i<C.size(); ++i) {
    if (i==root)
      sendl.push_back(G.interior_layout());
    else
      sendl.push_back(mpl::empty_layout<T>());
    sendd.push_back(0);
    if (C.rank()==root) {
      recvl.push_back(L.sub_layout(i));
      recvd.push_back(0);
    } else {
      recvl.push_back(mpl::empty_layout<T>());
      recvd.push_back(0);
    }
  }
  C.alltoallw(G.data(), sendl, sendd, 
	      L.data(), recvl, recvd);
}

template<std::size_t dim, typename T, typename A>
void gather(const mpl::cart_communicator &C, int root, 
	    const mpl::distributed_grid<dim, T, A> &G) {
  mpl::layouts<T> sendl, recvl;
  mpl::displacements sendd, recvd;
  for (int i=0; i<C.size(); ++i) {
    if (i==root)
      sendl.push_back(G.interior_layout());
    else
      sendl.push_back(mpl::empty_layout<T>());
    sendd.push_back(0);
    recvl.push_back(mpl::empty_layout<T>());
    recvd.push_back(0);
  }
  C.alltoallw(G.data(), sendl, sendd, 
	      static_cast<T *>(nullptr), recvl, recvd);
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
