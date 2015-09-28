#include <cstdlib>
#include <iostream>
#include <cmath>
#include <tuple>
#include <future>
#include <mpl/mpl.hpp>

typedef std::tuple<double, double> double_2;


template<std::size_t dim, typename T, typename A>
void update_overlap(const mpl::cart_communicator &C, 
		    mpl::distributed_grid<dim, T, A> &G, int tag=0) {
  mpl::shift_ranks ranks;
  mpl::irequest_pool r;
  for (std::size_t i=0; i<dim; ++i) {
    // send to left
    ranks=C.shift(i, -1);
    r.push(C.isend(G.data(), G.left_border_layout(i), ranks.dest, tag));
    r.push(C.irecv(G.data(), G.right_mirror_layout(i), ranks.source, tag));
    // send to right
    ranks=C.shift(i, +1);
    r.push(C.isend(G.data(), G.right_border_layout(i), ranks.dest, tag));
    r.push(C.irecv(G.data(), G.left_mirror_layout(i), ranks.source, tag));
  }
  r.waitall();
}


template<std::size_t dim, typename T, typename A>
void scatter(const mpl::cart_communicator &C, int root, 
	     const mpl::local_grid<dim, T, A> &L, 
	     mpl::distributed_grid<dim, T, A> &G) {
  mpl::irequest r(C.irecv(G.data(), G.interior_layout(), root));
  if (C.rank()==root)
    for (int i=0; i<C.size(); ++i)
      C.send(L.data(), L.sub_layout(i), i);
  r.wait();
}

template<std::size_t dim, typename T, typename A>
void scatter(const mpl::cart_communicator &C, int root, 
	     mpl::distributed_grid<dim, T, A> &G) {
  C.recv(G.data(), G.interior_layout(), root);
}


template<std::size_t dim, typename T, typename A>
void gather(const mpl::cart_communicator &C, int root, 
	    const mpl::distributed_grid<dim, T, A> &G, 
	    mpl::local_grid<dim, T, A> &L) {
  mpl::irequest r(C.isend(G.data(), G.interior_layout(), root));
  if (C.rank()==root)
    for (int i=0; i<C.size(); ++i)
      C.recv(L.data(), L.sub_layout(i), i);
  r.wait();
}

template<std::size_t dim, typename T, typename A>
void gather(const mpl::cart_communicator &C, int root, 
	    const mpl::distributed_grid<dim, T, A> &G) {
  C.send(G.data(), G.interior_layout(), root);
}


int main() {
  // world communicator
  const mpl::communicator & comm_world(mpl::environment::comm_world());
  // construct a two-dimensional Cartesian communicator with no periodic boundary conditions
  mpl::cart_communicator::sizes sizes( {{0, false}, {0, false}} );
  mpl::cart_communicator comm_c(comm_world, 
				mpl::dims_create(comm_world.size(), sizes));
  // total number of inner grid points
  int Nx=768, Ny=512;
  // grid points with extremal indices (-1, Nx or Ny) hold fixed boundary data
  // grid lengths and grid spacings
  double l_x=1.5, l_y=1, dx=l_x/(Nx+1), dy=l_y/(Ny+1);
  // distributed grids that hold each processor's subgrid plus one row and 
  // one collumn  of neighboring data
  mpl::distributed_grid<2, double> u_d1(comm_c, {{Nx, 1}, {Ny, 1}});
  mpl::distributed_grid<2, double> u_d2(comm_c, {{Nx, 1}, {Ny, 1}});
  // rank 0 inializes with some random data
  if (comm_c.rank()==0) {
    // local grid to store the whole set of inner grid points
    mpl::local_grid<2, double> u(comm_c, {Nx, Ny});
    for (auto j=u.begin(1), j_end=u.end(1); j<j_end; ++j)
      for (auto i=u.begin(0), i_end=u.end(0); i<i_end; ++i) 
	u(i, j)=std::rand()/static_cast<double>(RAND_MAX);
    // scater data to each processors subgrid
    scatter(comm_c, 0, u, u_d1);
  } else
    scatter(comm_c, 0, u_d1);
  // initiallize boundary data, loop with obegin and oend over all 
  // data including the overlap, initialize if local border is global border
  for (auto j : { u_d1.obegin(1), u_d1.oend(1)-1 } )
    for (auto i=u_d1.obegin(0), i_end=u_d1.oend(0); i<i_end; ++i) {
      if (u_d1.gindex(0, i)<0 or u_d1.gindex(1, j)<0)
	u_d1(i, j)=u_d2(i, j)=1;
      if (u_d1.gindex(0, i)>=Nx or u_d1.gindex(1, j)>=Ny)
  	u_d1(i, j)=u_d2(i, j)=0;
    }
  for (auto i : { u_d1.obegin(0), u_d1.oend(0)-1 } )
    for (auto j=u_d1.obegin(1), j_end=u_d1.oend(1); j<j_end; ++j) {
      if (u_d1.gindex(0, i)<0 or u_d1.gindex(1, j)<0)
	u_d1(i, j)=u_d2(i, j)=1;
      if (u_d1.gindex(0, i)>=Nx or u_d1.gindex(1, j)>=Ny)
  	u_d1(i, j)=u_d2(i, j)=0;
    }  
  double dx2=dx*dx, dy2=dy*dy;
  // loop until converged
  bool converged=false;
  int iterations=0;
  while (not converged) {
    iterations++;
    // exchange asynchronously overlapping boundary data
    std::future<void> update_overlap_f(std::async(std::launch::async, 
						  [&](){ update_overlap(comm_c, u_d1); } ));
    // apply one Jacobi iteration step for interior region
    double Delta_u=0, sum_u=0;
    for (auto j=u_d1.begin(1)+1, j_end=u_d1.end(1)-1; j<j_end; ++j)
      for (auto i=u_d1.begin(0)+1, i_end=u_d1.end(0)-1; i<i_end; ++i) {
  	u_d2(i, j)=(dy2*(u_d1(i-1, j)+u_d1(i+1, j)) + dx2*(u_d1(i, j-1)+u_d1(i, j+1)))/(2*(dx2+dy2));
	Delta_u+=std::abs(u_d2(i, j)-u_d1(i, j));
	sum_u+=std::abs(u_d2(i, j));
      }
    update_overlap_f.get();
    // apply one Jacobi iteration step for edge region, which requires overlapping boundary data
    for (auto j : { u_d1.begin(1), u_d1.end(1)-1 })
      for (auto i=u_d1.begin(0), i_end=u_d1.end(0); i<i_end; ++i) {
  	u_d2(i, j)=(dy2*(u_d1(i-1, j)+u_d1(i+1, j)) + dx2*(u_d1(i, j-1)+u_d1(i, j+1)))/(2*(dx2+dy2));
	Delta_u+=std::abs(u_d2(i, j)-u_d1(i, j));
	sum_u+=std::abs(u_d2(i, j));
      }
    for (auto j=u_d1.begin(1), j_end=u_d1.end(1); j<j_end; ++j)
      for (auto i : { u_d1.begin(0), u_d1.end(0)-1 }) {
  	u_d2(i, j)=(dy2*(u_d1(i-1, j)+u_d1(i+1, j)) + dx2*(u_d1(i, j-1)+u_d1(i, j+1)))/(2*(dx2+dy2));
	Delta_u+=std::abs(u_d2(i, j)-u_d1(i, j));
	sum_u+=std::abs(u_d2(i, j));
      }
    // determine global sum of Delta_u and sum_u and distribute to all processors
    double_2 Delta_sum_u{ Delta_u, sum_u };  // pack into pair
    comm_c.allreduce(mpl::make_function([](double_2 a, double_2 b) { 
	  // reduction adds component-by-component
	  return double_2{ std::get<0>(a)+std::get<0>(b), std::get<1>(a)+std::get<1>(b) };
	}), Delta_sum_u);
    std::tie(Delta_u, sum_u)=Delta_sum_u;  // unpack from pair
    converged=Delta_u/sum_u<1e-3;  // check for convergence
    u_d2.swap(u_d1);
  }
  // gather data and print result
  if (comm_c.rank()==0) {
    std::cerr << iterations << '\n';
    // local grid to store the whole set of inner grid points
    mpl::local_grid<2, double> u(comm_c, {Nx, Ny});
    gather(comm_c, 0, u_d1, u);
    for (auto j=u.begin(1), j_end=u.end(1); j<j_end; ++j) {
      for (auto i=u.begin(0), i_end=u.end(0); i<i_end; ++i) 
	std::cout << u(i, j) << '\t';	
      std::cout << '\n';
    }
  } else
    gather(comm_c, 0, u_d1);
  return EXIT_SUCCESS;
}
