#include <cstdlib>
#include <iostream>
#include <cmath>
#include <tuple>
#include <mpl/mpl.hpp>

typedef std::tuple<double, double> double_2;


template<std::size_t dim, typename T, typename A>
void update_overlap(const mpl::cart_communicator &C, mpl::distributed_grid<dim, T, A> &G, int tag=0) {
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
void gather(const mpl::cart_communicator &C, const mpl::distributed_grid<dim, T, A> &G, mpl::local_grid<dim, T, A> &L, int root, int tag=0) {
  mpl::irequest r(C.isend(G.data(), G.interior_layout(), root, tag));
  if (C.rank()==root)
    for (int i=0; i<C.size(); ++i)
      C.recv(L.data(), L.sub_layout(i), i, tag);
  r.wait();
}


template<std::size_t dim, typename T, typename A>
void scatter(const mpl::cart_communicator &C, const mpl::local_grid<dim, T, A> &L, mpl::distributed_grid<dim, T, A> &G, int root, int tag=0) {
  mpl::irequest r(C.irecv(G.data(), G.interior_layout(), root, tag));
  if (C.rank()==root)
    for (int i=0; i<C.size(); ++i)
      C.send(L.data(), L.sub_layout(i), i, tag);
  r.wait();
}


int main() {
  const mpl::communicator & comm_world(mpl::environment::comm_world());
  mpl::cart_communicator::sizes sizes( {{0, false}, {0, false}} );
  mpl::cart_communicator comm_c(comm_world, 
				mpl::dims_create(comm_world.size(), sizes));
  int Nx=768, Ny=512;
  double l_x=1.5, l_y=1, dx=l_x/(Nx+1), dy=l_y/(Ny+1);
  mpl::local_grid<2, double> u(comm_c, 
			       {comm_c.rank()==0 ? Nx : 0, comm_c.rank()==0 ? Ny : 0});
  mpl::distributed_grid<2, double> u_d(comm_c, 
				       {{Nx, 1}, {Ny, 1}});
  if (comm_c.rank()==0)
    for (auto j=u.begin(1), j_end=u.end(1); j<j_end; ++j)
      for (auto i=u.begin(0), i_end=u.end(0); i<i_end; ++i) 
	u(i, j)=std::rand()/static_cast<double>(RAND_MAX);
  scatter(comm_c, u, u_d, 0);
  for (auto j=u_d.obegin(1), j_end=u_d.oend(1); j<j_end; ++j)
    for (auto i=u_d.obegin(0), i_end=u_d.oend(0); i<i_end; ++i) {
      if (u_d.gindex(0, i)<0 or u_d.gindex(1, j)<0)
	u_d(i, j)=1;
      if (u_d.gindex(0, i)>=Nx or u_d.gindex(1, j)>=Ny)
  	u_d(i, j)=0;
    }
  double w=1.875, dx2=dx*dx, dy2=dy*dy;
  while (true) {
    update_overlap(comm_c, u_d);
    double Delta_u=0, sum_u=0;
    for (auto j=u_d.begin(1), j_end=u_d.end(1); j<j_end; ++j)
      for (auto i=u_d.begin(0), i_end=u_d.end(0); i<i_end; ++i) {
	double du=-w*u_d(i, j)+
	  w*(dy2*(u_d(i-1, j)+u_d(i+1, j)) + dx2*(u_d(i, j-1)+u_d(i, j+1)))/(2*(dx2+dy2));
  	u_d(i, j)+=du;
	Delta_u+=std::abs(du);
	sum_u+=std::abs(u_d(i, j));
      }
    double_2 Delta_sum_u{ Delta_u, sum_u };
    comm_c.allreduce(std::function<double_2(double_2, double_2)>( [](double_2 a, double_2 b) { 
	  return double_2{ std::get<0>(a)+std::get<0>(b), std::get<1>(a)+std::get<1>(b) };
	} ), Delta_sum_u);
    std::tie(Delta_u, sum_u)=Delta_sum_u;
    if (Delta_u/sum_u<1e-6)
      break;
  }
  gather(comm_c, u_d, u, 0);
  if (comm_c.rank()==0)
    for (auto j=u.begin(1), j_end=u.end(1); j<j_end; ++j) {
      for (auto i=u.begin(0), i_end=u.end(0); i<i_end; ++i) 
	std::cout << u(i, j) << '\t';	
      std::cout << '\n';
    }
  return EXIT_SUCCESS;
}
