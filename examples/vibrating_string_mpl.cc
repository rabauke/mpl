// solve one-dimensional wave equation 

#include <cstdlib>
#include <iostream>
#include <cmath>
#include <vector>
#include <mpl/mpl.hpp>

const int N=1001;
const double L=1, c=1, dt=0.001, t_end=2.4;
enum { left_copy, right_copy };

void string(const std::vector<double> &u, const std::vector<double> &u_old, 
	    std::vector<double> &u_new, double eps) {
  typedef std::vector<double>::size_type size_type;
  size_type N=u.size();
  u_new[0]=u[0];
  for (size_type i=1; i<N-1; ++i)
    u_new[i]=eps*(u[i-1]+u[i+1])+2.0*(1.0-eps)*u[i]-u_old[i];
  u_new[N-1]=u[N-1];
}

inline double u_0(double x) {
  if (x<=0 or x>=L)
    return 0;
  return std::exp(-200.0*(x-0.5*L)*(x-0.5*L));  
}

inline double u_0_dt(double x) {  
  return 0.0;  
}

int main() {
  double dx=L/(N-1), eps=dt*dt*c*c/(dx*dx);
  const mpl::communicator & comm_world(mpl::environment::comm_world());
  int C_size=comm_world.size();
  int C_rank=comm_world.rank();
  std::vector<int> N_l, N0_l;
  for (int i=0; i<C_size; ++i) {
    N_l.push_back((i+1)*(N-2)/C_size-i*(N-2)/C_size+2);
    N0_l.push_back(i*(N-2)/C_size);
  }
  std::vector<double> u_old_l(N_l[C_rank]);
  std::vector<double> u_l(N_l[C_rank]);
  std::vector<double> u_new_l(N_l[C_rank]);
  for (int i=0; i<N_l[C_rank]; ++i) {
    double x=(i+N0_l[C_rank])*dx;
    u_old_l[i]=u_0(x);
    u_l[i]=0.5*eps*(u_0(x-dx)+u_0(x+dx))+(1.0-eps)*u_0(x)+dt*u_0_dt(x);
  }
  for (double t=2*dt; t<=t_end; t+=dt) {
    string(u_l, u_old_l, u_new_l, eps);
    mpl::irequest_pool r;
    r.push(comm_world.isend(u_new_l[N_l[C_rank]-2],
			    C_rank+1<C_size ? C_rank+1 : mpl::environment::proc_null(), right_copy));
    r.push(comm_world.isend(u_new_l[1],   
			    C_rank-1>=0 ? C_rank-1 : mpl::environment::proc_null(), left_copy));
    r.push(comm_world.irecv(u_new_l[0],
			    C_rank-1>=0 ? C_rank-1 : mpl::environment::proc_null(), right_copy));
    r.push(comm_world.irecv(u_new_l[N_l[C_rank]-1],
			    C_rank+1<C_size ? C_rank+1 : mpl::environment::proc_null(), left_copy));
    r.waitall();
    std::swap(u_l, u_old_l);  std::swap(u_new_l, u_l);
  }
  mpl::counts counts;
  mpl::displacements displ;
  for (int i=0; i<C_size; ++i) {
    counts.push_back(N_l[i]-2);
    displ.push_back(N0_l[i]+1);
  }
  if (C_rank==0) {
    std::vector<double> u(N);
    comm_world.gatherv(0, u_l.data(), counts[C_rank], 
		       u.data(), counts, displ);
    for (int i=0; i<N; ++i)
      std::cout << u[i] << '\n';
  } else
    comm_world.gatherv(0, u_l.data(), counts[C_rank]);
  return EXIT_SUCCESS;
}
