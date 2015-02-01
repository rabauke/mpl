// solve one-dimensional wave equation 

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"

const int N=1000;  // total global number of grid points is N+1
const double L=1, c=1, dt=0.001, t_end=2.4;
enum { left_copy, right_copy };

// update string elongation
void string(double *u, double *u_old, double *u_new, int N, double eps) {
  u_new[0]=u[0];
  for (int i=1; i<N; ++i)
    u_new[i]=eps*(u[i-1]+u[i+1])+2.0*(1.0-eps)*u[i]-u_old[i];
  u_new[N]=u[N];
}

void * secure_malloc(size_t size) {
  void * p=malloc(size);
  if (p==NULL)
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  return p;
}

// initial eleongation
inline double u_0(double x) {  
  return exp(-200.0*(x-0.5*L)*(x-0.5*L));  
}

// initial velocity
inline double u_0_dt(double x) {  
  return 0.0;  
}

int main(int argc, char *argv[]) {
  int C_rank, C_size, *N_l, *N0_l;
  double dx=L/N, eps=dt*dt*c*c/(dx*dx), 
    *u=NULL, *u_l, *u_old_l, *u_new_l, *u_temp;
  MPI_Status statuses[4];
  MPI_Request requests[4];
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &C_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &C_rank);
  N_l=secure_malloc(C_size*sizeof(*N_l));
  N0_l=secure_malloc(C_size*sizeof(*N0_l));
  // calculate size and position of local grids
  // local grids include one mirror grid point at each end
  for (int i=0; i<C_size; ++i) {
    // number of local grid points is N_l[C_rank]+1
    N_l[i]=(i+1)*(N-1)/C_size-i*(N-1)/C_size+1;
    // position of local 
    N0_l[i]=i*(N-1)/C_size;
  }
  u_old_l=secure_malloc((N_l[C_rank]+1)*sizeof(*u_old_l)); 
  u_l=secure_malloc((N_l[C_rank]+1)*sizeof(*u_l));          
  u_new_l=secure_malloc((N_l[C_rank]+1)*sizeof(*u_new_l));
  // 1st time step uses elongation and velocity
  for (int i=0; i<=N_l[C_rank]; ++i) {
    double x=(i+N0_l[C_rank])*dx;
    u_old_l[i]=u_0(x);
    u_l[i]=0.5*eps*(u_0(x-dx)+u_0(x+dx))+(1.0-eps)*u_0(x)+dt*u_0_dt(x);
  }
  // solve wave equation by using elongation at current time and one step before
  for (double t=2*dt; t<=t_end; t+=dt) {
    string(u_l, u_old_l, u_new_l, N_l[C_rank], eps);
    MPI_Isend(&u_new_l[N_l[C_rank]-1], 1, MPI_DOUBLE,
              C_rank+1<C_size ? C_rank+1 : MPI_PROC_NULL, right_copy,
              MPI_COMM_WORLD, &requests[0]);
    MPI_Isend(&u_new_l[1],             1, MPI_DOUBLE,
              C_rank-1>=0 ? C_rank-1 : MPI_PROC_NULL, left_copy,
              MPI_COMM_WORLD, &requests[1]);
    MPI_Irecv(&u_new_l[0],           1, MPI_DOUBLE,
	      C_rank-1>=0 ? C_rank-1 : MPI_PROC_NULL, right_copy,
	      MPI_COMM_WORLD, &requests[2]);
    MPI_Irecv(&u_new_l[N_l[C_rank]], 1, MPI_DOUBLE,
	      C_rank+1<C_size ? C_rank+1 : MPI_PROC_NULL, left_copy,
	      MPI_COMM_WORLD, &requests[3]);
    MPI_Waitall(4, requests, statuses);
    u_temp=u_old_l;  u_old_l=u_l;  u_l=u_new_l;  u_new_l=u_temp;
  }
  // gather and print data
  if (C_rank==0) 
    u=secure_malloc((N+1)*sizeof(*u));
  ++N_l[C_size-1];
  // altough receiving data overlaps gather gets the right data
  MPI_Gatherv(u_l, N_l[C_rank], MPI_DOUBLE, u, N_l, N0_l, MPI_DOUBLE,
	      0, MPI_COMM_WORLD);
  if (C_rank==0) 
    for (int i=0; i<=N; ++i)
      printf("%g\n", u[i]);
  free(u);  free(u_old_l);  free(u_l);  free(u_new_l);
  MPI_Finalize();
  return EXIT_SUCCESS;
}
