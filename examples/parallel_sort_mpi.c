#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "mpi.h"
#include <unistd.h>

typedef struct vector {
  double *data;
  size_t N;
} vector;

void fill_random(vector v) {
  for (size_t i = 0; i < v.N; ++i)
    v.data[i] = (double)rand() / (RAND_MAX + 1.);
}

static int cmp_double(const void *p1_, const void *p2_) {
  const double *const p1 = p1_;
  const double *const p2 = p2_;
  return (*p1 == *p2) ? 0 : (*p1 < *p2 ? -1 : 1);
}

double *partition(double *first, double *last, double pivot) {
  for (; first != last; ++first)
    if (!((*first) < pivot))
      break;
  if (first == last)
    return first;
  for (double *i = first + 1; i != last; ++i) {
    if ((*i) < pivot) {
      double temp = *i;
      *i = *first;
      *first = temp;
      ++first;
    }
  }
  return first;
}

vector parallel_sort(vector v) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  double *local_pivots = malloc(size * sizeof(*local_pivots));
  if (local_pivots == NULL)
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  double *pivots = malloc(size * (size + 1) * sizeof(*pivots));
  if (pivots == NULL)
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  for (int i = 0; i < size - 1; ++i)
    local_pivots[i] = v.data[(size_t)(v.N * (double)rand() / (RAND_MAX + 1.))];
  MPI_Allgather(local_pivots, size - 1, MPI_DOUBLE, pivots, size - 1, MPI_DOUBLE,
                MPI_COMM_WORLD);
  qsort(pivots, size * (size - 1), sizeof(double), cmp_double);
  for (size_t i = 1; i < size; ++i)
    local_pivots[i - 1] = pivots[i * (size - 1)];
  double **pivot_pos = malloc((size + 1) * sizeof(*pivot_pos));
  if (pivot_pos == NULL)
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  pivot_pos[0] = v.data;
  for (size_t i = 0; i < size - 1; ++i)
    pivot_pos[i + 1] = partition(pivot_pos[i], v.data + v.N, local_pivots[i]);
  pivot_pos[size] = v.data + v.N;
  int *local_block_sizes = malloc(size * sizeof(*local_block_sizes));
  if (local_block_sizes == NULL)
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  int *block_sizes = malloc(size * size * sizeof(*block_sizes));
  if (block_sizes == NULL)
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  for (size_t i = 0; i < size; ++i)
    local_block_sizes[i] = pivot_pos[i + 1] - pivot_pos[i];
  MPI_Allgather(local_block_sizes, size, MPI_INT, block_sizes, size, MPI_INT, MPI_COMM_WORLD);
  int send_pos = 0, recv_pos = 0;
  int sendcounts[size], sdispls[size], recvcounts[size], rdispls[size];
  for (size_t i = 0; i < size; ++i) {
    sendcounts[i] = block_sizes[rank * size + i];
    sdispls[i] = send_pos;
    send_pos += block_sizes[rank * size + i];
    recvcounts[i] = block_sizes[rank + size * i];
    rdispls[i] = recv_pos;
    recv_pos += block_sizes[rank + size * i];
  }
  double *v2 = malloc(recv_pos * sizeof(*v2));
  if (v2 == NULL)
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  MPI_Alltoallv(v.data, sendcounts, sdispls, MPI_DOUBLE, v2, recvcounts, rdispls, MPI_DOUBLE,
                MPI_COMM_WORLD);
  qsort(v2, recv_pos, sizeof(double), cmp_double);
  free(v.data);
  free(block_sizes);
  free(local_block_sizes);
  free(pivot_pos);
  free(pivots);
  free(local_pivots);
  return (vector){v2, recv_pos};
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  srand(time(NULL) * rank);

  const size_t N = 100000000 / size;
  double *v = malloc(N * sizeof(*v));
  if (v == NULL)
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  fill_random((vector){v, N});
  vector sorted = parallel_sort((vector){v, N});
  free(sorted.data);
  MPI_Finalize();
  return EXIT_SUCCESS;
}
