#include <cstdlib>
#include <iostream>
#include <cmath>
#include <vector>
#include <mpl/mpl.hpp>

// some basic matrix class
template<typename T>
class matrix : private std::vector<T> {
  using base = std::vector<T>;

public:
  using typename base::size_type;
  using typename base::reference;
  using typename base::const_reference;
  using typename base::iterator;
  using typename base::const_iterator;

private:
  size_type nx, ny;

public:
  matrix(size_type nx, size_type ny) : base(nx * ny), nx(nx), ny(ny) {}

  const_reference operator()(size_type ix, size_type iy) const {
    return base::operator[](ix + nx * iy);
  }

  reference operator()(size_type ix, size_type iy) { return base::operator[](ix + nx * iy); }

  using base::begin;
  using base::end;
  using base::data;
};

int main() {
  const mpl::communicator &comm_world(mpl::environment::comm_world());
  int p = {comm_world.size()};  // total numbers of processors
  int p_l = {comm_world.rank()};
  // find integer px and py such that px*py=p and px and py as close as possible
  int px = {static_cast<int>(std::sqrt(static_cast<double>(p)))};
  while (p / px * px != p)
    --px;
  int py = {p / px};
  int nx = {31}, ny = {29};                // total size of the matrix
  matrix<int> nx_l(px, py), ny_l(px, py);  // sizes of sub matrices for both dimensions
  matrix<int> nx_0(px, py), ny_0(px, py);  // starts of sub matrices for both dimensions
  // matrix of layouts
  matrix<mpl::subarray_layout<int>> sub_matrix_l(px, py);
  // calculate all indices and sizes, generate layouts
  for (int iy = 0; iy < py; ++iy) {
    for (int ix = 0; ix < px; ++ix) {
      nx_l(ix, iy) = nx * (ix + 1) / px - nx * ix / px;
      ny_l(ix, iy) = ny * (iy + 1) / py - ny * iy / py;
      nx_0(ix, iy) = nx * ix / px;
      ny_0(ix, iy) = ny * iy / py;
      sub_matrix_l(ix, iy) = mpl::subarray_layout<int>(
          {{ny, ny_l(ix, iy), ny_0(ix, iy)}, {nx, nx_l(ix, iy), nx_0(ix, iy)}});
    }
  }
  // process local position in global data grid
  int py_l = {p_l / px}, px_l{p_l - px * py_l};

  // gather via send-recv
  {
    // fill some local matrix with data
    matrix<int> M_l(nx_l(px_l, py_l), ny_l(px_l, py_l));
    std::fill(M_l.begin(), M_l.end(), p_l);
    mpl::contiguous_layout<int> matrix_l(nx_l(px_l, py_l) * ny_l(px_l, py_l));
    // send local sub-matrix to rank 0
    mpl::irequest r(comm_world.isend(M_l.data(), matrix_l, 0));
    if (p_l == 0) {
      // gather all submatrices into one large matrix
      matrix<int> M(nx, ny);
      std::fill(M.begin(), M.end(), 0 - ' ');
      for (int iy = 0; iy < py; ++iy) {
        for (int ix = 0; ix < px; ++ix)
          comm_world.recv(M.data(), sub_matrix_l(ix, iy), ix + px * iy);
      }
      for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix)
          std::cout << static_cast<unsigned char>(M(ix, iy) + 'A');
        std::cout << '\n';
      }
      std::cout << '\n';
    }
    r.wait();
  }

  // gather via gatherv
  {
    // fill some local matrix with data
    matrix<int> M_l(nx_l(px_l, py_l), ny_l(px_l, py_l));
    std::fill(M_l.begin(), M_l.end(), p_l);
    // build the layouts for the gatherv operation
    int root = 0;
    mpl::contiguous_layout<int> matrix_l(nx_l(px_l, py_l) * ny_l(px_l, py_l));
    if (p_l == root) {
      mpl::layouts<int> recvl;
      for (int i = 0; i < p; ++i)
        recvl.push_back(sub_matrix_l(i % px, i / px));
      matrix<int> M(nx, ny);
      comm_world.gatherv(root, M_l.data(), matrix_l, M.data(), recvl);
      for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix)
          std::cout << static_cast<unsigned char>(M(ix, iy) + 'A');
        std::cout << '\n';
      }
      std::cout << '\n';
    } else
      comm_world.gatherv(root, M_l.data(), matrix_l);
  }

  // gather via alltoallv
  {
    // fill some local matrix with data
    matrix<int> M_l(nx_l(px_l, py_l), ny_l(px_l, py_l));
    std::fill(M_l.begin(), M_l.end(), p_l);
    // build the layouts for alltoallv to implement a gather operation
    int root = 0;
    mpl::layouts<int> sendl, recvl;
    for (int i = 0; i < p; ++i) {
      if (i == root)
        sendl.push_back(mpl::contiguous_layout<int>(nx_l(px_l, py_l) * ny_l(px_l, py_l)));
      else
        sendl.push_back(mpl::empty_layout<int>());
    }
    if (p_l == root) {
      for (int i = 0; i < p; ++i)
        recvl.push_back(sub_matrix_l(i % px, i / px));
      matrix<int> M(nx, ny);
      comm_world.alltoallv(M_l.data(), sendl, M.data(), recvl);
      for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix)
          std::cout << static_cast<unsigned char>(M(ix, iy) + 'A');
        std::cout << '\n';
      }
      std::cout << '\n';
    } else {
      for (int i = 0; i < p; ++i)
        recvl.push_back(mpl::empty_layout<int>());
      comm_world.alltoallv(M_l.data(), sendl, reinterpret_cast<int *>(0), recvl);
    }
  }

  return EXIT_SUCCESS;
}
