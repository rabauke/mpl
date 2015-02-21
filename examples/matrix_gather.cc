#include <cstdlib>
#include <iostream>
#include <cmath>
#include <vector>
#include <mpl/mpl.hpp>

template<typename T>
class matrix : private std::vector<T> {
  typedef std::vector<T> base;
public:
  typedef typename base::size_type size_type;
  typedef typename base::reference reference;
  typedef typename base::const_reference const_reference;
  typedef typename base::iterator iterator;
  typedef typename base::const_iterator const_iterator;
private:
  size_type nx, ny;
public:
  matrix(size_type nx, size_type ny) : base(nx*ny), nx(nx), ny(ny) {
  }
  const_reference operator()(size_type ix, size_type iy) const {
    return base::operator[](ix+nx*iy);
  }
  reference operator()(size_type ix, size_type iy) {
    return base::operator[](ix+nx*iy);
  }
  using base::begin;
  using base::end;
  using base::data;
};

int main() {
  const mpl::communicator & comm_world(mpl::environment::comm_world());
  int p={comm_world.size()};
  int p_l={comm_world.rank()};
  int px={static_cast<int>(std::sqrt(static_cast<double>(p)))};
  while (p/px*px!=p)
    --px;
  int py={p/px};
  int nx={31}, ny={29}; // total size of the matrix
  matrix<int> nx_l(px, py), ny_l(px, py);  // sizes of sub matrices
  matrix<int> nx_0(px, py), ny_0(px, py);  // starts of sub matrices
  matrix<mpl::subarray_layout<int> > sub_matrix_l(px, py);
  for (int iy=0; iy<py; ++iy) {
    for (int ix=0; ix<px; ++ix) {
      nx_l(ix, iy)=nx*(ix+1)/px-nx*ix/px;
      ny_l(ix, iy)=ny*(iy+1)/py-ny*iy/py;
      nx_0(ix, iy)=nx*ix/px;
      ny_0(ix, iy)=ny*iy/py;
      sub_matrix_l(ix, iy)=mpl::subarray_layout<int>(
	{ {ny, ny_l(ix, iy), ny_0(ix, iy)}, 
	  {nx, nx_l(ix, iy), nx_0(ix, iy)} });
    }
  }
  int py_l={p_l/px}, px_l{p_l-px*py_l};

  // gather via send-recv
  {
    matrix<int> M_l(nx_l(px_l, py_l), ny_l(px_l, py_l));
    std::fill(M_l.begin(), M_l.end(), p_l);
    mpl::contiguous_layout<int> matrix_l(nx_l(px_l, py_l)*ny_l(px_l, py_l));
    mpl::irequest r(comm_world.isend(M_l.data(), matrix_l, 0));
    if (p_l==0) {
      // gather all submatrices
      matrix<int> M(nx, ny);
      std::fill(M.begin(), M.end(), 0-' ');
      for (int iy=0; iy<py; ++iy) {
	for (int ix=0; ix<px; ++ix) {
	  comm_world.recv(M.data(), sub_matrix_l(ix, iy), ix+px*iy);
	  for (int iy=0; iy<ny; ++iy) {
	    for (int ix=0; ix<nx; ++ix) {
	      std::cout << static_cast<unsigned char>(M(ix, iy)+'A');
	    }
	    std::cout << '\n';
	  }
	  std::cout << '\n';
	}
      }
    }
    r.wait();
  }
  
  // gather via alltoallw
  {
    matrix<int> M_l(nx_l(px_l, py_l), ny_l(px_l, py_l));
    std::fill(M_l.begin(), M_l.end(), p_l);
    mpl::layouts<int> sendl, recvl;
    mpl::counts sendcounts, recvcounts;
    mpl::displacements senddispls, recvdispls;
    sendl.push_back(mpl::contiguous_layout<int>(nx_l(px_l, py_l)*ny_l(px_l, py_l)));
    sendcounts.push_back(1);
    senddispls.push_back(0);
    for (int i=1; i<p; ++i) {
      sendl.push_back(mpl::empty_layout<int>());
      sendcounts.push_back(0);
      senddispls.push_back(0);
    }
    if (p_l==0) {
      for (int i=0; i<p; ++i) {
  	recvl.push_back(sub_matrix_l(i%px, i/px));
  	recvcounts.push_back(1);
  	recvdispls.push_back(0);
      }
      matrix<int> M(nx, ny);
      comm_world.alltoallw(M_l.data(), sendl, sendcounts, senddispls,
       			   M.data(), recvl, recvcounts, recvdispls);
      for (int iy=0; iy<ny; ++iy) {
  	for (int ix=0; ix<nx; ++ix) {
  	  std::cout << static_cast<unsigned char>(M(ix, iy)+'A');
  	}
  	std::cout << '\n';
      }
      std::cout << '\n';
    } else {
      for (int i=0; i<p; ++i) {
	recvl.push_back(mpl::empty_layout<int>());
  	recvcounts.push_back(0);
  	recvdispls.push_back(0);
      }
      comm_world.alltoallw(M_l.data(), sendl, sendcounts, senddispls,
			   reinterpret_cast<int *>(0), recvl, recvcounts, recvdispls);
    }
  }

  // gather via alltoallw
  {
    auto par=mpl::dims_create(p, {{0,false}, {0, false}});
    std::cout << par.dims(0) << '\t' << par.dims(1) << '\n';
    mpl::cart_communicator cart_comm(comm_world, par);
    // matrix<int> M_l(nx_l(px_l, py_l), ny_l(px_l, py_l));
    // std::fill(M_l.begin(), M_l.end(), p_l);
    // mpl::layouts<int> sendl, recvl;
    // mpl::counts sendcounts, recvcounts;
    // mpl::displacements senddispls, recvdispls;
    // sendl.push_back(mpl::contiguous_layout<int>(nx_l(px_l, py_l)*ny_l(px_l, py_l)));
    // sendcounts.push_back(1);
    // senddispls.push_back(0);
    // for (int i=1; i<p; ++i) {
    //   sendl.push_back(mpl::empty_layout<int>());
    //   sendcounts.push_back(0);
    //   senddispls.push_back(0);
    // }
    // if (p_l==0) {
    //   for (int i=0; i<p; ++i) {
    // 	recvl.push_back(sub_matrix_l(i%px, i/px));
    // 	recvcounts.push_back(1);
    // 	recvdispls.push_back(0);
    //   }
    //   matrix<int> M(nx, ny);
    //   comm_world.alltoallw(M_l.data(), sendl, sendcounts, senddispls,
    //    			   M.data(), recvl, recvcounts, recvdispls);
    //   for (int iy=0; iy<ny; ++iy) {
    // 	for (int ix=0; ix<nx; ++ix) {
    // 	  std::cout << static_cast<unsigned char>(M(ix, iy)+'A');
    // 	}
    // 	std::cout << '\n';
    //   }
    //   std::cout << '\n';
    // } else {
    //   for (int i=0; i<p; ++i) {
    // 	recvl.push_back(mpl::empty_layout<int>());
    // 	recvcounts.push_back(0);
    // 	recvdispls.push_back(0);
    //   }
    //   comm_world.alltoallw(M_l.data(), sendl, sendcounts, senddispls,
    // 			   reinterpret_cast<int *>(0), recvl, recvcounts, recvdispls);
    // }
  }
  return EXIT_SUCCESS;
}
