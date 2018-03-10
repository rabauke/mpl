#define BOOST_TEST_MODULE icollectivev

#include <boost/test/included/unit_test.hpp>
#include <mpl/mpl.hpp>

template<typename T>
bool iscatterv_test() {
  const mpl::communicator &comm_world=mpl::environment::comm_world();
  const int N=(comm_world.size()*comm_world.size()+comm_world.size())/2;
  std::vector<T> v1(N), v2(N);
  std::iota(begin(v1), end(v1), 1);
  mpl::layouts<T> l;
  for (int i=0, i_end=comm_world.size(), offset=0; i<i_end; ++i) {
    l.push_back(mpl::indexed_layout<T>({{i+1, offset}}));
    offset+=i+1;
  }
  if (comm_world.rank()==0) {
    auto r{comm_world.iscatterv(0, v1.data(), l, v2.data(), l[0])};
    r.wait();
  } else {
    auto r{comm_world.iscatterv(0, v2.data(), l[comm_world.rank()])};
    r.wait();
  }
  for (int i=0, i_end=comm_world.size(), k=0; i<i_end; ++i)
    for (int j=0, j_end=i+1; j<j_end; ++j, ++k)
      if (i==comm_world.rank())
        if (v1[k]!=v2[k])
          return false;
  return true;
}

template<typename T>
bool igatherv_test() {
  const mpl::communicator &comm_world=mpl::environment::comm_world();
  const int N=(comm_world.size()*comm_world.size()+comm_world.size())/2;
  std::vector<T> v1(N), v2(N);
  std::iota(begin(v1), end(v1), 1);
  mpl::layouts<T> l;
  for (int i=0, i_end=comm_world.size(), offset=0; i<i_end; ++i) {
    l.push_back(mpl::indexed_layout<T>({{i+1, offset}}));
    offset+=i+1;
  }
  if (comm_world.rank()==0) {
    auto r{comm_world.igatherv(0, v1.data(), l[0], v2.data(), l)};
    r.wait();
    if (v1!=v2)
      return false;
  } else {
    auto r{comm_world.igatherv(0, v1.data(), l[comm_world.rank()])};
    r.wait();
  }
  return true;
}

template<typename T>
bool iallgatherv_test() {
  const mpl::communicator &comm_world=mpl::environment::comm_world();
  const int N=(comm_world.size()*comm_world.size()+comm_world.size())/2;
  std::vector<T> v1(N), v2(N);
  std::iota(begin(v1), end(v1), 1);
  mpl::layouts<T> l;
  for (int i=0, i_end=comm_world.size(), offset=0; i<i_end; ++i) {
    l.push_back(mpl::indexed_layout<T>({{i+1, offset}}));
    offset+=i+1;
  }
  auto r{comm_world.iallgatherv(v1.data(), l[comm_world.rank()], v2.data(), l)};
  r.wait();
  return v1==v2;
}

template<typename T>
bool ialltoallv_test() {
  const mpl::communicator &comm_world=mpl::environment::comm_world();
  std::vector<T> v_send, v_recv((comm_world.size()*comm_world.size()+comm_world.size())/2), v_expected;
  mpl::layouts<T> l_send, l_recv;
  for (int i=0, i_end=comm_world.size(); i<i_end; ++i)
    for (int j=0, j_end=comm_world.rank()+1; j<j_end; ++j)
      v_send.push_back(comm_world.rank()+1+i);
  for (int i=0, i_end=comm_world.size(); i<i_end; ++i) {
    l_send.push_back(mpl::indexed_layout<T>({{comm_world.rank()+1, (comm_world.rank()+1)*i}}));
    l_recv.push_back(mpl::indexed_layout<T>({{i+1, (i*i+i)/2}}));
  }
  auto r{comm_world.ialltoallv(v_send.data(), l_send, v_recv.data(), l_recv)};
  r.wait();
  for (int i=0, i_end=comm_world.size(); i<i_end; ++i)
    for (int j=0, j_end=i+1; j<j_end; ++j)
      v_expected.push_back(i+1+comm_world.rank());
  return v_recv==v_expected;
}


BOOST_AUTO_TEST_CASE(icollectivev) {
  BOOST_TEST(iscatterv_test<double>());
  BOOST_TEST(igatherv_test<double>());
  BOOST_TEST(iallgatherv_test<double>());
  BOOST_TEST(ialltoallv_test<double>());
}
