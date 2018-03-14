#define BOOST_TEST_MODULE cart_communicator

#include <boost/test/included/unit_test.hpp>
#include <mpl/mpl.hpp>

bool cart_communicator_test() {
  const mpl::communicator &comm_world=mpl::environment::comm_world();
  mpl::cart_communicator::sizes sizes({{ 0, mpl::cart_communicator::periodic },
                                       { 0, mpl::cart_communicator::nonperiodic }});
  mpl::cart_communicator comm_c(comm_world,
                                mpl::dims_create(comm_world.size(), sizes));
  if (comm_c.dim()!=2)
    return false;
  int rank{ comm_c.rank() };
  int size{ comm_c.size() };
  auto coords{ comm_c.coords() };
  if (comm_c.rank(coords)!=rank)
    return false;
  auto dims{ comm_c.dims() };
  if (dims[0]*dims[1]!=comm_c.size())
    return false;
  auto p{ comm_c.is_periodic() };
  if (not(p[0]==mpl::cart_communicator::periodic and
          p[1]==mpl::cart_communicator::nonperiodic))
    return false;

  auto[source, dest]=comm_c.shift(0, 1);
  ++coords[0];
  if (coords[0]>=dims[0])
    coords[0]=0;
  int dest1{ comm_c.rank(coords) };
  coords[0]-=2;
  if (coords[0]<0)
    coords[0]+=dims[0];
  int source1{ comm_c.rank(coords) };
  if (not(source==source1 and dest==dest1))
    return false;
  {
    double x=1;
    std::vector<double> y(4, 0.);
    comm_c.neighbor_allgather(x, y.data());
    if ((y[0]!=0 and y[0]!=1)or(y[1]!=0 and y[1]!=1)or(y[2]!=0 and y[2]!=1)or(y[3]!=0 and y[3]!=1))
      return false;
  }
  {
    std::vector<double> x(4, rank+1.0);
    std::vector<double> y(4, 0.0);
    comm_c.neighbor_alltoall(x.data(), y.data());
    auto[n00, n01]=comm_c.shift(0, 1);
    auto[n10, n11]=comm_c.shift(1, 1);
    if (n00!=mpl::proc_null and y[0]!=n00+1.)
      return false;
    if (n01!=mpl::proc_null and y[1]!=n01+1.)
      return false;
    if (n10!=mpl::proc_null and y[2]!=n10+1.)
      return false;
    if (n11!=mpl::proc_null and y[3]!=n11+1.)
      return false;
  }
  {
    std::vector<double> x(4, rank+1.0);
    std::vector<double> y(4, 0.0);
    mpl::layouts<double> ls;
    ls.push_back(mpl::indexed_layout<double>({{ 1, 0 }}));
    ls.push_back(mpl::indexed_layout<double>({{ 1, 1 }}));
    ls.push_back(mpl::indexed_layout<double>({{ 1, 2 }}));
    ls.push_back(mpl::indexed_layout<double>({{ 1, 3 }}));
    comm_c.neighbor_alltoallv(x.data(), ls, y.data(), ls);
    auto[n00, n01]=comm_c.shift(0, 1);
    auto[n10, n11]=comm_c.shift(1, 1);
    if (n00!=mpl::proc_null and y[0]!=n00+1.)
      return false;
    if (n01!=mpl::proc_null and y[1]!=n01+1.)
      return false;
    if (n10!=mpl::proc_null and y[2]!=n10+1.)
      return false;
    if (n11!=mpl::proc_null and y[3]!=n11+1.)
      return false;
  }
  return true;
}

BOOST_AUTO_TEST_CASE(cart_communicator) {
  BOOST_TEST(cart_communicator_test());
}
