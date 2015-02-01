MPL_HEADERS = $(shell ls ../mpl/*hpp)
BINS = hello_world blocking blocking_vector probe nonblocking nonblocking_mult standard_types struct gather collective reduce_lcm layouts subarray vibrating_string_mpi vibrating_string_mpl
CC = mpicc
CXX = mpic++
#CXXFLAGS = -std=c++98 -pedantic -Wall -Wno-long-long
CCFLAGS = -std=c99 -Wall -O3
CXXFLAGS = -std=c++11 -Wall -O3 -DMPL_DEBUG

all: $(BINS)

hello_world: hello_world.cc $(MPL_HEADERS)
	$(CXX) $(CXXFLAGS) -o hello_world -I .. hello_world.cc

blocking: blocking.cc $(MPL_HEADERS)
	$(CXX) $(CXXFLAGS) -o blocking -I .. blocking.cc

blocking_vector: blocking_vector.cc $(MPL_HEADERS)
	$(CXX) $(CXXFLAGS) -o blocking_vector -I .. blocking_vector.cc

probe: probe.cc $(MPL_HEADERS)
	$(CXX) $(CXXFLAGS) -o probe -I .. probe.cc

nonblocking: nonblocking.cc $(MPL_HEADERS)
	$(CXX) $(CXXFLAGS) -o nonblocking -I .. nonblocking.cc

nonblocking_mult: nonblocking_mult.cc $(MPL_HEADERS)
	$(CXX) $(CXXFLAGS) -o nonblocking_mult -I .. nonblocking_mult.cc

standard_types: standard_types.cc $(MPL_HEADERS)
	$(CXX) $(CXXFLAGS) -o standard_types -I .. standard_types.cc

struct: struct.cc $(MPL_HEADERS)
	$(CXX) $(CXXFLAGS) -o struct -I .. struct.cc

gather: gather.cc $(MPL_HEADERS)
	$(CXX) $(CXXFLAGS) -o gather -I .. gather.cc

collective: collective.cc $(MPL_HEADERS)
	$(CXX) $(CXXFLAGS) -o collective -I .. collective.cc

reduce_lcm: reduce_lcm.cc $(MPL_HEADERS)
	$(CXX) $(CXXFLAGS) -o reduce_lcm -I .. reduce_lcm.cc

layouts: layouts.cc $(MPL_HEADERS)
	$(CXX) $(CXXFLAGS) -o layouts -I .. layouts.cc

subarray: subarray.cc $(MPL_HEADERS)
	$(CXX) $(CXXFLAGS) -o subarray -I .. subarray.cc

vibrating_string_mpi: vibrating_string_mpi.c
	$(CC) $(CCFLAGS) -o vibrating_string_mpi -I .. vibrating_string_mpi.c -lm

vibrating_string_mpl: vibrating_string_mpl.cc
	$(CXX) $(CXXFLAGS) -o vibrating_string_mpl -I .. vibrating_string_mpl.cc

clean:
	rm -f $(BINS) *.o
