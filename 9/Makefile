GCC=g++
NVCC=nvcc
OPTLEVEL=-O2 
PROFILING=
LDFLAGS=-lcuda
CFLAGS=-c
EXE=scan
ENABLEDOUBLE=-arch=sm_13
##--maxrregcount 22
##OTHEROPTS=--ptxas-options -v
OTHEROPTS=

deg: atomic2.cu
	$(NVCC) -arch=sm_13 atomic2.cu --compiler-bindir /usr/bin -o deg

clean:
	rm -f deg *~ *.o 