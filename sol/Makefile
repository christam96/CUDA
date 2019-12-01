
GCC = g++

GCCFLAGS = -c

NVCC = nvcc

SRCCC =

SRCCU = min_plus_mult.cu

### NVCCFLAGS = -c -O2 --compiler-bindir /usr/bin//gcc-4.8
NVCCFLAGS = -c -O2 -ccbin cuda-gcc

EXE = min_plus_mult

RM = rm -f

OBJ = $(SRCCC:.c=.o) $(SRCCU:.cu=.o)

all: $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $*.cu

clean:
	$(RM) *.o *~ *.linkinfo a.out *.log $(EXE)
