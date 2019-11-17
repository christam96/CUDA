
GCC = g++

GCCFLAGS = -c

NVCC = nvcc

SRCCC =

SRCCU = arrayReversal_multiblock_fast.cu

NVCCFLAGS = -c -O2 --compiler-bindir /usr/bin

EXE = arrayReversal_multiblock_fast

RM = rm -f

OBJ = $(SRCCC:.c=.o) $(SRCCU:.cu=.o)

all: $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $*.cu

clean:
	$(RM) *.o *~ *.linkinfo a.out *.log $(EXE)
