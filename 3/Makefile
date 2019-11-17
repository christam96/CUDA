
GCC = g++

GCCFLAGS = -c

NVCC = nvcc

SRCCC =

SRCCU = moveArrays.cu

NVCCFLAGS = -c -O2 --compiler-bindir /usr/bin

EXE = moveArrays

RM = rm -f

OBJ = $(SRCCC:.c=.o) $(SRCCU:.cu=.o)

all: $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $*.cu

clean:
	$(RM) *.o *~ *.linkinfo a.out *.log $(EXE)
