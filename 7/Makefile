memset:	memset.cu
	nvcc -O2 --compiler-bindir /usr/bin  memset.cu -o memset

run:
	./memset

clean:
	rm -f *.out *.linkinfo memtest *.log memset *~
