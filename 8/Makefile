#!/usr/bin/env make

FILE=simpleCUDA

all:
	nvcc $(FILE).cu --compiler-bindir /usr/bin -O2  -o $(FILE)

clean:
	\rm -f $(FILE).linkinfo $(FILE) *~

