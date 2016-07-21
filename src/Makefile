INCLUDE_FLAG = -I include -I .
LINK_FLAGS = -lblas -llapack -lm -lgomp -lstdc++
OPT = -O3
CXX = g++
CXXFLAGS = $(INCLUDE_FLAG) $(OPT) -std=c++11 -fopenmp
RLIB_FLAGS = -fPIC -shared
SRC_FILES = $(filter-out main.cpp, $(wildcard *.cpp))

pfar.so: obj
	R CMD SHLIB -o pfar.so $(addsuffix .o, $(notdir $(basename $(SRC_FILES)))) $(LINK_FLAGS)

obj:
	$(CXX) $(CXXFLAGS) $(RLIB_FLAGS) -c $(SRC_FILES)

pfa: main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LINK_FLAGS)

.PHONY: clean

clean:
	rm -f pfa
