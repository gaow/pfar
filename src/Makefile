INCLUDE_FLAG = -I include -I .
LINK_FLAGS = -lopenblas -llapack -lm -lgomp -lstdc++
OPT = -O4
CXX = g++
CXXFLAGS = $(INCLUDE_FLAG) $(OPT) -std=c++11 -fopenmp
RLIB_FLAGS = -fPIC -shared
SRC_FILES = $(filter-out main.cpp, $(wildcard *.cpp))

pfar.so: obj
	R CMD SHLIB -o pfar.so $(addsuffix .o, $(notdir $(basename $(SRC_FILES)))) $(LINK_FLAGS)

obj:
	$(CXX) $(CXXFLAGS) $(RLIB_FLAGS) -c $(SRC_FILES)

play: main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LINK_FLAGS)

.PHONY: clean

clean:
	rm -f pfa
