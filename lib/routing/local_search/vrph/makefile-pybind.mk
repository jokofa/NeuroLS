CXX       := g++
CXX_FLAGS := -std=c++17 -ggdb -Wall -O3

BIN     := bin
SRC     := $(filter-out src/source.cpp, $(wildcard src/*.cpp))
INCLUDE := inc
PYTHON3 := $(if $(PYTHON3),$(PYTHON3),python3)
OBJS = model.o #$(subst .cpp,.o, $(SRCS))

LIBRARIES   := 
EXECUTABLE  := VRPH


all:	$(BIN)/$(EXECUTABLE)

run:	clean all
	clear
	./$(BIN)/$(EXECUTABLE)

$(BIN)/$(EXECUTABLE): $(SRC)
	$(CXX) -shared $(CXX_FLAGS) -fPIC `$(PYTHON3) -m pybind11 --includes` -I$(INCLUDE) $^ -o $@.so

clean:
	-rm $(BIN)/*.so
	
	
# o operations`$(PYTHON3)-config --extension-suffix` 
# -o $@ $(LIBRARIES)
