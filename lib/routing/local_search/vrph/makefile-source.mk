CXX       := g++
CXX_FLAGS := -std=c++17 -ggdb -Wall -O3

BIN     := bin
SRC     :=  $(filter-out src/module.cpp, $(wildcard src/*.cpp))
INCLUDE := inc

LIBRARIES   := 
EXECUTABLE  := VRPH-source


all:	$(BIN)/$(EXECUTABLE)

run:	clean all
	clear
	./$(BIN)/$(EXECUTABLE)

$(BIN)/$(EXECUTABLE): $(SRC)
	$(CXX) $(CXX_FLAGS) -I$(INCLUDE) $^ -o $@ $(LIBRARIES)

clean:
	-rm $(BIN)/$@
	
	
# o operations`$(PYTHON3)-config --extension-suffix` 
# -o $@ $(LIBRARIES)
