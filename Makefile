# Release binaries and generated files live in build/.
LLAMA_DIR ?= ./llama.cpp
CXX = clang++
CPPFLAGS += -Iinclude -Ibuild -I$(LLAMA_DIR)/include -I$(LLAMA_DIR)/ggml/include -I$(LLAMA_DIR)/vendor
CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra -Wno-unknown-pragmas
LDFLAGS += -L$(LLAMA_DIR)/build/bin
LDLIBS := -lllama -lggml -lggml-base -lggml-cpu
HEADERS := $(wildcard include/*.h)
RUNNER_SOURCES := src/runner.cpp src/rocket.cpp src/rocket_adapter.cpp
ifeq ($(shell uname -s),Darwin)
LDFLAGS += -Wl,-rpath,@loader_path/../$(LLAMA_DIR)/build/bin
METAL_OBJ := build/rocket_metal.o
LDLIBS += -framework Metal -framework Foundation
else
LDFLAGS += -Wl,-rpath,'$$ORIGIN/../$(LLAMA_DIR)/build/bin'
METAL_OBJ := build/rocket_metal_stub.o
endif

.PHONY: all runner test_rocket test test_metal clean
all: runner test_rocket
runner: build/runner
test_rocket: build/test_rocket

build:
	mkdir -p $@

build/rocket_metal_source.h: src/metal/rocket_metal.metal | build
	python3 -c 'from pathlib import Path; Path("$@").write_text("static const char* rocket_metal_source = R\"MSL(" + Path("$<").read_text() + ")MSL\";\n")'

build/rocket_metal.o: src/metal/rocket_metal.mm $(HEADERS) build/rocket_metal_source.h $(LLAMA_DIR)/ggml/include/ggml-metal.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -fobjc-arc -c $< -o $@

build/rocket_metal_stub.o: src/metal/rocket_metal_stub.cpp $(HEADERS) | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

build/runner: $(RUNNER_SOURCES) $(HEADERS) $(METAL_OBJ) $(LLAMA_DIR)/include/llama-rocket.h | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(RUNNER_SOURCES) $(METAL_OBJ) -o $@ $(LDFLAGS) $(LDLIBS)

build/test_rocket: tests/test_rocket.cpp src/rocket.cpp include/rocket.h | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) tests/test_rocket.cpp src/rocket.cpp -o $@

test: build/test_rocket
	./build/test_rocket

build/test_metal: tests/test_metal.cpp src/rocket.cpp $(HEADERS) $(METAL_OBJ) | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) tests/test_metal.cpp src/rocket.cpp $(METAL_OBJ) -o $@ $(LDFLAGS) $(LDLIBS)

test_metal: build/test_metal
	./build/test_metal

clean:
	rm -f build/runner build/test_rocket build/test_metal build/rocket_metal.o build/rocket_metal_stub.o build/rocket_metal_source.h
