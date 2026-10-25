# Makefile for RocketKV llama.cpp Runner & Tests
LLAMA_DIR := ./llama.cpp
CXX := clang++
CXXFLAGS := -I. -I$(LLAMA_DIR)/include -I$(LLAMA_DIR)/ggml/include -I$(LLAMA_DIR)/vendor -std=c++17 -O3
LDFLAGS := -L$(LLAMA_DIR)/build/bin -lllama -lggml -lggml-base -lggml-cpu -lggml-metal -Wl,-rpath,@loader_path/$(LLAMA_DIR)/build/bin -framework Foundation -framework Metal -framework MetalPerformanceShaders

all: test_rocket runner

test_rocket: test_rocket.cpp rocket.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

runner: runner.cpp rocket.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -f runner test_rocket