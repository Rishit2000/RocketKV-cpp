# Makefile for RocketKV llama.cpp Runner & Tests
LLAMA_DIR ?= ./llama.cpp
CXX = clang++
CPPFLAGS += -I. -I$(LLAMA_DIR)/include -I$(LLAMA_DIR)/ggml/include -I$(LLAMA_DIR)/vendor
CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra -Wno-unknown-pragmas
LDFLAGS += -L$(LLAMA_DIR)/build/bin
LDLIBS := -lllama -lggml -lggml-base -lggml-cpu
ifeq ($(shell uname -s),Darwin)
LDFLAGS += -Wl,-rpath,@loader_path/$(LLAMA_DIR)/build/bin
else
LDFLAGS += -Wl,-rpath,'$$ORIGIN/$(LLAMA_DIR)/build/bin'
endif

.PHONY: all test clean

all: test_rocket runner

test_rocket: test_rocket.cpp rocket.cpp rocket.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) test_rocket.cpp rocket.cpp -o $@

runner: runner.cpp rocket.cpp rocket_adapter.cpp rocket.h rocket_adapter.h $(LLAMA_DIR)/include/llama-rocket.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) runner.cpp rocket.cpp rocket_adapter.cpp -o $@ $(LDFLAGS) $(LDLIBS)

test: test_rocket
	./test_rocket

clean:
	rm -f runner test_rocket