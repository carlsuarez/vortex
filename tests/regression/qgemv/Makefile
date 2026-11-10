ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := qgemv

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp $(SRC_DIR)/host/repack.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

# ggml from llama.cpp's CPU build beside this repo: quantization and the reference check
LLAMA_DIR := $(VORTEX_HOME)/../llama.cpp
CXXFLAGS += -isystem $(LLAMA_DIR)/ggml/include
LDFLAGS += -L$(LLAMA_DIR)/build-cpu/bin -lggml-base -Wl,-rpath,$(LLAMA_DIR)/build-cpu/bin

KERNEL_LIB := vortex2

include ../common.mk