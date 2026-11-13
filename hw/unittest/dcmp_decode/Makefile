ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

LATENCY ?= 2

PROJECT := dcmp_decode

RTL_DIR := $(VORTEX_HOME)/hw/rtl
SRC_DIR := $(VORTEX_HOME)/hw/unittest/$(PROJECT)

LDFLAGS += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common -I$(SW_COMMON_DIR)
CXXFLAGS += -I$(THIRD_PARTY_DIR)/softfloat/source/include
CXXFLAGS += -DLATENCY=$(LATENCY)

PARAMS := -GLATENCY=$(LATENCY)

CONFIGS += -DSIMULATION

SRCS += $(SW_COMMON_DIR)/softfloat_ext.cpp
SRCS += $(SW_COMMON_DIR)/rvfloats.cpp
SRCS += $(SRC_DIR)/main.cpp

RTL_INCLUDE := -I$(ROOT_DIR)/sw -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/core -I$(SRC_DIR)
RTL_PKGS += $(RTL_DIR)/VX_gpu_pkg.sv
VL_FLAGS += -I$(ROOT_DIR)/hw

TOP := VX_dcmp_decode

include ../common.mk
