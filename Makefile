CXX ?= c++
TARGET ?= Aphrodite
BUILD_DIR ?= build_hpc
MP_PRECISION_DIGITS10 ?= 100
USE_LAPACK ?= 0
USE_CHEB_SIMD ?= 0
NATIVE_ARCH ?= 0
LAPACK_LIBS ?= -llapack -lblas
BOOST_INCLUDE_DIR ?= $(firstword $(foreach d,/opt/homebrew/opt/boost/include /opt/homebrew/include /usr/local/include /usr/include,$(if $(wildcard $(d)/boost/multiprecision/cpp_dec_float.hpp),$(d))))
EIGEN_INCLUDE_DIR ?= $(firstword $(foreach d,/opt/homebrew/include/eigen3 /usr/local/include/eigen3 /usr/include/eigen3,$(if $(wildcard $(d)/Eigen/Dense),$(d))))

CPPFLAGS += -Iinclude -DMP_PRECISION_DIGITS10=$(MP_PRECISION_DIGITS10)
ifneq ($(strip $(BOOST_INCLUDE_DIR)),)
CPPFLAGS += -I$(BOOST_INCLUDE_DIR)
endif
ifneq ($(strip $(EIGEN_INCLUDE_DIR)),)
CPPFLAGS += -I$(EIGEN_INCLUDE_DIR)
endif
CXXFLAGS += -std=c++20 -O3 -Wall -Wextra -Wpedantic -MMD -MP
ifeq ($(NATIVE_ARCH),1)
CXXFLAGS += -march=native
endif
LDLIBS += -pthread
ifeq ($(USE_LAPACK),1)
CPPFLAGS += -DCDET_USE_LAPACK=1
ifeq ($(shell uname -s),Darwin)
LDLIBS += -framework Accelerate
else
LDLIBS += $(LAPACK_LIBS)
endif
endif
ifneq ($(USE_CHEB_SIMD),0)
CPPFLAGS += -DCDET_USE_CHEB_SIMD=$(USE_CHEB_SIMD)
endif

OBJ := $(BUILD_DIR)/main.o
.PHONY: all clean print-config
all: $(TARGET)
$(TARGET): $(OBJ)
	$(CXX) $(OBJ) -o $@ $(LDLIBS)
$(OBJ): src/main.cpp
	@mkdir -p "$(BUILD_DIR)"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
print-config:
	@printf '%s\n' 'CXX=$(CXX)' 'CPPFLAGS=$(CPPFLAGS)' 'CXXFLAGS=$(CXXFLAGS)' 'LDLIBS=$(LDLIBS)'
clean:
	rm -rf "$(BUILD_DIR)" "$(TARGET)"
-include $(OBJ:.o=.d)
