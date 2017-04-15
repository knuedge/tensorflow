CC_FILES = hdf5_queue_op.cc
TF_INC=$(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
PB_LIB := /usr/local/lib/python2.7/dist-packages/external/protobuf/python/google/protobuf/pyext/_message.so
TF_LIB := $(shell find $(TF_INC)/../ -name "*.so")
HDF5_INC := /usr/include/hdf5/serial/

INC = -I $(TF_INC) -I $(HDF5_INC)
LIBS = $(TF_LIB) $(PB_LIB) -lhdf5

all:
	g++ -g -std=c++11 -shared $(CC_FILES) -o hdf5_queue_op.so -fPIC $(INC) -I/home/amacdonald/work/tflow/tensorflow/ $(LIBS)
