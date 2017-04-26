/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// See docs in ../ops/data_flow_ops.cc.

#include <deque>
#include <vector>
#include <hdf5.h>
#include <assert.h>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/fifo_queue.h"
#include "tensorflow/core/kernels/queue_base.h"
#include "tensorflow/core/kernels/queue_op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/common_runtime/dma_helper.h"

namespace tensorflow {

typedef FIFOQueue<std::deque<PersistentTensor> > SimpleFIFOQueue;

// Defines a FIFOQueueOp, which produces a Queue (specifically, one
// backed by FIFOQueue) that persists across different graph
// executions, and sessions. Running this op produces a single-element
// tensor of handles to Queues in the corresponding device.
class FIFOQueueOp : public TypedQueueOp {
 public:
  explicit FIFOQueueOp(OpKernelConstruction* context) : TypedQueueOp(context) {
    OP_REQUIRES_OK(context, context->GetAttr("shapes", &component_shapes_));
  }

 private:
  Status CreateResource(QueueInterface** ret) override
      EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    SimpleFIFOQueue* queue = 
      new SimpleFIFOQueue(capacity_, component_types_,
			  component_shapes_, cinfo_.name());
    return CreateTypedQueue(queue, ret);
  }

  std::vector<TensorShape> component_shapes_;
  TF_DISALLOW_COPY_AND_ASSIGN(FIFOQueueOp);
};

REGISTER_KERNEL_BUILDER(Name("FIFOQueue").Device(DEVICE_CPU), FIFOQueueOp);
REGISTER_KERNEL_BUILDER(Name("FIFOQueueV2").Device(DEVICE_CPU), FIFOQueueOp);

class HDF5QueueOp : public TypedQueueOp {
 public:
  explicit HDF5QueueOp(OpKernelConstruction* context) : TypedQueueOp(context) {
    context->GetAttr("filename", &filename);
    context->GetAttr("datasets", &datasets);
    context->GetAttr("overwrite", &overwrite);
  }

 private:
  
  // Mimic std::deque
  class HDF5SubQueue {
  public:
    HDF5SubQueue(hid_t dataset, TensorShape s, DataType dt) :
      dataset(dataset), s(s), dt(dt) {  }
    
    size_t size() const {
      return write_ind - read_ind;
    }

    PersistentTensor operator[](size_t pos) {
      PersistentTensor pt;
      //ctx->allocate_temp(dt, s, &t);
      const Tensor t(pt);
      
      hid_t dataspace, memspace;
      prep_H5space(dataset, t.shape(), dataspace, memspace, pos);
      H5Dread(dataset, get_type(dt), 
	      memspace, dataspace, H5P_DEFAULT,
	      const_cast<void*>(DMAHelper::base(&t)));	  
      return pt;
    }

    void pop_front() {
      ++read_ind;
    }

    void push_back(const PersistentTensor &pt) {
      const Tensor t(pt);
      hid_t dataspace, memspace;
      
      prep_H5space(dataset, t.shape(), dataspace, memspace, write_ind);
      // Write to Hyperslab
      H5Dwrite(dataset, get_type(t.dtype()), 
	       memspace, dataspace, H5P_DEFAULT, 
	       const_cast<void*>(DMAHelper::base(&t)));
      
      std::vector<hsize_t> dims(t.dims()+1);
      H5Sget_simple_extent_dims(H5Dget_space(dataset), dims.data(), NULL);
      dims[0]++;
      H5Dset_extent(dataset, dims.data());
      ++write_ind;
    }

    void push_front(const PersistentTensor &pt) {
      // I think FIFOQueue just uses read_ind to re-enqueue things the same data
      --read_ind;
    }
    
    void prep_H5space(const hid_t &dset, const TensorShape &shp,
		      hid_t &dspace, hid_t &mspace, int row_number) {
      dspace = H5Dget_space(dset);
      
      std::vector<hsize_t> offset(shp.dims()+1, 0), count(shp.dims());

      offset[0] = row_number;
      for (int j=0;j<shp.dims();j++)
	count[j] = shp.dim_size(j);
      mspace = H5Screate_simple(shp.dims(), count.data(), NULL);
      count.insert(count.begin(), 1); // 1 row

      dspace = H5Dget_space(dset);
      herr_t status = H5Sselect_hyperslab(dspace, H5S_SELECT_SET, offset.data(), 
					  NULL, count.data(), NULL);
      assert(status == 0);
      assert(shp.num_elements() == H5Sget_select_npoints(dspace));
      assert(shp.num_elements() == H5Sget_select_npoints(mspace));
    }

    static
    hid_t get_type(DataType t) {
      switch (t) {
      case DT_FLOAT:      return H5T_IEEE_F32LE;
      case DT_DOUBLE:     return H5T_IEEE_F64LE;
      case DT_INT8:       return H5T_STD_I8LE;
      case DT_INT16:      return H5T_STD_I16LE;
      case DT_INT32:      return H5T_STD_I32LE;
      case DT_INT64:      return H5T_STD_I64LE;
      case DT_UINT8:      return H5T_STD_U8LE;
      case DT_UINT16:     return H5T_STD_U16LE;
	//case DT_STRING:     return H5T_C_STRING; // TODO: Add String support. This doesn't build
      case DT_BOOL:       return H5T_NATIVE_HBOOL;
      case DT_COMPLEX64:  return H5T_IEEE_F64LE; // TODO: Fix
      case DT_COMPLEX128: return H5T_IEEE_F64LE; // TODO: Fix
      case DT_QINT8:      return H5T_STD_I8LE; // TODO: Figure these out
      case DT_QINT32:     return H5T_STD_I32LE;
      case DT_QUINT8:     return H5T_STD_U8LE;
      }
      return H5T_IEEE_F32LE;
    }

    hid_t dataset;
    hsize_t read_ind, write_ind;
    TensorShape s;
    DataType dt;
  };

  class HDF5Queue : public FIFOQueue<HDF5SubQueue> {
  public:
    HDF5Queue(int32 capacity, const DataTypeVector& component_dtypes,
	      const std::vector<TensorShape>& component_shapes,
	      const string& name) :
      FIFOQueue<HDF5SubQueue>(capacity, component_dtypes, component_shapes, name) { }


    Status MatchesNodeDef(const NodeDef& node_def) override {
      if (!MatchesNodeDefOp(node_def, "HDF5Queue").ok()) {
	return errors::InvalidArgument("Expected HDF5Queue, found ", node_def.op());
      }
      TF_RETURN_IF_ERROR(MatchesNodeDefCapacity(node_def, capacity_));
      TF_RETURN_IF_ERROR(MatchesNodeDefTypes(node_def));
      TF_RETURN_IF_ERROR(MatchesNodeDefShapes(node_def));
      return Status::OK();
    }
    
    Status Initialize(string filename, std::vector<string> datasets, bool overwrite) {
      if (component_dtypes_.empty()) {
	return errors::InvalidArgument("Empty component types for queue ", name_);
      }
      if (!component_shapes_.empty() &&
	  component_dtypes_.size() != component_shapes_.size()) {
	return errors::InvalidArgument(
				       "Different number of component types.  ", "Types: ",
				       DataTypeSliceString(component_dtypes_), ", Shapes: ",
				       ShapeListString(component_shapes_));
      }

      mutex_lock lock(mu_);
      queues_.reserve(num_components());

      if (Env::Default()->FileExists(filename).ok()) {
	// Open file
	file = H5Fopen(filename.c_str(), 
		       (overwrite ? H5F_ACC_TRUNC : H5F_ACC_RDWR), 
		       H5P_DEFAULT);
      } else {
	file = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, 
			 H5P_DEFAULT, H5P_DEFAULT);
      }
      assert(file >= 0); // File opened properly

      for (int i = 0; i < num_components(); ++i) {
	size_t pos = datasets[i].rfind("/");
	if (pos != string::npos) {
	  string path = datasets[i].substr(0, pos);
	  status = H5Eset_auto2(0, NULL, NULL);
	  status = H5Gget_objinfo(file, path.c_str(), 0, NULL);
	  if (status != 0) { // Path doesn't exist, create it
	    hid_t lcpl = H5Pcreate(H5P_LINK_CREATE);
	    H5Pset_create_intermediate_group(lcpl, 1);
	    assert(H5Gcreate(file, path.c_str(), lcpl, H5P_DEFAULT, H5P_DEFAULT) > 0);
	  }
	}

	hid_t dset_id;
	if (H5Lexists(file, datasets[i].c_str(), H5P_DEFAULT) > 0) {
	  dset_id = H5Dopen2(file, datasets[i].c_str(), H5P_DEFAULT);
	  hid_t dspace = H5Dget_space(dset_id);
	  assert(H5Sget_simple_extent_ndims(dspace) == component_shapes_[i].dims()+1);
	} else {
	  const TensorShape &s = component_shapes_[i];
	  std::vector<hsize_t> dims(s.dims());
	  for (int i=0;i<s.dims();i++)
	    dims[i] = s.dim_size(i);
	  dims.insert(dims.begin(), 1);
	  std::vector<hsize_t> maxdims(dims), chunkdims(dims);
	  maxdims[0] = H5S_UNLIMITED;
	  chunkdims[0] = 5; // Design choice

	  hid_t prop = H5Pcreate(H5P_DATASET_CREATE);
	  H5Pset_chunk(prop, s.dims()+1, chunkdims.data());
	  assert(prop > 0);
	  hid_t dspace = H5Screate_simple(s.dims()+1, dims.data(), maxdims.data());
	  assert(dspace > 0);
	  assert(H5Sget_simple_extent_ndims(dspace) == dims.size());

	  dset_id = H5Dcreate2(file, datasets[i].c_str(), 
			       HDF5SubQueue::get_type(component_dtypes_[i]), dspace,
			       H5P_DEFAULT, prop, H5P_DEFAULT);
	}
	assert(dset_id >= 0);
	
	queues_.push_back(HDF5SubQueue(dset_id, this->component_shapes_[i], 
				       this->component_dtypes_[i]));
      }
      return Status::OK();
    }
    
  private:
    hid_t file;
  };

  Status CreateResource(QueueInterface** ret) override
      EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    HDF5Queue* queue = new HDF5Queue(capacity_, component_types_,
				     component_shapes_, cinfo_.name());
    if (queue == nullptr) {
      return errors::ResourceExhausted("Failed to allocate queue.");
    }
    *ret = queue;
    return queue->Initialize(filename, datasets, overwrite);
  }

  string filename;
  std::vector<string> datasets;
  bool overwrite;
  std::vector<TensorShape> component_shapes_;

  TF_DISALLOW_COPY_AND_ASSIGN(HDF5QueueOp);
};

REGISTER_OP("HDF5Queue")
  .Output("handle: resource")
  .Attr("filename: string")
  .Attr("datasets: list(string)")
  .Attr("overwrite: bool = false")
  .Attr("component_types: list(type) >= 0 = []")
  .Attr("shapes: list(shape) >= 0 = []")
  .Attr("shared_name: string = ''")
  .Attr("container: string = ''")
  .Attr("capacity: int = -1")
  .SetIsStateful()
  .SetShapeFn(TwoElementOutput);


REGISTER_KERNEL_BUILDER(Name("HDF5Queue").Device(DEVICE_CPU), HDF5QueueOp);

}  // namespace tensorflow
