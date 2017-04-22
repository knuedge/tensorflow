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

#ifndef TENSORFLOW_KERNELS_FIFO_QUEUE_H_
#define TENSORFLOW_KERNELS_FIFO_QUEUE_H_

#include <algorithm>
#include <deque>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/typed_queue.h"
#include "tensorflow/core/kernels/queue_base.h"
#include "tensorflow/core/framework/queue_interface.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

template <typename SubQueue>
class FIFOQueue : public TypedQueue<SubQueue> {
 public:
  typedef typename TypedQueue<SubQueue>::Tuple Tuple;
  typedef typename TypedQueue<SubQueue>::DoneCallback DoneCallback;
  typedef typename TypedQueue<SubQueue>::CallbackWithTuple CallbackWithTuple;
  typedef typename TypedQueue<SubQueue>::Attempt Attempt;
  typedef typename TypedQueue<SubQueue>::RunResult RunResult;

  FIFOQueue(int32 capacity, const DataTypeVector& component_dtypes,
            const std::vector<TensorShape>& component_shapes,
            const string& name);

  // Implementations of QueueInterface methods --------------------------------

  void TryEnqueue(const Tuple& tuple, OpKernelContext* ctx,
                  DoneCallback callback) override;
  void TryEnqueueMany(const Tuple& tuple, OpKernelContext* ctx,
                      DoneCallback callback) override;
  void TryDequeue(OpKernelContext* ctx, CallbackWithTuple callback) override;
  void TryDequeueMany(int num_elements, OpKernelContext* ctx,
                      bool allow_small_batch,
                      CallbackWithTuple callback) override;
  Status MatchesNodeDef(const NodeDef& node_def) override;

  int32 size() override;

 protected:
  ~FIFOQueue() override {}

  // Helper for dequeuing a single element from queues_.
  void DequeueLocked(OpKernelContext* ctx, Tuple* tuple)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  static Status GetElementComponentFromBatch(const Tuple& tuple, int64 index,
                                             int component,
                                             OpKernelContext* ctx,
                                             PersistentTensor* out_element);

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(FIFOQueue);
};

template <typename SubQueue>
FIFOQueue<SubQueue>::FIFOQueue(int capacity, const DataTypeVector& component_dtypes,
                     const std::vector<TensorShape>& component_shapes,
                     const string& name)
    : TypedQueue<SubQueue>(capacity, component_dtypes, component_shapes, name) {}

template <typename SubQueue>
void FIFOQueue<SubQueue>::DequeueLocked(OpKernelContext* ctx, Tuple* tuple) {
  DCHECK_GT(this->queues_[0].size(), size_t{0});
  (*tuple).reserve(this->num_components());
  for (int i = 0; i < this->num_components(); ++i) {
    (*tuple).push_back(*this->queues_[i][0].AccessTensor(ctx));
    this->queues_[i].pop_front();
  }
}

template <typename SubQueue>
int32 FIFOQueue<SubQueue>::size() {
  mutex_lock lock(this->mu_);
  return this->queues_[0].size();
}

template <typename SubQueue>
void FIFOQueue<SubQueue>::TryEnqueue(const Tuple& tuple, OpKernelContext* ctx,
                           DoneCallback callback) {
  CancellationManager* cm = ctx->cancellation_manager();
  CancellationToken token = cm->get_cancellation_token();
  bool already_cancelled;
  {
    mutex_lock l(this->mu_);
    already_cancelled = !cm->RegisterCallback(
        token, [this, cm, token]() { this->Cancel(this->kEnqueue, cm, token); });
    if (!already_cancelled) {
      this->enqueue_attempts_.emplace_back(
          1, callback, ctx, cm, token,
          [tuple, this](Attempt* attempt) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
            if (this->closed_) {
              attempt->context->SetStatus(
                  errors::Cancelled("FIFOQueue '", this->name_, "' is closed."));
              return this->kComplete;
            }
            if (this->queues_[0].size() < static_cast<size_t>(this->capacity_)) {
              for (int i = 0; i < this->num_components(); ++i) {
                this->queues_[i].push_back(PersistentTensor(tuple[i]));
              }
              return this->kComplete;
            } else {
              return this->kNoProgress;
            }
          });
    }
  }
  if (!already_cancelled) {
    this->FlushUnlocked();
  } else {
    ctx->SetStatus(errors::Cancelled("Enqueue operation was cancelled"));
    callback();
  }
}

template <typename SubQueue>
/* static */
Status FIFOQueue<SubQueue>::GetElementComponentFromBatch(const FIFOQueue<SubQueue>::Tuple& tuple,
                                               int64 index, int component,
                                               OpKernelContext* ctx,
                                               PersistentTensor* out_tensor) {
  TensorShape element_shape(tuple[component].shape());
  element_shape.RemoveDim(0);
  Tensor* element_access = nullptr;
  TF_RETURN_IF_ERROR(ctx->allocate_persistent(
      tuple[component].dtype(), element_shape, out_tensor, &element_access));
  TF_RETURN_IF_ERROR(
  FIFOQueue<SubQueue>::CopySliceToElement(tuple[component], element_access, index));
  return Status::OK();
}

template <typename SubQueue>
void FIFOQueue<SubQueue>::TryEnqueueMany(const Tuple& tuple, OpKernelContext* ctx,
                               DoneCallback callback) {
  const int64 batch_size = tuple[0].dim_size(0);
  if (batch_size == 0) {
    callback();
    return;
  }

  CancellationManager* cm = ctx->cancellation_manager();
  CancellationToken token = cm->get_cancellation_token();
  bool already_cancelled;
  {
    mutex_lock l(this->mu_);
    already_cancelled = !cm->RegisterCallback(
        token, [this, cm, token]() { this->Cancel(this->kEnqueue, cm, token); });
    if (!already_cancelled) {
      this->enqueue_attempts_.emplace_back(
          batch_size, callback, ctx, cm, token,
          [tuple, this](Attempt* attempt) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
            if (this->closed_) {
              attempt->context->SetStatus(
                  errors::Cancelled("FIFOQueue '", this->name_, "' is closed."));
              return this->kComplete;
            }
            RunResult result = this->kNoProgress;
            while (this->queues_[0].size() < static_cast<size_t>(this->capacity_)) {
              result = this->kProgress;
              const int64 index =
                  tuple[0].dim_size(0) - attempt->elements_requested;
              for (int i = 0; i < this->num_components(); ++i) {
                PersistentTensor element;
                attempt->context->SetStatus(GetElementComponentFromBatch(
                    tuple, index, i, attempt->context, &element));
                if (!attempt->context->status().ok()) return this->kComplete;
                this->queues_[i].push_back(element);
              }
              --attempt->elements_requested;
              if (attempt->elements_requested == 0) {
                return this->kComplete;
              }
            }
            return result;
          });
    }
  }
  if (!already_cancelled) {
    this->FlushUnlocked();
  } else {
    ctx->SetStatus(errors::Cancelled("Enqueue operation was cancelled"));
    callback();
  }
}

template <typename SubQueue>
void FIFOQueue<SubQueue>::TryDequeue(OpKernelContext* ctx, CallbackWithTuple callback) {
  CancellationManager* cm = ctx->cancellation_manager();
  CancellationToken token = cm->get_cancellation_token();
  bool already_cancelled;
  {
    mutex_lock l(this->mu_);
    already_cancelled = !cm->RegisterCallback(
        token, [this, cm, token]() { this->Cancel(this->kDequeue, cm, token); });
    if (!already_cancelled) {
      // TODO(josh11b): This makes two copies of callback, avoid this if possible.
      this->dequeue_attempts_.emplace_back(
          1, [callback]() { callback(Tuple()); }, ctx, cm, token,
          [callback, this](Attempt* attempt) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
            const int64 queue_size = this->queues_[0].size();
            if (this->closed_ && queue_size == 0) {
              attempt->context->SetStatus(errors::OutOfRange(
                  "FIFOQueue '", this->name_, "' is closed and has ",
                  "insufficient elements (requested ", 1, ", current size ",
                  queue_size, ")"));
              return this->kComplete;
            }
            if (queue_size > 0) {
              Tuple tuple;
              DequeueLocked(attempt->context, &tuple);
              attempt->done_callback = [callback, tuple]() { callback(tuple); };
              return this->kComplete;
            } else {
              return this->kNoProgress;
            }
          });
    }
  }
  if (!already_cancelled) {
    this->FlushUnlocked();
  } else {
    ctx->SetStatus(errors::Cancelled("Dequeue operation was cancelled"));
    callback(Tuple());
  }
}

template <typename SubQueue>
void FIFOQueue<SubQueue>::TryDequeueMany(int num_elements, OpKernelContext* ctx,
                               bool allow_small_batch,
                               CallbackWithTuple callback) {
  if (!this->specified_shapes()) {
    ctx->SetStatus(errors::InvalidArgument(
        "FIFOQueue's DequeueMany and DequeueUpTo require the "
        "components to have specified shapes."));
    callback(Tuple());
    return;
  }
  if (num_elements == 0) {
    Tuple tuple;
    tuple.reserve(this->num_components());
    for (int i = 0; i < this->num_components(); ++i) {
      // TODO(josh11b,misard): Switch to allocate_output().  Problem is
      // this breaks the abstraction boundary since we don't *really*
      // know if and how the Tensors in the tuple we pass to callback
      // correspond to the outputs of *ctx.  For example, the
      // ReaderRead Op uses TryDequeue() to get a filename out of a
      // queue that is used internally by the reader and is not
      // associated with any output of the ReaderRead.
      // mrry@ adds:
      // Maybe we need to pass a std::function<Tensor*(...)> (or
      // better signature) that calls the appropriate allocator
      // function in addition to ctx?  (Or support a shim Allocator
      // that has an internal OpKernelContext*, and dispatches to the
      // appropriate method?)
      // misard@ adds:
      // I don't see that a std::function would help. The problem is
      // that at this point (allocation time) the system doesn't know
      // what is going to happen to the element read out of the
      // queue. As long as we keep the generality that TensorFlow Ops
      // do their own dynamic allocation in arbitrary C++ code, we
      // need to preserve robustness to allocating output Tensors with
      // the 'wrong' attributes, and fixing up with a copy. The only
      // improvement I can see here in the future would be to support
      // an optimized case where the queue 'knows' what attributes to
      // use, and plumbs them through here.
      Tensor element;
      Status status = ctx->allocate_temp(this->component_dtypes_[i],
                                         this->ManyOutShape(i, 0), &element);
      if (!status.ok()) {
        ctx->SetStatus(status);
        callback(Tuple());
        return;
      }
      tuple.emplace_back(element);
    }
    callback(tuple);
    return;
  }

  CancellationManager* cm = ctx->cancellation_manager();
  CancellationToken token = cm->get_cancellation_token();
  bool already_cancelled;
  {
    mutex_lock l(this->mu_);
    already_cancelled = !cm->RegisterCallback(
        token, [this, cm, token]() { this->Cancel(this->kDequeue, cm, token); });
    if (!already_cancelled) {
      // TODO(josh11b): This makes two copies of callback, avoid this if possible.
      this->dequeue_attempts_.emplace_back(
          num_elements, [callback]() { callback(Tuple()); }, ctx, cm, token,
          [callback, allow_small_batch, this](Attempt* attempt)
              EXCLUSIVE_LOCKS_REQUIRED(mu_) {
                int64 queue_size = this->queues_[0].size();

                if (this->closed_ && queue_size < attempt->elements_requested) {
                  // If we don't have enough for a full dequeue, we have
                  // to reset the attempt tuple.
                  if (!attempt->tuple.empty()) {
                    // Restore already-dequeued elements to the front of the
                    // queue.
                    for (int64 i = attempt->tuple[0].dim_size(0) -
                                   attempt->elements_requested - 1;
                         i >= 0; --i) {
                      for (int j = 0; j < this->num_components(); ++j) {
                        PersistentTensor element;
                        Status s = GetElementComponentFromBatch(
                            attempt->tuple, i, j, attempt->context, &element);
                        if (!s.ok()) {
                          attempt->context->SetStatus(
                              errors::DataLoss("Failed to restore element from "
                                               "partially-dequeued batch "
                                               "to FIFOQueue: ",
                                               s.error_message()));
                        }
                        this->queues_[j].push_front(element);
                      }
                    }
                  }
                  if (allow_small_batch && this->queues_[0].size() > 0) {
                    // Request all remaining elements in the queue.
                    queue_size = this->queues_[0].size();
                    attempt->tuple.clear();
                    attempt->elements_requested = queue_size;
                  } else {
                    if (allow_small_batch) {
                      // There may be some other attempts containing
                      // values.  If so, we'll yield and wait for them
                      // to add elements to the queue.
                      if (!this->enqueue_attempts_.empty()) return this->kProgress;
                    }
                    if (attempt->context->status().ok()) {
                      attempt->context->SetStatus(errors::OutOfRange(
                          "FIFOQueue '", this->name_, "' is closed and has ",
                          "insufficient elements (requested ",
                          attempt->elements_requested, ", current size ",
                          queue_size, ")"));
                    }
                    return this->kComplete;
                  }
                }

                RunResult result = this->kNoProgress;
                for (; queue_size > 0; --queue_size) {
                  if (attempt->tuple.empty()) {
                    // Only allocate tuple when we have something to dequeue
                    // so we don't use excessive memory when there are many
                    // blocked dequeue attempts waiting.
                    attempt->tuple.reserve(this->num_components());
                    for (int i = 0; i < this->num_components(); ++i) {
                      const TensorShape shape =
                          this->ManyOutShape(i, attempt->elements_requested);
                      Tensor element;
                      attempt->context->SetStatus(
                          attempt->context->allocate_temp(this->component_dtypes_[i],
                                                          shape, &element));
                      if (!attempt->context->status().ok()) return this->kComplete;
                      attempt->tuple.emplace_back(element);
                    }
                  }
                  result = this->kProgress;
                  Tuple tuple;
                  DequeueLocked(attempt->context, &tuple);
                  const int64 index = attempt->tuple[0].dim_size(0) -
                                      attempt->elements_requested;
                  for (int i = 0; i < this->num_components(); ++i) {
                    attempt->context->SetStatus(this->CopyElementToSlice(
                        tuple[i], &attempt->tuple[i], index));
                    if (!attempt->context->status().ok()) return this->kComplete;
                  }
                  tuple.clear();
                  --attempt->elements_requested;
                  if (attempt->elements_requested == 0) {
                    tuple = attempt->tuple;
                    attempt->done_callback = [callback, tuple]() {
                      callback(tuple);
                    };
                    return this->kComplete;
                  }
                }
                return result;
              });
    }
  }
  if (!already_cancelled) {
    this->FlushUnlocked();
  } else {
    ctx->SetStatus(errors::Cancelled("Dequeue operation was cancelled"));
    callback(Tuple());
  }
}

template <typename SubQueue>
Status FIFOQueue<SubQueue>::MatchesNodeDef(const NodeDef& node_def) {
  if (!this->MatchesNodeDefOp(node_def, "FIFOQueue").ok() &&
      !this->MatchesNodeDefOp(node_def, "FIFOQueueV2").ok()) {
    return errors::InvalidArgument("Expected FIFOQueue, found ", node_def.op());
  }
  TF_RETURN_IF_ERROR(this->MatchesNodeDefCapacity(node_def, this->capacity_));
  TF_RETURN_IF_ERROR(this->MatchesNodeDefTypes(node_def));
  TF_RETURN_IF_ERROR(this->MatchesNodeDefShapes(node_def));
  return Status::OK();
}

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_FIFO_QUEUE_H_
