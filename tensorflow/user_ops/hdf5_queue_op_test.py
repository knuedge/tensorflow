# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for custom user ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops.data_flow_ops import QueueBase, _as_type_list, _as_shape_list, _as_name_list

import os.path, sys, h5py, numpy
import tensorflow as tf

_mod = tf.load_op_library('./hdf5_queue_op.so')
_op_def_lib = _mod._op_def_lib

class HDF5Queue(QueueBase):
  def __init__(self, filename, datasets, dtypes, shapes, overwrite=False,
               capacity=100, names=None, name="hdf5_queue"):
    dtypes = _as_type_list(dtypes)
    shapes = _as_shape_list(shapes, dtypes)
    names = _as_name_list(names, dtypes)
    queue_ref = _op_def_lib.apply_op("HDF5Queue", filename=filename, datasets=datasets, 
                                     overwrite=overwrite,
                                     capacity=capacity, component_types=dtypes, shapes=shapes,
                                     name=name, container=None, shared_name=None)
    super(HDF5Queue, self).__init__(dtypes, shapes, 
                                    names, queue_ref)



class DuplicateOpTest(tf.test.TestCase):

  def testBasic(self):
    shapes = [[1], [2, 3], [1, 1]]
    dtypes = [numpy.int32, numpy.float32, numpy.float32]
    queue = HDF5Queue("/tmp/test.hdf5",
                      ['a', 'b/c', 'd/e/f'],
                      dtypes, shapes)

    with self.test_session() as sess:
      sess.run(queue.enqueue([numpy.zeros(s) for s in shapes]))
      sess.run(queue.enqueue([numpy.ones(s) for s in shapes]))
      sess.run(queue.enqueue([numpy.ones(s) for s in shapes]))
      sess.run(queue.enqueue([numpy.zeros(s) for s in shapes]))
      
      for a, b in zip(sess.run(queue.dequeue()), 
                      [numpy.zeros(s, dtype=d) 
                       for s, d in zip(shapes, dtypes)]):
        self.assertEqual(a, b)
      for a, b in zip(sess.run(queue.dequeue()), 
                      [numpy.ones(s, dtype=d) 
                       for s, d in zip(shapes, dtypes)]):
        self.assertEqual(a, b)
      for a, b in zip(sess.run(queue.dequeue()), 
                      [numpy.ones(s, dtype=d) 
                       for s, d in zip(shapes, dtypes)]):
        self.assertEqual(a, b)
      for a, b in zip(sess.run(queue.dequeue()), 
                      [numpy.zeros(s, dtype=d) 
                       for s, d in zip(shapes, dtypes)]):
        self.assertEqual(a, b)

if __name__ == '__main__':
  tf.test.main()
