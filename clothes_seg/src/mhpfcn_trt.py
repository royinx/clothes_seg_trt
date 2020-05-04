import sys
sys.path.insert(0, '')
# sys.path.append('..')

import numpy
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import os

from .mhpfcn_base import MHPFCNBase

import json
from collections import namedtuple
def _json_object_hook(d): return namedtuple('cfg_model', d.keys())(*d.values())
def json2obj(data): return json.load(data, object_hook=_json_object_hook)

# Debugging
def var(var):
    def print_array(var):
        print(type(var), " shape:",var.shape,"range: [{},{}]".format(var.min(),var.max()))

    vtype = type(var)
    if vtype == numpy.ndarray:
        print_array(var)
    elif vtype == list:
        print(type(var),len(var))
        if type(var[0]) == numpy.ndarray:
            for array in var:
                print_array(array)
        else:
            print(type(var[0]))
    else:
        pass
    exit()
    return

# Simple helper data class that's a little nicer to use than a 2-tuple.

def read_json(obj, path):
    with open(path, 'r') as infile:
        data = json.load(infile)
        for key, value in data.items():
            setattr(obj, key, value)

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class MHPFCNTRT(MHPFCNBase):
    """docstring for Inference_TensorRT"""

    def __init__(self, model_config_path: str = 'clothes_seg/config/config_model.json'):
        super().__init__(model_config_path)
        self.__init_tensorRT()

    def __init_tensorRT(self):
        # init log
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        with open('clothes_seg/model/fcn_resnet18_mhp_640x360_batch1.trt', 'rb') as fin, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(fin.read())

        self.output_shapes = []
        self.input_shapes = []
        for binding in self.engine:
            if self.engine.binding_is_input(binding):
                self.input_shapes.append(tuple(
                    [self.engine.max_batch_size] + list(self.engine.get_binding_shape(binding))))
            else:
                self.output_shapes.append(tuple(
                    [self.engine.max_batch_size] + list(self.engine.get_binding_shape(binding))))
        assert len(self.input_shapes) == 1, 'Only one input data is supported.'
        self.input_shape = self.input_shapes[0]
        assert self.input_shape[2] == self.net_h and self.input_shape[3] == self.net_w

        # create executor
        self.executor = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = self.__allocate_buffers(
            self.engine)

    # TensorRT GPU Resource allocation
    def __allocate_buffers(self, engine):
        inputs = []
        outputs = []
        bindings = []
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(
                binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings

    def _inference(self, input_batch):
        # input_batch = numpy.array(input_batch, dtype=numpy.float32, order='C')

        # tic = perf_counter()
        # for _ in range(100):
        # input_batch = numpy.tile(input_img,[64,1,1,1])
        input_batch = numpy.ascontiguousarray(input_batch)
        # input_batch = numpy.array(input_batch, dtype=numpy.float32, order='C')
        self.inputs[0].host = input_batch
        [cuda.memcpy_htod(inp.device, inp.host) for inp in self.inputs]
        self.executor.execute(
            batch_size=self.engine.max_batch_size, bindings=self.bindings)
        [cuda.memcpy_dtoh(output.host, output.device)
         for output in self.outputs]
        outputs = [out.host for out in self.outputs]

        outputs = [numpy.squeeze(output.reshape(shape))
                   for output, shape in zip(outputs, self.output_shapes)]
        return numpy.array(outputs)


# class MHPFCNTRTIS(MHPFCNBase):
#     def __init__(self, model_config_path: str = '/py/clothes_seg/config/config_model.json'):
#         super().__init__(model_config_path)

#     def infer(self, data: list, split_output: bool = True) -> (dict, float):
#         start_time = time.time()
#         res = self.infer_ctx.run(
#             self._prepare_input(data),
#             self._output_format
#         )
#         duration = time.time() - start_time
#         if split_output:
#             res = self._split_result(res)
#         return (res, duration)
