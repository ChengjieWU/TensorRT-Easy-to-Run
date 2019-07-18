# HACK Should look for a better way/place to do this
from ctypes import cdll, c_char_p
from typing import NewType

import numpy as np
import tensorrt as trt

from .tensorrt_engine import Engine


libcudart = cdll.LoadLibrary('libcudart.so')
libcudart.cudaGetErrorString.restype = c_char_p


def cudaSetDevice(device_idx):
    ret = libcudart.cudaSetDevice(device_idx)
    if ret != 0:
        error_string = libcudart.cudaGetErrorString(ret)
        raise RuntimeError("cudaSetDevice: " + error_string)


def _tensorrt_version():
    return [int(n) for n in trt.__version__.split('.')]


# If TensorRT major is >= 5, then we use new Python bindings
USE_PYBIND = _tensorrt_version()[0] >= 5

if USE_PYBIND:
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
else:
    raise NotImplementedError("TensorRT < 5.0 is not supported")


class DeviceType(object):
    _Type = NewType('_Type', int)
    CPU = _Type(0)  # type: _Type
    CUDA = _Type(1)  # type: _Type


class Device(object):
    """
    Describes device type and device id
    syntax: device_type:device_id(optional)
    example: 'CPU', 'CUDA', 'CUDA:1'
    """
    def __init__(self, device):  # type: (Text) -> None
        options = device.split(':')
        self.type = getattr(DeviceType, options[0])
        self.device_id = 0
        if len(options) > 1:
            self.device_id = int(options[1])


class TensorRTEngine:
    def __init__(self, model, device):
        if not isinstance(device, Device):
            device = Device(device)
        self._set_device(device)

        self._logger = TRT_LOGGER
        with open(model, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            trt_engine = runtime.deserialize_cuda_engine(f.read())
        if trt_engine is None:
            raise RuntimeError("Failed to load TensorRT engine")
        self.engine = Engine(trt_engine)

    def _set_device(self, device):
        self.device = device
        assert (device.type == DeviceType.CUDA)
        cudaSetDevice(device.device_id)

    def run(self, inputs, **kwargs):
        """Execute the prepared engine and return the outputs as a named tuple.

        inputs -- Input tensor(s) as a Numpy array or list of Numpy arrays.
        """
        if isinstance(inputs, np.ndarray):
            inputs = [inputs]
        outputs = self.engine.run(inputs)
        # output_names = [output.name for output in self.engine.outputs]
        # return outputs[0], output_names
        return outputs[0]
