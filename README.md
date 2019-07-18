# TRE - TensorRT Running Engine

This is a tiny wrapper around [TensorRT](https://developer.nvidia.com/tensorrt) python API which loads a serialized TensorRT engine and runs inferences. It makes the inference process simpler. It somehow supplements what [ONNX-TensorRT](https://github.com/onnx/onnx-tensorrt) misses.

## Installation

### Dependencies

- [TensorRT](https://developer.nvidia.com/tensorrt) >= 5.1.5.0
- NumPy

Before installing TensorRT, you may need to install cuDNN and PyCUDA. See [Installing cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/) and [Installing PyCUDA](https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html#installing-pycuda). Follow the instructions to install TensorRT carefully. Make Sure the TensorRT lib is in your `LD_LIBRARY_PATH`.

#### Download the code

Clone the code from GitHub.

```bash
git clone https://github.com/ChengjieWU/TRE.git
```

Install the TRE wheel file.

```bash
cd TRE
python setup.py sdist bdist_wheel
pip install dist/TRE-0.0.1-py3-none-any.whl
```

## Usage

The TensorRT Running Engine can be used in Python as follows:

```python
from TRE import Engine
import numpy as np

engine = Engine("/path/to/serialized/TensorRT/engine", "CUDA:0")
input_data = np.random.random(size=(32, 3, 224, 224)).astype(np.float32)
output_data = engine.run(input_data)
print(output_data)
print(output_data.shape)
```

