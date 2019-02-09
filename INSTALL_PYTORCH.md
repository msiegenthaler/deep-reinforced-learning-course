# Install Pytorch with CUDA

* Install XCode Command Line Tools 9.4
* Clone pytorch
* Build it

  ```bash
  env TORCH_CUDA_ARCH_LIST=6.1 CUDA_HOME=/usr/local/cuda CUDA_CUDA_LIB=/usr/local/cuda/lib/libcuda.dylib python setup.py install
  ```

* Build will fail at 90-something percent
* Do `cp -R torch/lib/tmp_install/* torch`
* Build again, it should now suceed
* Test it

  ```python
  import torch
  torch.cuda.is_available() # should returns True
  torch.cuda.device_count() # returns 1
  ```
