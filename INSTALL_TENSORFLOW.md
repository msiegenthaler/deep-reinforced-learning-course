# Install Tensorflow with Cuda on macos

## Prepare

* Clone tensorflow and checkout `r1.10`
* Download CUDA Toolkit 10.0 from nvidia
* Download cuDNN 7.x for cuda from nvidia
  * Copy lib and include to /usr/local/cuda/{include,lib}
* Download NCCL v2.x.x, for CUDA 10.0 (os-agnostic)
  * Copy lib and include to /usr/local/cuda/{include,lib}
  * link to tensorflow

    ```bash
    cd <tensorflow-clone>/third_party/nccl
    ln -s /Developer/NVIDIA/CUDA-10.0/include/nccl.h
    ```

* Install/Downgrade to Command Line Tools for Xcode 8.3.2

## Patch Tensorflow

* Remove `__align__(sizeof(T))` from the following files
  * `tensorflow/core/kernels/depthwise_conv_op_gpu.cu.cc`
  * `tensorflow/core/kernels/split_lib_gpu.cu.cc`
  * `tensorflow/core/kernels/concat_lib_gpu_impl.cu.cc`
* Remove `linkopts = [“-lgomp”]` from `tensorflow/third_party/gpus/cuda/BUILD.tpl`

## Configure

In the tensorflow git clone:

```bash
./configure
```

* Cuda support: yes
* cuDNN version: 7.4
* Device: 6.1 (for a GTX 10xx)

## Build

```bash
bazel build --config=cuda --config=opt --action_env PATH --action_env LD_LIBRARY_PATH --action_env DYLD_LIBRARY_PATH //tensorflow/tools/pip_package:build_pip_package
```

## References

* <https://qiita.com/anianinya/items/12b4b2c4f86155ca8403>