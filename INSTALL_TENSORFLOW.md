# Install Tensorflow with Cuda on macos

## Prepare

* Clone tensorflow and checkout `r1.11`
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

## Install

```bash
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg2
pip install /tmp/tensorflow_pkg2/tensorflow-1.11.0-cp36-cp36m-macosx_10_13_x86_64.whl
```

In case of `image not found` errors when executing: Update rnames (update paths):

```bash
cd /Users/ms/.pyenv//versions/3.6.8/envs/reinforced-learning/lib/python3.6/site-packages/tensorflow

install_name_tool -change @rpath/libcusolver.10.0.dylib /usr/local/cuda/lib/libcusolver.10.0.dylib -change @rpath/libcudart.10.0.dylib /usr/local/cuda/lib/libcudart.10.0.dylib -change @rpath/libcublas.10.0.dylib /usr/local/cuda/lib/libcublas.10.0.dylib python/_pywrap_tensorflow_internal.so

install_name_tool -change @rpath/libcudart.10.0.dylib /usr/local/cuda/lib/libcudart.10.0.dylib -change @rpath/libcublas.10.0.dylib /usr/local/cuda/lib/libcublas.10.0.dylib -change @rpath/libcudnn.7.dylib /usr/local/cuda/lib/libcudnn.7.dylib -change @rpath/libcufft.10.0.dylib /usr/local/cuda/lib/libcufft.10.0.dylib -change @rpath/libcurand.10.0.dylib /usr/local/cuda/lib/libcurand.10.0.dylib -change @rpath/libcudart.10.0.dylib /usr/local/cuda/lib/libcudart.10.0.dylib libtensorflow_framework.so
```

## Test

```bash
python3 -c "import tensorflow as tf; tf.enable_eager_execution(); print(tf.reduce_sum(tf.random_normal([1000, 1000])))"```

## References

* <https://qiita.com/anianinya/items/12b4b2c4f86155ca8403>
* <https://github.com/tensorflow/tensorflow/issues/19720>
