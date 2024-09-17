经过前面几轮的计算图优化，我们得到了一个粗粒度的计算图（TOSA dialect），它还需要进一步 lowering，而不同的生成目标（Target）有不同的 lowering & codegen 路径。

AI 计算图大多跑在 CPU 上，它们一般依托框架上的解释器（e.g. PyTorch）、执行器（e.g. TF）或 虚拟机（e.g. Java/Python）来顺序执行图节点，而这些节点就是一个个的算子，它们背后的计算会下发到各种加速器上。这样计算图执行就会在一个 CPU+加速器的异构系统上流转。

计算图会转换成字节码由虚拟机在CPU上执行，细节后面章节再聊。这章说的 target backend 主要是针对算子的，比如 "tosa.matmul"，它是很高阶的算子，如果用C语言来实现计算的话，需要做三层循环计算：

```
    // C = A x B，A shape: (M, K)，B shape: (K, N)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < K; ++k) {
               C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
```

在 MLIR 这条线需要经过 linalg、SCF、affine、llvm 这些方言的逐级转换才能为 "tosa.matmul" 生成上述 C代码的 IR，然后再交给 LLVM 生成目标机器码。

除了 MLIR 这条线外，也可以通过手写或其他程序生成的方式提供算子，例如，CUDA/OpenCL/Triton，这种模式其实是更主流的。

fleet compiler 目标是能支持 x86/RISC-V/GPU 这三种目标架构以及上述两类算子：

| Target Backend | Kernel | Target |
| --- | --- | --- |
| sycl | SYCL | x86 / RISC-V |
| cuda | CUDA | Nvidia GPU |
| llvm-cpu | MLIR | RISC-V |
| llvm-gpu | MLIR | Nvidia GPU |

# SYCL

SYCL（发音为“sickle”）的编程模型和 CUDA 是比较相似的，是一种用于异构计算的开放标准编程模型，旨在通过单一源码的方式简化多种硬件架构上的并行编程。它是由Khronos Group开发的，基于C++语言，特别是C++17及其后续版本，提供了一种高层次的抽象，使得开发者能够编写可在不同设备（如CPU、GPU、FPGA等）上运行的代码。

本质上讲 SYCL 只是一套异构计算的 API，它还需要一个实现：oneAPI。oneAPI 由英特尔发起，并由UXL Foundation管理，通过对SYCL的实现和扩展，提供了一个统一的多架构编程环境。[Intel oneAPI toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#analytics-kit) 、[oneAPI construction kit](https://developer.codeplay.com/products/oneapi/construction-kit/4.0.0/guides/overview/introduction/architecture) 和 [oneapi-src](https://github.com/oneapi-src) 为开发 SYCL 程序提供一系列编译器、运行时、调试工具和针对 AI 的高性能计算库。

fleet compiler 会用它们开发 x86 / RISC-V 目标架构的算子库。

# CUDA

虽然 SYCL / oneAPI 也支持 GPU 后端，但在 Nvidia GPU 上做开发最优选择当然还是它自家的 CUDA。关于 CUDA 编程这里就不赘述了。

# LLVM

llvm 的 backend 是通过将 tosa dialect 经过 linalg、SCF、affine、llvm 等 dialect 的逐级转换来创建 kernel 的。MLIR 对算子开发是有提供支持的，例如，linalg 就支持对 op tiling，另外，还可以开发 pass 做灵活的算子融合，可以比较轻松地实现对新模型的算子支持。

# --target-backend

```
fleet_compiler_cli %s --emitMLIR --only-compile --opt --target-backend sycl
```

在编译的时候通过 "--target-backend" 指定目标后端会给 module 生成相应的属性，运行时会根据它来注册相应的算子库。

```
# CHECK-NEXT:   %3117 = "vm.call" @device.transpose(%2979,%3047) : (tensor<50257x768xf32>,tensor<2xi32>) -> (tensor<768x50257xf32>)
# CHECK-NEXT:   %2774 = "vm.call" @device.matmul(%2801,%3117) : (tensor<16x768xf32>,tensor<768x50257xf32>) -> (tensor<16x50257xf32>)
# CHECK-NEXT:   "vm.call" @print(%2774) : (tensor<16x50257xf32>) -> ()
# CHECK-NEXT: }) {target_info = {target_backend = sycl}} : () -> ()
```


---
