# 前言：从0到1写自己的AI编译器
你好，我是A君，一名AI从业者，最近几年，我一直在研究深度学习框架和编译器这些底层技术。。

去年底的一次偶然经历让我意识到，市面上的AI编译器都有各自的局限性，无法满足我的需求。既然我对框架和编译器都有所了解，为什么不自己动手写一个呢？加上那段时间刚好有空，我开始了这个项目，[fleet-compiler](https://github.com/alexshuang/fleet-compiler) 就这样诞生了。**它的目标是基于MLIR，将numpy模型编译成可以在X86/RISC-V上运行的程序**。

开发过程中，我萌生了写专栏来记录这个过程的想法。我在大学时用几个月的时间跟着一本书，《自己动手写操作系统》，手撸了一个简单的操作系统。虽然当时看来那些技术离Linux这样的真实应用还很远，但它的确带我入了门，更重要的是，整个过程真的很爽。

这也是这个专栏诞生的原因，希望也能给你带来相同的体验。

# 为什么是 Numpy?

我写 fleet-compiler 的初衷是因为遇到一个问题：我用numpy写了一个模型，我想把它生成一个计算图（比如 MLIR/ONNX），但发现市面上并没有相关的工具可用，最后只能用tensorflow来重写该模型。从PyTorch的迅速崛起、Google改革TensorFlow2以及JAX的定位可以看出，python/numpy的易用性、动态性和灵活性对框架的用户来说至关重要。

fleet-compiler可以支持对 python/numpy 前端语言做端到端的编译。当然，你也可以只用它做python/numpy到MLIR的转换，接着用 MLIR/IREE 对生成的MLIR 来做图优化和Codegen。

# MLIR based

fleet-compiler 会将 python 转换成 AST，再翻译成 IR。fleet-compiler IR 从 MLIR/xDSL 那获得很多灵感，它类似于 MLIR 的架构，各 dialect 和 op 的语义和 MLIR 是一致的。fleet-compiler IR 可以导出成 MLIR，便于用户在图优化的任意阶段将后续的编译传递给 MLIR/LLVM。

# 要完成哪些挑战？

既然是自己动手，那么编译器的全栈模块以纯手写实现为主，总的来说，需要实现编译器和运行时，并适配目标模型（GPT2）。

### 首先，需要实现一个 python 的编译器前端

编译器前端经过词法分析、语法分析和语义分析，将一个用 numpy 实现的 [GPT2 model](https://github.com/alexshuang/fleet-compiler/blob/main/examples/gpt2.py)（出自 [GPT in 60 Lines of NumPy](https://github.com/jaymody/picoGPT/blob/29e78cc52b58ed2c1c483ffea2eb46ff6bdec785/gpt2_pico.py#L3-L58)）转换成 AST module。

### 第二，需要将 AST 翻译成 MLIR IR

编译器会定义自己的 IR，这套 IR 采用和 MLIR 一样的 dialect 和 ops。翻译得到的 IR 需要能导出成 MLIR，并被 mlir-opt 解析。AST module 会翻译成 TOSA + numpy + arith + math + tensor dialects 混合的 IR。

### 第三，需要实现 lowering passes 来 lowering IR and codegen

编译器会定义一套和 MLIR 一样的 rewrite、interface、pass 机制，会提供一系列的 pass 来将高阶 IR lowering 成 LLVM IR，并通过 LLVM 做 codegen。

### 最后，需要实现运行时

编译器会提供多个运行时以运行 AST、MLIR 和 RISC-V ISA。AST 的运行时是基于python的，模型中的算子是用 python/numpy 实现的。最终的executable 可能是 vmbytecode + RISC-V ISA format，通过手写虚拟机来执行。

# Get Start

话不多说，正文再见，我已经准备好开启AI编译器的冒险了~

Fleet-compiler: https://github.com/alexshuang/fleet-compiler

---


