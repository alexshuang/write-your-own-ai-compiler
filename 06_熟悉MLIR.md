fleet-compiler 会将语义分析后的 AST 模型翻译成 IR 并将它逐步降低到 RISC-V 汇编。fleet-compiler IR 的创作灵感来自于 MLIR，它类似于 MLIR 的架构，定义有多级 dialects，各 dialect 和 op 的语义和 MLIR 是一致的，便于在图优化的任意阶段将编译操作下发给 MLIR/LLVM。因此，在开始着手翻译 AST 之前，先让我们熟悉一下 MLIR。 

# MLIR

MLIR，全称为**多级中间表示**（Multi-Level Intermediate Representation），是一个由Google在2019年开源的编译器框架，现在由LLVM项目维护 。它的主要目标是通过提供一个可重用、可扩展的编译器架构，解决软件碎片化问题，改善异构硬件的编译效率，并降低构建特定领域编译器的成本。

本章着眼于几个关键点，为的是让你更容易理解 fleet-compiler IR，而非 MLIR 完整教程。官方教程见：https://mlir.llvm.org/docs/Tutorials/Toy/

# Operation 的组成

在 MLIR 中，Operation（op）是抽象和计算的核心单元，类似于 LLVM 指令。MLIR 之所以这些年这么火，其中一个重要特性就是它的可扩展性，op 有很多组成，包括 operands, type, attribute, trait等，这些组件的数量、内容都是用户可以灵活定义的。这样就意味着说，相比其他想 XLA/TVM 等编译器，你要创建自定义的一组 op 是很方便的。

```
%t_tensor = "toy.transpose"(%tensor) {inplace = true} : (tensor<2x3xf64>) -> tensor<3x2xf64>
```
这里是一个 Operation 的示例，我们来看下它的组件：

- "%t_tensor"：它是 op 的 output，一个 SSA `Value` 对象。如果 op 是生产者的话，那么 Value 就是他的产出。
- "toy.transpose"：op name。
- (%tensor)：%tensor 是 operand，它可能是其他 op 的 output（Value），也可能是当前 `Block` 的输入（`BlockArgument`）。
- {inplace = true}：attributes，它是一个字典，内容是 <'inplace': BoolAttr>。和 attributes 类似的属性还有 preperties。
- (tensor<2x3xf64>) -> tensor<3x2xf64>：functional type，'->' 左侧是 op operand types，右侧是 output types。

另外，还有 `Region` 和 `Block`。每个 Operation 可以有多个 Region，每个 Region 可能有多个 Block，Block 里又可能有多个 Operation。例如这里的 "func.func" op 它就有一个 region 和 block，"{ }" 内的 op 就是 block 里的内容。

```
func.func @gelu(%arg0: tensor<*xf32>) -> (tensor<*xf32>) {
  %0 = "arith.constant" () {value = 3.141592653589793: f32} : () -> f32
  %1 = "arith.constant" () {value = 0.5: f32} : () -> f32
  ...
}
```

当只有一个 block，一般 block name 和 arguments 不会显示打印出来，除非有多 block：

```
{
  ^bb1(%1: i32):  // pred: ^bb0
    "some_dialect.some_op"() : () -> ()
    "some_dialect.some_op"() : () -> ()
  ^bb2(%2: i64):  // pred: ^bb0
    "some_dialect.some_op"() : () -> ()
    "some_dialect.some_op"() : () -> ()
}
```

# 方言，多层次 IR

一个方言（dialect）定义一组 op，MLIR 提供了[几十个方言](https://mlir.llvm.org/docs/Dialects/)以及一套完善的方言间转换机制，提倡通过逐步 lowering 的方式，从高阶方言（TOSA/Linalg/...）转换到底层方言（LLVM IR），再交给 LLVM 做 codegen。

例如，Tensorflow keras 的 Add layer，它转换成 MLIR 得到 `"tosa.add"` op，它的 lowering 路径可以是：tosa.add -> linalg.add -> scf.loop -> cf -> llvm ir，操作数从 tensor -> memref -> llvm type。

# 通用的优化和转换框架

MLIR 为开发者提供了一个通用的优化和转换框架，使其可以复用现成的功能模块。例如 canonicalizer（IR 规范化），cse（消除公共子表达式），shape inference 等优化和分析 pass，用户只需为 op 实现相应的回调函数和接口。另外，MLIR 还提供一套方言间转换机制（部分和完全转换），以及一些现成的转换 pass 如 convert-ub-to-llvm, convert-vector-to-llvm 等，减轻开发者工作量。

---
