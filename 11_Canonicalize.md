上一章定义了用于形状推断的 pass 和接口，完成了对 sample IR 的形状推断。这一章我们会引入 Canonicalize（规范化）优化将内联后的 IR 做进一步简化，来解决复杂计算图的形状推断。

规范化是一个重要的编译器优化步骤，其主要作用是通过一系列模式重写规则，使 IR 更加标准化和简洁。这不仅有助于后续的编译器分析和优化，还能提高代码的可读性和维护性。

规范化的主要目标包括：
- 简化：通过消除冗余操作和折叠常量来简化 IR。
- 优化：将操作转换为更高效的形式，从而减少计算复杂度。
- 统一性：确保相似的结构以统一的方式表示，这使得进一步的转换和分析更容易。
- 为后续处理做准备：使 IR 更加适合后续的优化和分析步骤。

Compiler 通过创建 CanonicalizePass（fleet_compiler/ir/transforms/canonicalize.py）实现 IR 规范化。它会遍历所有操作，如果操作 `hasCanonicalizer == True`，通过 `get_canonicalize_patterns()` 获取该操作的规范化匹配模式来重写 IR。

```
class Canonicalize(RewritePattern):
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        if not op.hasCanonicalizer:
            return False

        patterns = op.get_canonicalize_patterns()
        for p in patterns:
            if p.match_and_rewrite(op, rewriter):
                return True
        return False

class CanonicalizePass(Pass):
    def run_on_operation(self, op: ModuleOp) -> None:
        RewritePatternApplier([Canonicalize()]).apply(op)
```

想要为操作指定规范化重写模式，需要定义 `hasCanonicalizer` 变量，实现 get_canonicalize_patterns 方法。匹配模式一般定义在 fleet_compiler/ir/transforms/canonicalize_patterns 目录。

```
class CastOp(Operation):
    hasCanonicalizer = True

    def __init__(self, input: Value, result_type: IRType):
        super().__init__(operands=[input], result_types=[result_type])

    def get_canonicalize_patterns(self):
        from ..transforms.canonicalize_patterns.tosa import (
            RemoveRedundantCast
        )
        return [RemoveRedundantCast()]
```

# 需要解决的问题

想要对 gpt2（gpt2.py） 的 IR 做形状推断，除了要为相关 ops 定义 infer_shapes() 外，还需要创建规范化 pass 来删除一些多余的 cast op 来简化计算图。

## 删除用于转换函数参数的 cast

```
  func.func @foo(%arg0: tensor<*xf32>) -> (tensor<*xf32>) {
    %0 = "numpy.sqrt" (%arg0) : (tensor<*xf32>) -> tensor<*xf32>
    return %0: tensor<*xf32>
  }
  %2 = "arith.constant" () {value = 20.0: f32} : () -> f32
  %3 = func.call @foo(%2) : (f32) -> (tensor<*xf32>)
```

由于函数定义时输入参数的类型未知，因此会统一将函数的输入定义为 `tensor<*xf32>` 类型，但 func.call 的输入类型是明确的，因此在内联时需要插入 cast op 将它转成 unrank 类型：
```
%16 = "tosa.add" (%15, ...) : (...) -> tensor<4xf32>
%42 = "tosa.cast" (%16) : (tensor<4xf32>) -> tensor<*xf32>
%43 = "tosa.sub" (%42, ...) : (tensor<*xf32>, ...) -> tensor<*xf32>
```
这些 cast 并非用于数据类型转换，因此需要删除。另外，如果存在嵌套内联的时候，就会存在多个 cast，尤其是 `tensor<*xf32> -> tensor<*xf32>`，它们也需要删除。

```
class RemoveRedundantCast(RewritePattern):
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        '''
        case 1:
        %42 = "tosa.cast" (%16) : (tensor<*xf32>) -> tensor<*xf32>
        '''
        if op.operands[0].type == op.results[0].type and \
            isinstance(op.operands[0].type, UnrankedTensorType):
                rewriter.replace_all_uses_with(op.results[0], op.operands[0])
                rewriter.erase_op(op)
                return True

        '''
        case 2:
        %16 = "tosa.add" (%15, ...) : (...) -> tensor<4xf32>
        %42 = "tosa.cast" (%16) : (tensor<4xf32>) -> tensor<*xf32>
        %43 = "tosa.sub" (%42, ...) : (tensor<*xf32>, ...) -> tensor<*xf32>
        '''
        if isinstance(op.results[0].type, UnrankedTensorType) and \
            not isinstance(op.operands[0].type, UnrankedTensorType):
                rewriter.replace_all_uses_with(op.results[0], op.operands[0])
                rewriter.erase_op(op)
                return True
        
        return False
```

RemoveRedundantCast 用于处理上述 case。

## 删除多余的 cast

`tosa.gather` 需要输入一个 int类型的张量（indices），同样是由于函数参数的动态性，为了确保操作的输入的正确性，插入了一个 cast 来将任意输入类型转换成 int类型，而如果实际属于就是 int类型，那么那些转换操作就没必要了：

```
class RemoveCastedIndiceOperandForGather(RewritePattern):
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        '''
        %27 = "tosa.cast" (%13) : (tensor<16xi32>) -> tensor<*xf32>
        %22 = "tosa.cast" (%27) : (tensor<*xf32>) -> tensor<*xi32>
        %23 = "tosa.gather" (%20,%22) : (tensor<50257x768xf32>,tensor<*xi32>) -> tensor<*xf32>
        '''
        if isinstance(op.operands[1].type, UnrankedTensorType):
            indice = op.operands[1]
            if isinstance(cast1 := indice.owner(), tosa.CastOp):
                if isinstance(cast1.operands[0].type, UnrankedTensorType):
                    if isinstance(cast2 := cast1.operands[0].owner(), tosa.CastOp):
                        if isinstance(cast2.operands[0].type, RankedTensorType) and \
                            isinstance(cast2.operands[0].type.element_type, IntegerType):
                                rewriter.replace_all_uses_with(cast1.results[0], cast2.operands[0])
                                rewriter.erase_op(cast1)
                                rewriter.erase_op(cast2)
                                return True
        return False
```

---
