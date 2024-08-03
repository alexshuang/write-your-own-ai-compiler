由于 Python 是动态语言，函数声明的参数列表并没有形状约束，因此，"FuncOp" 在创建之处，它的输入和返回值都是 "UnrankedTensorType" -- `tensor<*xf32>`，在 IR lowering 的初期需要对它们做形状推断，基于其输入形状和操作的属性来推导操作的输出形状。

```
# tests/mlir/test_shape_inference.py
"builtin.module" () ({
  func.func @foo(%arg0: tensor<*xf32>) -> (tensor<*xf32>) {
    %0 = "numpy.sqrt" (%arg0) : (tensor<*xf32>) -> tensor<*xf32>
    return %0: tensor<*xf32>
  }
  %2 = "arith.constant" () {value = 20.0: f32} : () -> f32
  %3 = func.call @foo(%2) : (f32) -> (tensor<*xf32>)
  "python.print" (%3) : (tensor<*xf32>) -> ()
}) : () -> ()
```

# Inline function

函数形状推断完成后，应该是要生成新的带形状的 FuncOp（特化函数），考虑到输入动态性可能会同一函数产生多个特化函数。考虑到特化函数的特性以及模型计算图的特点，这里采用内联函数的方式来展开函数操作，去掉控制流，减轻后续 lowering 难度。

我们通过 InlineFunctionPass（fleet_compiler/ir/transforms/inline.py）来内联目标函数：
- 重写模式 InlineFunction 会遍历所有 CallOp，通过符号属性 "callee" 找到 FuncOp。
- 通过 clone FuncOp 来创建新的操作（func_op.operations），并将这些操作插入 CallOp 所在位置。
- 最后用新的 value 替换掉 CallOp 的输入、输出，并删除 CallOp 和原来的 FuncOp。

```
class InlineFunction(RewritePattern):
    func_op_table: dict[str, func.FuncOp] | None = None

    def _get_func_op_by_name(self, module: ModuleOp, name: str):
        if self.func_op_table is None:
            self.func_op_table = {o.attributes['sym_name'].value:o
                                  for o in module.operations if isinstance(o, func.FuncOp)}
        return self.func_op_table[name]

    @op_rewrite_pattern
    def match_and_rewrite(self, op: func.CallOp, rewriter: PatternRewriter):
        assert isinstance((module := op.parent_node.parent_node.parent_node), ModuleOp), \
            "Nested functions are not supported"
        assert isinstance(block := op.parent_node, Block)

        func_name = op.attributes['callee'].sym_name.value
        func_op = self._get_func_op_by_name(module, func_name)
        new_func_op = func_op.clone()
        inlined_block = new_func_op.regions[0].blocks[0]

        # cast ranked tensors to unranked tensors
        cast_ops = [tensor.CastOp(operand, type)
                    for operand, type in zip(op.operands,
                                             new_func_op.attributes['function_type'].input_types)]
        for o in cast_ops:
            block.insert_before(o, op)
        
        # replace operands
        args = inlined_block.arguments
        new_operands = [o.results[0] for o in cast_ops]
        for old, new in zip(args, new_operands):
            rewriter.replace_all_uses_with(old, new)
        
        # inline block before call op
        new_ops = [o for o in inlined_block.operations if not isinstance(o, func.ReturnOp)]
        for o in new_ops:
            o.parent = None
            block.insert_before(o, op)

        last_op = new_ops[-1]

        # replace results
        # MUST: return value are the previous output
        for old, new in zip(op.results, last_op.results):
            rewriter.replace_all_uses_with(old, new)
            for use in old.uses:
                new.uses.append(Use(use.operation, use.index))

        # erase call op & func decl op
        rewriter.erase_op(op)
        rewriter.erase_op(func_op)
```

# CastOp

内联后 function body 的 op 要求 UnrankedTensorType 输入（`"numpy.sqrt" (%arg0) : (tensor<*xf32>) -> tensor<*xf32>`），和实际输入的不符，因此需要添加一个 CastOp 来将 `%2 : f32` 转换成 `tensor<*xf32>`：

```
        # cast ranked tensors to unranked tensors
        cast_ops = [tensor.CastOp(operand, type)
                    for operand, type in zip(op.operands,
                                             new_func_op.attributes['function_type'].input_types)]
        for o in cast_ops:
            block.insert_before(o, op)
```

# After inline

```
"builtin.module" () ({
  %2 = "arith.constant" () {value = 20.0: f32} : () -> f32
  %6 = "tensor.cast" (%2) : (f32) -> tensor<*xf32>
  %7 = "math.sqrt" (%6) : (tensor<*xf32>) -> tensor<*xf32>
  "python.print" (%7) : (tensor<*xf32>) -> ()
}) : () -> ()
```

# OpInterface

MLIR 允许开发者为操作定义通用的接口（OpInterface），从而实现操作行为的模块化和可重用性。同样的，我们 IR 也采用相同的机制。

我们创建了一个用于形状推断的操作接口 ShapeInferenceOpInterface，有形状推断需求的操作只要实现该接口的 `infer_shapes` 方法，ShapeInferencePass 在遍历操作的时候，如果发现操作定义有 infer_shapes 方法，就会调用它来推断操作的输出形状。

```
class ShapeInferenceOpInterface(OpInterface):
    def infer_shapes(self, op: Operation):
        raise NotImplementedError

class CastOp(Operation, ShapeInferenceOpInterface):
    def __init__(self, input: Value, output_type: RankedTensorType | UnrankedTensorType):
        super().__init__(operands=[input], result_types=[output_type])

    def infer_shapes(self, op: Operation):
        if isinstance(in_type := op.operands[0].type, UnrankedTensorType):
            raise ValueError(f"Invalid operand type: {in_type}")
        op.results[0].type = in_type
```

# Final

```
"builtin.module" () ({
  %2 = "arith.constant" () {value = 20.0: f32} : () -> f32
  %6 = "tensor.cast" (%2) : (f32) -> f32
  %7 = "math.sqrt" (%6) : (f32) -> f32
  "python.print" (%7) : (f32) -> ()
}) : () -> ()
```

---
