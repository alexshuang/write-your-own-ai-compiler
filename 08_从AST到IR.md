fleet-compiler 会将语义分析后的 AST 模型翻译成 IR 并将它逐步降低到 RISC-V 汇编。翻译过程步骤概述：
- 遍历 AST：首先，需要遍历 AST，访问每个节点。
- 构建 IR 操作（Operation）：根据 AST 节点的类型和属性，构建相应的操作，设置适当的属性和操作数。
- 插入操作到 IR 模块：将构建的操作插入到 IR 模块中。
- 处理嵌套结构：处理函数、循环等嵌套结构，确保正确的控制流和数据流。

# 遍历 AST

遍历 AST 的运作机制和前面语义分析的 pass 一样，创建继承于 `AstVistior` 的 `ConvertASTtoMLIR` importer 类（fleet_compiler/ir/importer.py），定义一组 visit 方法，对不同类型的节点构建相应的操作。

```
class ConvertASTtoMLIR(AstVisitor):
    ast: AstModule
    module: ModuleOp

    def __init__(self, ast: AstModule) -> None:
        super().__init__()
        self.ast = ast
        self.has_explicit_ret = False

    def convert(self):
        return self.visitModule(self.ast)

    def visitModule(self, node: AstModule):
        self.module = ModuleOp()
        self.visitBlock(node.block, self.module.body.blocks[0])
        return self.module
```

# 构建 IR 操作

首先需要为 AST importer 定义一组 visit 方法来为 AST 节点构建相应的 op，例如 visitModule() 用于构建 `bulitin.ModuleOp`，visitVariableDef() 用于构建 `arith.ConstantOp`，visitFunctionDef() 用于构建 `func.FuncOp`。

我们需要将构建的 op 的输出（OpResult）以 <ast_node_name: Value Object> 的格式记录在符号表中。

例如 python 代码 "a = 20"，它的 AST node 是 `VariableDef`，会调用 `visitVariableDef()` 来构建 IR。init 是 `visitIntegerLiteral()` 创建的 arith.ConstantOp 的输出。`self.declare()` 用于将 Value 写入符号表中。

```
    def visitVariableDef(self, node: VariableDef):
        init = self.visit(node.init)
        self.declare(node.name, init)

    def visitIntegerLiteral(self, node: IntegerLiteral):
        type = IntegerType(32, True)
        attr = IntegerAttr(node.value, type)
        return self.create(arith.ConstantOp(attr, type)).results[0]
```

后面代码需要用到变量 "a" 的时候，通过 `self.get()` 在符号表中查找 key "a" 就能获取到相应的 IR Value。

```
    def visitVariable(self, node: Variable):
        return self.get(node.name)
```

另外，由于 Block 会为 arguments 创建别名（`%argX`），因此，block argument 构建时，需要做双重符号映射：ast_name: "%argX"，"%argX": Value。

```
    def visitBlock(self, ast_block: ASTBlock, block: Block):
        with Builder.at_end(block) as builder:
            for o in block.arguments:
                # ast name mapping to ir name
                self.declare(o.ast_name, o.name)
                # ir name mapping to Value
                self.declare(o.name, o)
```

# 插入操作到 IR 模块

IR 通过 Builder（fleet_compiler/ir/builder.py）来构建 op。InsertionPoint 用于指定新 op 在 Block 中的位置，可以指定插入到 Block Operations list 的开头、结尾或指定某 op 的当前位置。前面提到的符号表 SymbolTable 也是定义在 builder 中的。

```
    def visitModule(self, node: AstModule):
        self.module = ModuleOp()
        self.visitBlock(node.block, self.module.body.blocks[0])
        return self.module

    def visitBlock(self, ast_block: ASTBlock, block: Block):
        with Builder.at_end(block) as builder:
            ...

    def visitIntegerLiteral(self, node: IntegerLiteral):
        type = IntegerType(32, True)
        attr = IntegerAttr(node.value, type)
        return self.create(arith.ConstantOp(attr, type)).results[0]

    def create(self, op: Operation):
        builder = ImplicitBuilder().get()
        builder.insert(op)
        return op
```

with Builder.at_end(block) 是指在这个 scope 下所有通过 `builder.insert()` 插入的 op，都会插入到 block.operaitons 列表的末尾。

# 处理嵌套结构

目前我们只需要解决的函数作用域问题。以下方 sample 为例，它的 AST 有两个 block，不同 op 需要能插入不同 block，value 也需要写到不同的符号表里。

```
def foo():
    b = 10

a = 10
```

```
Module:
  Block:
    Function define foo
      Signature
        []
      Block:
        Variable define b, init: 10
    Variable define a, init: 10
```

因此每当访问到 block 节点，需要创建创建新的 builder 并将它压入 BuilderStack：

```
    def visitBlock(self, ast_block: ASTBlock, block: Block):
        with Builder.at_end(block) as builder:
          ...

class Builder(contextlib.AbstractContextManager):
    def __enter__(self) -> Any:
        ImplicitBuilder().push(self)
        return ImplicitBuilder().get()
```

在 importer 需要往 module 插入新 op 时，通过 ImplicitBuilder().get() 获取使用最新创建 builder：

```
class ConvertASTtoMLIR(AstVisitor):
    def visitIntegerLiteral(self, node: IntegerLiteral):
        type = IntegerType(32, True)
        attr = IntegerAttr(node.value, type)
        return self.create(arith.ConstantOp(attr, type)).results[0]

    def create(self, op: Operation):
        builder = ImplicitBuilder().get()
        builder.insert(op)
        return op
```

这样当开始转换 foo 函数时，importer 会创建新的 builder 来构建 op 直至 function body 里的 AST node 都转换完成，然后该 builder 被释放，上一个 builder 又会成为 current builder。

---
