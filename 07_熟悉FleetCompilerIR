IR（Intermediate Representation）是一种介于高级语言和底层机器代码之间的抽象表示形式，编译器能够通过它对程序进行各种优化，并最终将它转换为特定平台的机器代码。

fleet-compiler IR 的创作灵感来自于 MLIR, MLIR 的多级 IR（dialect）架构，dialect 和 op 的语义也尽量和 MLIR 保持一致，此外，它还可以导出成 MLIR，便于用户在编译的任一阶段切换到 MLIR/LLVM 做后续编译，例如：

```
# CHECK-NEXT:   func.func @gpt2(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> (tensor<*xf32>) {
# CHECK-NEXT:     %202 = "arith.constant" () {value = 50257: i32} : () -> i32
# CHECK-NEXT:     %203 = "arith.constant" () {value = 1024: i32} : () -> i32
# CHECK-NEXT:     %204 = "arith.constant" () {value = 768: i32} : () -> i32
# CHECK-NEXT:     %205 = "numpy.random.randn" (%202,%204) : (i32,i32) -> tensor<50257x768xf32>
```
这是 IR printer 的输出（tests/mlir/test_gpt2.py），可以通过 `mlir-opt` 解析成 MLIR module。

# Dialect

目录 fleet_compiler/ir/dialects 定义有 arith、tosa、tensor、func、builtin 等 dialect，它们的 op、语义和接口等都尽可能和 MLIR 对齐。除了这些标准 MLIR dialect，还定义有 python、numpy dialect 来表示 python/numpy 算子。

# Operation

```
@dataclass
class Operation(IRNode):
    name = ""
    parent: Block | None = None
    operands: Sequence[OpResult | BlockArgument] = field(default_factory=list)
    results: Sequence[OpResult] = field(default_factory=list)
    successors: Sequence[Block] = field(default_factory=list)
    attributes: dict[str, Attribute] = field(default_factory=dict)
    properties: dict[str, Attribute] = field(default_factory=dict)
    regions: Sequence[Region] = field(default_factory=list)
    traits: Sequence[OpTrait] = field(default_factory=list)
```

`Operation` （fleet_compiler/ir/core.py）是所有 dialect op 的基类。它的组成和 MLIR Operation 一致，这里重点聊一下其中几个重要组成：
- name：op name，默认情况下你不需要为，它会根据 dialect 文件名和 op 类名来赋值，例如，TOSA AddOp（fleet_compiler/ir/dialects/tosa.py），name 会被赋值 "tosa.add"。 如果 dialect 分层，比如 "numpy.random.randn"，类名需要通过"_"隔开：Random_RandnOp。
- operands: op 的输入，`OpResult` 或 `BlockArgument` 对象列表。
- results: op 的输出，`OpResult` 对象列表。
- attributes：`Attribute` 对象字典，是编译时的静态信息。
- regions: 和 MLIR 一样，op 里可以有多个 Region。
- traits：和 MLIR 一样，表示 IR 特性，一些 pass 会根据 IR 特性来采取相应的优化操作。

# Region

```
@dataclass
class Region(IRNode):
    blocks: Sequence[Block] = field(default_factory=list)
    parent: Operation | None = None
```

Region是一个包含多个 Block 的容器。每个 Region 可以嵌套在 Operation 内部，形成层次化的结构。

# Block

```
@dataclass
class Block(IRNode):
    name: str = ""
    arguments: Sequence[BlockArgument] = field(default_factory=list)
    operations: Sequence[Operation] = field(default_factory=list)
    parent: Region | None = None
```

Block 是一个有序的 Operation 列表。每个Block可以包含多个 Operation，并且可以通过控制流指令（如条件分支、循环等）与其他 Block 连接。Block 通常嵌套在 Region 中，而 Region 又嵌套在 Operation 中。

# IRType

Type用于描述值的性质和结构。IR 里的每个值都有一个对应的 Type，Type 定义了该值的数据类型和形状。MLIR的类型系统是高度可扩展的，允许用户定义自定义类型，并规定其语义。

builtin dialect（fleet_compiler/ir/dialects/builtin.py）定义了常见的类型：
- 标量：如整数（IntegerType）、浮点数（FloatType）、字符串（StringType）等。
- 数组、矩阵：如 ArrayType、RankedTensorType、UnrankedTensorType。

每个 dialect 都可以创建新的 type 类型：

```
@dataclass
class ArrayType(IRType):
    dims: list
    element_type: IntegerType | FloatType
```

# Attribute

Attribute 是一种编译期静态信息，用于描述操作的特性和元数据。与操作数（Operand）不同，Attribute 在编译期是已知的，而操作数通常只有在运行时才能确定。Attribute 可以是整数、浮点数、字符串等常量值，也可以是更复杂的数据结构。

builtin dialect 定义了常见的标量属性，如 IntegerAttr、FloatAttr、StringAttr 等；矩阵和向量属性，如 DenseIntOrFPElementsAttr、ArrayAttr。

每个 dialect 都可以创建新的 attribute 类型，attribute 的组成包括有具体的值和 type：

```
@dataclass
class ArrayAttr(Attribute):
    value: list
    type: ArrayType
```

IR 中的 attribute 通过键值对词典的形式保存在 Operation，比如 { "value": ArrayAttr }:

```
%229 = "arith.constant" () {value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]>: tensor<16xi32>} : () -> tensor<16xi32>
```

# Value / OpResult / BlockArgument

Operation 的输入（operands）它不是 OpResult 就是 BlockArgument 对象。OpResult 是 Operation 的输出，BlockArgument 是 Block 的参数。

OpResult 和 BlockArgument 都是 Value 的子类，它们的区别更多是体现在名字上：OpResult name = "%" + 当前 IR 数，如 "%123"；BlockArgument name = "%arg" + index，如 "%arg0"。

```
@dataclass
class Value(ABC):
    _name: str
    type: IRType
    uses: Sequence[Use]
    _name_regex = re.compile(r"(%[A-Za-z_0-9]*)")
```

# Builder

和 MLIR 一样，IR 通过 Builder（fleet_compiler/ir/builder.py）来构建 op。InsertionPoint 用于指定新 op 在 Block 中的位置，可以指定插入到 Block Operations list 的开头、结尾或指定某 op 的当前位置。每个 builder 还会定义一个符号表 SymbolTable 用于记录 op name 和 Value 对象的映射关系。

```
class Builder(contextlib.AbstractContextManager):
    def __init__(self, insertion_point: InsertionPoint) -> None:
        super().__init__()
        self.insertion_point = insertion_point
        self.symbol_table = SymbolTable()
```

# Printer

通过调用 ModuleOp 的 dump() 可以将 IR 以 MLIR 的格式打印出来。

```
class ModuleOp(Operation):
    sym_name = StringAttr("main")
    body: Region

    def dump(self):
        from ..printer import Printer
        Printer().print(self)
```

Printer（fleet_compiler/ir/printer.py） 为 Operation 提供了标准的输出格式，但如果你想要为 op 创建自定义的格式，需要设置 op 属性 “hasCustomAssemblyFormat = True”，并实现相应的 print()。

```
    def print(self, op: Operation):
        if isinstance(op, Operation):
            if op.hasCustomAssemblyFormat:
                op.print(self)
            else:
                self._print_results(op)
                self._print_op_with_default_format(op)
```

例如，"func.return" op：

```
class ReturnOp(Operation):
    hasCustomAssemblyFormat = True

    def __init__(self, operands: list[Value]):
        super().__init__(operands=operands)

    def print(self, p: Printer):
        prefix = p._get_indent()
        if len(self.operands) > 0:
            rets = ','.join([o.name for o in self.operands])
            ret_types = ','.join([p._get_type_str(o.type) for o in self.operands])
            ret = f' {rets}: {ret_types}'
        else:
            ret = ''
        p._print_string(f"{prefix}return{ret}\n")
```

---

