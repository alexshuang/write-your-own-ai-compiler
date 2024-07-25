AST 运行时（AST Runtime）是指在程序执行期间，解释器或虚拟机使用抽象语法树（AST）来解释和执行代码的过程。fleet-compiler AST 解释器 [Interpreter](https://github.com/alexshuang/fleet-compiler/blob/main/fleet_compiler/frontend/runtime.py#L40) 通过遍历和处理 AST 节点来执行相应的操作。

```
class Interpreter(AstVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.call_stack = []
        self.ops = Operation()
    
    def visitModule(self, node: AstModule):
        self.enter()
        return super().visitModule(node)
```

Interpreter 的运作机制和引用消解（语义分析）的 pass 一样，这里不赘述了。除了要为 Interpreter 定义一套 visit 方法外，还要解决几个问题：
- 创建栈帧
- 处理函数输入参数和返回值
- 支持 python/numpy 算子

# 栈帧

栈帧（Stack Frame）是计算机程序在执行过程中用于管理函数调用和局部变量的一种数据结构。它在函数调用时创建，在函数返回时销毁。

```
class StackFrame(contextlib.AbstractContextManager):
    def __init__(self) -> None:
        self.variables = {} # local variables
        self.ret_val = None # return value

    def update(self, name: str, value):
        self.variables[name] = value

    def get(self, name: str):
        return self.variables[name] if name in self.variables else None

    def __enter__(self) -> Any:
        CallStack().push(self)
        return CallStack().get()

    def __exit__(self, __exc_type: type[BaseException] | None,
                 __exc_value: BaseException | None,
                 __traceback: contextlib.TracebackType | None) -> bool | None:
        return CallStack().pop()
```

函数调用时需要为被调用函数创建新栈帧，并将函数输入以 <变量名: 值> 键值对形式通过 `update()` 入栈，在函数返回前，将返回值写回到上一个栈帧。

创建新栈帧：
```
        with StackFrame():
            return super().visitModule(node)
```

处理函数的 args 和 kwargs 输入，并将它们入栈：
```
                    # args
                    func = node.sym.node
                    params = func.signature.param_list.params
                    args = node.arg_list.args
                    num_args = len(args)
                    num_params = len(params)
                    if num_params < num_args:
                        raise TypeError(f"Interpreter: Expect {num_params} arguments but got {num_args}")

                    names, values = [], []
                    for i, arg in enumerate(args):
                        names.append(params[i].name if isinstance(arg, PositionalArgument) else arg.name)
                        values.append(self.visit(arg.value))
                    # args with default-value
                    for p in params[num_args:]:
                        names.append(p.name)
                        values.append(self.visit(p.init))

                    # set variable values
                    for k, v in zip(names, values):
                        self.update_variable_value(k, v)
```

写回返回值：
```
    # set return value
    def visitBlock(self, node: Block):
        def should_run(node):
            # Instructions that should be executed
            return not isinstance(node, FunctionDef)

        ret = None
        for o in node.stmts:
            if should_run(o):
                ret = self.visit(o)
                if isinstance(ret, RetVal):
                    if ret.value is not None: # not return by BlockEnd
                        self.set_return_value(ret.value)
                    return ret

    def set_return_value(self, value):
        self.call_stack.get(-2).ret_val = value
```

# 算子库

fleet_compiler/frontend/operators 目录是运行时的算子库，支持 python builtin 和 numpy 算子，例如，`print`、`np.transpose` 等。

```
fleet_compiler/frontend/
|-- operators
|   |-- __init__.py
|   |-- numpy
|   |   |-- __init__.py
|   |   |-- ops.py
|   |   `-- random
|   |       |-- __init__.py
|   |       `-- ops.py
|   `-- python
|       |-- __init__.py
|       |-- ops.py
|       `-- time
|           |-- __init__.py
|           `-- ops.py
```

如果需要增加新算子的支持，只需要找到算子名对应的目录的 ops.py 定义同名函数即可。例如，要支持 `np.random.randn()`，只要在 `fleet_compiler/frontend/operators/numpy/random/ops.py` 定义 randn 函数即可。

```
# fleet_compiler/frontend/operators/numpy/random/ops.py
import numpy.random as random

def randn(args, kwargs):
    return random.randn(*args, **kwargs)
```

---
