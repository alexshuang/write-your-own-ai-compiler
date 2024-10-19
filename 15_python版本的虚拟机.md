上一章完成了从IR到字节码的转换，得到一个python backend 的字节码模型：ByteCodeModule，这章介绍如何通过虚拟机把模型跑起来。

```
bc = ByteCodeConverter(module).convert()
VM(bc).run()
```

![](https://github.com/alexshuang/write-your-own-ai-compiler/blob/main/images/bc_vm_layout.png)

ByteCodeModule 代表我们的程序，有字节码、常量池，还记录了程序需要创建的变量数以及运行的目标平台（target_backend）。target_backend == "python" ，表示跑在 python 版虚拟机 -- VM（fleet_compiler/vm/vm.py）上。

# VM

VM 是栈机，它的重要数据结构是操作数栈，用来存放计算需要的输入和输出（[设计虚拟机](https://github.com/alexshuang/write-your-own-ai-compiler/blob/main/13_%E8%AE%BE%E8%AE%A1%E8%99%9A%E6%8B%9F%E6%9C%BA.md)），以及一个存放本地变量的列表，用于保存计算结果以供后续计算使用。

```
class StackFrame:
    def __init__(self, variable_size) -> None:
        self.operand_stack = []
        self.local_variable = [None] * variable_size

class VM:
    def __init__(self, bc: ByteCodeModule) -> None:
        self.bc = bc
        self.stackframe = StackFrame(self.bc.variable_size)
        self.current_stackframe = self.stackframe # should only main stackframe
    
    def push_operand(self, val: Any):
        self.current_stackframe.operand_stack.append(val)

    def pop_operand(self):
        return self.current_stackframe.operand_stack.pop()
```

# Run

```
    def run(self):
        idx = 0
        code_size = len(self.bc.code)
        while idx < code_size:
            if self.bc.code[idx] == OpCode.fconst_0:
                self.push_operand(0.)
                idx += 1
            elif self.bc.code[idx] == OpCode.fconst_1:
                self.push_operand(1.)
                idx += 1
            elif self.bc.code[idx] == OpCode.fconst_2:
                self.push_operand(2.)
                idx += 1
```

run() 是虚拟机的执行引擎，你一眼就可以看到它就是个大循环，每个迭代取出 index 下标的字节码，根据指令做相应动作。

虚拟机的执行顺序大致如下：
1. 将函数调用需要的操作数，从常量池（consts）或本地变量中读取 index 下标的内容，压入操作数栈
2. 函数调用：解析函数名，获取目标函数，从栈中取出目标操作数，计算，将结果压入操作数栈
3. 从操作数栈弹出计算结果，并写入本地变量列表
4. 字节码遍历完成，或回到步骤1，调用下一个 op

我们先看函数调用：

```
            elif self.bc.code[idx] == OpCode.invokestatic:
                hint = self.bc.consts[self.bc.code[idx+1]]
                disp_info = DispatchHintParser(hint).parse()
                disp_info.target = self.bc.target
                self.invoke(disp_info)
                idx += 2
```

函数调用码 OpCode.invokestatic 后面跟着一个长字符串函数名，例如， "device_add_sig_ii_i_ins_i32_i32_outs_i32"，需要用 DispatchHintParser() 来解析这个函数名，根据得到的 DispatchInfo 来选择合适算子执行。

```
@dataclass
class DispatchInfo:
    type: str
    target: str
    name: str
    sig: list[str]
    in_types: list[str]
    out_types: list[str]
    attrs: dict[str, list|Any]
```

invoke() 会根据 disp_info 来调用相应 python/numpy API 来做相应计算，计算结果会压入操作数栈，后面的 OpCode.Xstore 机器码再把它们从栈中弹出，写入指定 index 位置的本地变量列表：

```
            elif self.bc.code[idx] == OpCode.astore:
                self.current_stackframe.local_variable[self.bc.code[idx+1]] = self.pop_operand()
                idx += 2
```

操作数入栈代码也是，再将data入栈之前，需要先解析数据类型、shape、元素数据类型等信息，然后对tensor做reshape、填充处理。

```
            elif self.bc.code[idx] == OpCode.ldc:
                tcode = self.bc.code[idx+1]
                if tcode == TypeCode.scalar:
                    val = self.bc.consts[self.bc.code[idx+2]]
                    self.push_operand(val)
                    idx += 3
                elif tcode == TypeCode.tensor:
                    num_ranks = self.bc.code[idx+2]
                    shape = [self.bc.code[i] for i in range(idx+3, idx+3+num_ranks)]
                    # element type: idx+3+num_ranks, unused here
                    val = self.bc.consts[self.bc.code[idx+3+num_ranks+1]]
                    if isinstance(val, Iterable):
                        val = np.reshape(val, shape)
                    else:
                        val = np.full(shape, val)  # fill scalar to ndarray
                    self.push_operand(val)
                    idx += 3+num_ranks+2
                else:
                    raise ValueError(f"Unsupported data type: {tcode}")
 
```

# Operators

```
def python_get_dispatch_function(info: DispatchInfo):
    dispatch_functions = {
        "add": np.add,
        "sub": np.subtract,
        "mul": np.multiply,
        "div": np.divide,
        "gather": lambda array, idx: array[idx].squeeze(),
        "transpose": np.transpose,
        "matmul": np.matmul,
        "mean": np.mean,
        "var": np.var,
        "sqrt": np.sqrt,
        "pow": np.power,
        "tanh": np.tanh,
        "max": np.max,
        "exp": np.exp,
        "slice": slice_func,
        "concat": concat_func,
        "reshape": reshape_func,
        "reduce_sum": np.sum,
        "reduce_max": np.max,
        "cast": lambda x: x,
        "splat": splat_func(info),
        "print": print,
    }
    return dispatch_functions[info.name]
```

对于 gpt2 example 来说，python虚拟机需要提供上述算子，它们大部分都是通过 numpy 实现的。后续要支持 SYCL、CUDA 这些 backend 也只需要提供这些算子即可。

# END

至此，我们已经完成了 python backend 编译器和运行时开发，为 python/numpy 程序提供端到端编译支持：Source code -> AST -> IR -> MLIR -> bytecode -> run by vm。先撒个花庆祝下吧🎉。

我们试着跑一下 gpt2 example，可以看到它的计算结果和用 python 跑出来的是对齐的，只是整个过程用时超过了1分钟，可能有小伙伴在尝试时等不及，以为程序挂了，其实这是因为超过 95%的时间都是花在程序编译了，而 python backend 的编译和运行时在同一程序上的。

后续开发 SYCL backend 的时候，会开发一个 C 版本的虚拟机程序，将编译器和运行时分开。

```
root@300cb41040c1:/work/new/fleet-compiler$ time fleet_compiler_cli examples/gpt2.py
[[-4.21660253e-04  7.29883815e-04 -4.33100290e-05 ...  5.03254805e-04
  -3.66641127e-04  2.89550343e-04]
 [-4.19659544e-04  7.24772203e-04 -4.58018037e-05 ...  5.08032994e-04
  -3.70418946e-04  2.88506467e-04]
 [-4.21276194e-04  7.27972861e-04 -4.24837327e-05 ...  5.01558398e-04
  -3.71175064e-04  2.93489106e-04]
 ...
 [-4.18536527e-04  7.23852492e-04 -4.42954715e-05 ...  5.02633920e-04
  -3.65614700e-04  2.87673096e-04]
 [-4.17000882e-04  7.26545760e-04 -4.21995092e-05 ...  5.04763755e-04
  -3.64839513e-04  2.91270895e-04]
 [-4.18463483e-04  7.26882803e-04 -4.55622031e-05 ...  5.04293242e-04
  -3.67897237e-04  2.88245682e-04]]

real    1m17.047s
user    1m24.956s
sys     0m6.210s
```

---
