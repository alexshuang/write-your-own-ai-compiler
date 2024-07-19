# 01: Hello World

学习任何一门语言的第一课都是 hello world，本章带你用少量代码来实现一个能编译并运行 python hello world 程序的编译器和运行时。代码实现部分虽然简单，但它贯穿了编译和运行的核心，理解了这些你大致就能理解编译器的工作原理了。

为了配合本章的内容，我专门对 fleet-compiler 相关代码做了简化，具体见：code/01/hello_compiler.py

# 目标

我们的目标是开发一个简单的编译器，hello_compiler.py，对下列 python 源码，hello.py，做编译和运行，在终端上显示 "Hello World!"。

```
def func():
  print('Hello World!')
func()
```

这个程序很简单，只有一个函数调用，被调用函数内只有一行代码：调用内置函数打印 "Hello World!"。

# 词法分析

稍微了解过编译原理的，大致都听过词法分析、语法分析、语义分析这些概念。词法分析是指用词法解析器（lexer）将源码文件解析成一系列的 token：

```
tokens = [
    Token(kind=TokenKind.Keyword, data='def'),
    Token(kind=TokenKind.Identifier, data='func'),
    Token(kind=TokenKind.Separator, data='('),
    Token(kind=TokenKind.Separator, data=')'),
    Token(kind=TokenKind.Separator, data=':'),
    Token(kind=TokenKind.Terminator, data=''),
    Token(kind=TokenKind.Indentation, data='  '),
    Token(kind=TokenKind.Identifier, data='print'),
    Token(kind=TokenKind.Separator, data='('),
    Token(kind=TokenKind.StringLiteral, data='Hello World!'),
    Token(kind=TokenKind.Separator, data=')'),
    Token(kind=TokenKind.Terminator, data=''),
    Token(kind=TokenKind.Identifier, data='func'),
    Token(kind=TokenKind.Separator, data='('),
    Token(kind=TokenKind.Separator, data=')'),
    Token(kind=TokenKind.Terminator, data=''),
]
```
每个 token 都赋予一个类型，比如，关键字 'def'，标识符 'func'，字符串字面量 'Hello World!' 等。

# 语法分析

语法分析是指用语法解析器（parser）将前面得到的 token 列表解析成一棵抽象语法树（AST）：

```
Module:
  Block:
    Function define func
      Signature
        []
      Block:
        Function Call print, arg_list: ['<Arg 0 = "Hello World!">']  (resolved)
    Function Call func, arg_list: []  (resolved)
```

AST 是源码的抽象表示，它是树结构的，每个 node 表示一个语法元素。比如，根节点 `Module` 表示源码/程序，`Block` 表示 body 部分。程序内的 `Function define` 和 `Function Call` 两个节点表示函数定义和函数调用。

```
class AstNode(ABC):
    @abstractmethod
    def accept(self, visitor):
        pass

class AstModule(AstNode):
    def __init__(self, block: Block) -> None:
        self.block = block

    def accept(self, visitor):
        return visitor.visitModule(self)

class Block(Statement):
    def __init__(self, stmts: list) -> None:
        super().__init__()
        self.stmts = stmts
        self.indent = None

    def accept(self, visitor):
        return visitor.visitBlock(self)
```

hello_compiler.py 通过 hardcode 的方式从 token 列表中匹配出 AstModule, FunctionDef, FunctionCall 节点。完整的 parser 的实现会在后续再介绍。

```
def parser(tokens: list[Token]):
    def parse_func_call(tokens: list[Token]):
        assert tokens[0].kind == TokenKind.Identifier
        name = tokens[0].data
        assert tokens[1].data == '('
        args = []
        if tokens[2].data != ')':
            assert tokens[2].kind == TokenKind.StringLiteral and tokens[3].data == ')'
            args.append(tokens[2].data)
            tokens = tokens[4:]
        else:
            tokens = tokens[3:]
        return FunctionCall(name, args), tokens[1:]

    def parse_func_def(tokens: list[Token]):
        assert tokens[0].data == 'def'
        name = tokens[1].data
        assert tokens[2].data == '(' and tokens[3].data == ')' and \
            tokens[4].data == ':' and tokens[5].kind == TokenKind.Terminator
        assert tokens[6].kind == TokenKind.Indentation
        call_stmt, tokens = parse_func_call(tokens[7:])
        return FunctionDef(name, Block([call_stmt])), tokens

    func_def, tokens = parse_func_def(tokens)
    func_call, tokens = parse_func_call(tokens)
    return AstModule(Block([func_def, func_call]))
```

# 遍历 AST

通过访问者模式可以从根节点（AstModule）出发遍历 AST，来完成：语义分析、图优化、MLIR 转换、解析执行等操作。

```
class AstVisitor:
    def visit(self, node: AstNode):
        return node.accept(self)

    def visitModule(self, node: AstModule):
        return self.visitBlock(node.block)

    def visitBlock(self, node: Block):
        for o in node.stmts:
            self.visit(o)

    def visitFunctionDef(self, node: FunctionDef):
        self.visitBlock(node.block)
    
    def visitFunctionCall(self, node: FunctionCall): ...
```


# 语义分析

语义分析涉及的内容很多，像是类型检查/推理，变量作用域/生命周期分析管理，控制流分析/优化，数据流分析/优化等。本章只关心搞定 hello.py 所需的技术：引用消解，也就是让函数调用节点（Function Call）能找到它要引用的函数（Function define）。


这过程涉及到符号管理，即在创建了函数节点后，在键值对符号表中插入记录 {"func": func_def_node}。这样，在执行函数调用时，就能从符号表中查找符号 "func" 对应的函数，然后遍历执行 function block 中的节点。

```
class SemanticAnalysis(AstVisitor):
    def __init__(self, sym_table: SymbolTable) -> None:
        super().__init__()
        self.sym_table = sym_table

    def visitFunctionDef(self, node: FunctionDef):
        self.visitBlock(node.block)
        self.sym_table.update(node.name, node)
```

# 运行时：AST 解释器

执行 AST 需实现一个解释器来遍历并执行 AST 节点。它遍历树节点的过程和做语义分析的pass大体一致，区别在于它需要从符号表查找目标函数，处理输入输出等。

```
class Interpreter(AstVisitor):
    def __init__(self, sym_table: SymbolTable) -> None:
        super().__init__()
        self.sym_table = sym_table
    
    def visitModule(self, node: AstModule):
        for stmt in node.block.stmts:
            if isinstance(stmt, FunctionCall):
                self.visit(stmt)

    def visitFunctionCall(self, node: FunctionCall):
        if node.name == 'print':
            print(node.args[0])
        else:
            func_def = self.sym_table.get(node.name)
            for stmt in func_def.block.stmts:
                self.visit(stmt)
```

# 完整流程

```
# 词法分析
tokens = lexer()

# 语法分析
ast_module = parser(tokens)

# 语义分析
sym_table = SymbolTable()
SemanticAnalysis(sym_table).visit(ast_module)

# 解释执行
Interpreter(sym_table).visit(ast_module)


# Hello World!
```

# END

怎么样，Hello world 级别的编译器也不难实现吧？虽然它现在有不少 hardcode，但不妨碍它成为打通编译流程的极简案例。

---
