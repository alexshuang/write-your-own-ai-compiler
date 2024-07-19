from abc import ABC, abstractmethod
from enum import Enum, auto


class TokenKind(Enum):
    Keyword = 0
    Identifier = auto()
    Separator = auto()
    StringLiteral = auto()
    IntegerLiteral = auto()
    DecimalLiteral = auto()
    NoneLiteral = auto()
    BooleanLiteral = auto()
    Terminator = auto()
    Indentation = auto()
    Newline = auto()
    OP = auto()
    EOF = auto()


class Token:
    def __init__(self, kind=TokenKind.EOF, data="") -> None:
        self.kind = kind
        self.data = data
    
    def __repr__(self) -> str:
        return self.data


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


class AstNode(ABC):
    @abstractmethod
    def accept(self, visitor):
        pass


class Statement(AstNode):
    def __init__(self) -> None:
        super().__init__()


class Expression(AstNode):
    def __init__(self) -> None:
        super().__init__()


class Block(Statement):
    def __init__(self, stmts: list) -> None:
        super().__init__()
        self.stmts = stmts
        self.indent = None

    def accept(self, visitor):
        return visitor.visitBlock(self)


class StringLiteral(Expression):
    def __init__(self, data: str) -> None:
        super().__init__()
        self.data = data
    
    def accept(self, visitor):
        return visitor.visitStringLiteral(self)


class AstModule(AstNode):
    def __init__(self, block: Block) -> None:
        self.block = block

    def accept(self, visitor):
        return visitor.visitModule(self)


class Block(AstNode):
    def __init__(self, stmts: list[AstNode]) -> None:
        self.stmts = stmts

    def accept(self, visitor):
        return visitor.visitBlock(self)


class FunctionDef(Statement):
    def __init__(self, name: str, block: Block) -> None:
        super().__init__()
        self.name = name
        self.block = block
    
    def accept(self, visitor):
        return visitor.visitFunctionDef(self)


class FunctionCall(Statement):
    def __init__(self, name: str, args: list[str]) -> None:
        super().__init__()
        self.name = name
        self.args = args
    
    def accept(self, visitor):
        return visitor.visitFunctionCall(self)


class SymbolTable:
    def __init__(self) -> None:
        self.sym_table = {}
    
    def update(self, name: str, node: AstNode):
        self.sym_table[name] = node
    
    def get(self, name: str):
        if name in self.sym_table:
            return self.sym_table[name]
        return None


def lexer():
    return tokens


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
            tokens[4].data == ':' and tokens[5].kind == TokenKind.Terminator and \
            tokens[6].kind == TokenKind.Indentation
        call_stmt, tokens = parse_func_call(tokens[7:])
        return FunctionDef(name, Block([call_stmt])), tokens

    func_def, tokens = parse_func_def(tokens)
    func_call, tokens = parse_func_call(tokens)
    return AstModule(Block([func_def, func_call]))


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


class SemanticAnalysis(AstVisitor):
    def __init__(self, sym_table: SymbolTable) -> None:
        super().__init__()
        self.sym_table = sym_table

    def visitFunctionDef(self, node: FunctionDef):
        self.visitBlock(node.block)
        self.sym_table.update(node.name, node)


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


if __name__ == '__main__':
    # 词法分析
    tokens = lexer()

    # 语法分析
    ast_module = parser(tokens)

    # 语义分析
    sym_table = SymbolTable()
    SemanticAnalysis(sym_table).visit(ast_module)

    # 解释执行
    Interpreter(sym_table).visit(ast_module)
