import ply.lex as lex
import ply.yacc as yacc

"""Module that deals with the grammar and parser for the basic computer algebra 
language.

Example:
   >>> parser.parse("[3 4 5] + [4 5 6]")
   (Exp (Sum (Vec (Elements 3  (Elements 4  (Element5)))) (Vec (Elements 4  (Elements 5  (Element6))))))


"""

tokens = [
    'LBRACKET', 'RBRACKET', 'PLUS', 'MINUS',
     'INTEGER', 'NAME', 'MUL'
]

t_LBRACKET = r'\['
t_RBRACKET = r'\]'
t_PLUS = r'\+'
t_MINUS = r'\-'
t_MUL = r'\*'

def t_NAME(t):
    r'[a-zA-Z_][a-zA-Z0-9_]*'
    t.type = reserved.get(t.value, 'NAME')
    return t

def t_INTEGER(t):
    r'-?[0-9]+'
    t.value = int(t.value)
    return t

t_ignore = '\t\n'

def t_error(t):
    print("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)
    
lexer = lex.lex()


### Grammar

def p_program(p):
    "expressions : expression expressions"
    p[0] = Exps(p[1], p[2])

def p_program_empty(p):
    "expressions : expression"
    p[0] = Exp(p[1])

def p_expression_int(p):
    "expression : INTEGER"
    p[0] = Int(p[1])

def p_expression_var(p):
    "expression : NAME"
    p[0] = Var(p[1])

def p_expression_vector(p):
    "expression : vector"
    p[0] = Vec(p[1])

def p_expression_matrix(p):
    "expression : matrix"
    p[0] = Matrix(p[1])


def p_expression_sum(p):
    "expression : expression PLUS expression"
    p[0] = Sum(p[1], p[3])

def p_expression_minus(p):
    "expression : expression MINUS expression"
    p[0] = Minus(p[1], p[3])

def p_expression_mul(p):
    "expression : expression MUL expression"
    p[0] = Product(p[1], p[3])

def p_vector(p):
    "vector : LBRACKET elements RBRACKET"
    p[0] = p[2]

def p_matrix(p):
    "matrix : LBRACKET vectors RBRACKET"
    p[0] = p[2]

def p_vectors(p):
    "vectors : vector vectors"
    p[0] = Vectors(p[1], p[2])

def p_vectors_one(p):
    "vectors : vector"
    p[0] = Vec(p[1])

def p_elements(p):
    "elements : INTEGER elements"
    p[0] = Elements(p[1], p[2])

def p_elements_int(p):
    "elements : INTEGER"
    p[0] = Element(p[1])

def p_error(p):
    print("Syntax error at '%s'" % p.value)
    
parser = yacc.yacc()

###-------------------
###  AST NODES
###------------------

class Exps:
    "PROGRAM node."
    __match_args__ = ('exp', 'expressions')
    def __init__(self, exp,  expressions):
        self.exp = exp
        self.expressions = expressions

    def __repr__(self):
        return f'(Exps {self.exp}  {self.expressions})'

class Exp:
    "PROGRAM node."
    __match_args__ = ('exp',)
    def __init__(self, exp):
        self.exp = exp
      

    def __repr__(self):
        return f'(Exp {self.exp})'

class Int:
    "INT node."
    __match_args__ = ('num',)
    def __init__(self, num):
        self.num = num

    def __repr__(self):
        return f'(Int {self.num})'

class Vec:
    __match_args__ = ('elements',)
    def __init__(self, elements):
        self.elements = elements
    
    def __repr__(self):
        return f'(Vec {self.elements})'

class Matrix:
    __match_args__ = ('elements',)
    def __init__(self, elements):
        self.elements = elements
    
    def __repr__(self):
        return f'(Matrix {self.elements})'


class Var:
    __match_args__ = ('var',)
    def __init__(self, var):
        self.var = var

    def __repr__(self):
        return f'(Var {self.var})'


class Sum:
    __match_args__ = ('e1', 'e2')
    def __init__(self, e1, e2):
        self.e1 = e1
        self.e2 = e2

    def __repr__(self):
        return f'(Sum {self.e1} {self.e2})'


class Minus:
    __match_args__ = ('e1', 'e2')
    def __init__(self, e1, e2):
        self.e1 = e1
        self.e2 = e2

    def __repr__(self):
        return f'(Minus {self.e1} {self.e2})'

class Product:
    __match_args__ = ('e1', 'e2')
    def __init__(self, e1, e2):
        self.e1 = e1
        self.e2 = e2

    def __repr__(self):
        return f'(Product {self.e1} {self.e2})'

class Elements:
    __match_args__ = ('e1', 'elements')
    "PROGRAM node."
    def __init__(self, e1,  elements):
        self.e1 = e1
        self.elements = elements

    def __repr__(self):
        return f'(Elements {self.e1}  {self.elements})'

class Element:
    __match_args__ = ('exp',)
    "PROGRAM node."
    def __init__(self, exp):
        self.exp = exp
      

    def __repr__(self):
        return f'(Element { self.exp})'

class Vectors:
    __match_args__ = ('exp', 'expressions')
    "PROGRAM node."
    def __init__(self, exp,  expressions):
        self.exp = exp
        self.expressions = expressions

    def __repr__(self):
        return f'(Vectors {self.exp}  {self.expressions})'
