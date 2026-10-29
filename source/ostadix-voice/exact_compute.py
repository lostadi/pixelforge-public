"""Bounded exact arithmetic; never evaluates Python code or invokes a model."""
import ast
from fractions import Fraction
import re

def calculate(expression):
    if not isinstance(expression, str) or not 1 <= len(expression) <= 200:
        raise ValueError('Use an arithmetic expression of 1–200 characters')
    tree = ast.parse(expression.strip(), mode='eval')
    if sum(1 for _ in ast.walk(tree)) > 64:
        raise ValueError('Expression is too complex')
    def bounded(value):
        if value.numerator.bit_length() > 2048 or value.denominator.bit_length() > 2048:
            raise ValueError('Arithmetic result exceeds the size limit')
        return value
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            literal = ast.get_source_segment(expression.strip(), node)
            if not literal or len(literal) > 64 or not re.fullmatch(r'[0-9]+(?:\.[0-9]+)?', literal):
                raise ValueError('Use ordinary integer or decimal literals')
            return bounded(Fraction(literal))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            a,b=visit(node.left),visit(node.right)
            if isinstance(node.op,ast.Add): value=a+b
            elif isinstance(node.op,ast.Sub): value=a-b
            elif isinstance(node.op,ast.Mult): value=a*b
            elif isinstance(node.op,ast.Div): value=a/b
            elif isinstance(node.op,ast.FloorDiv): value=Fraction(a//b)
            elif isinstance(node.op,ast.Mod): value=a%b
            elif isinstance(node.op,ast.Pow):
                if b.denominator!=1 or not -32 <= b.numerator <= 32:
                    raise ValueError('Exponent must be an integer between -32 and 32')
                value=a**b.numerator
            else: raise ValueError('Unsupported arithmetic operator')
            return bounded(value)
        raise ValueError('Only numeric arithmetic is allowed')
    value=visit(tree.body)
    return {'expression':expression,'exact':str(value),'numerator':value.numerator,
            'denominator':value.denominator,'backend':'deterministic-python','model_calls':0}
