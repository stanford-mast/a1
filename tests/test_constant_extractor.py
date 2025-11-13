"""
Test ConstantExtractor for compile-time value extraction.
"""

import ast
import pytest
from pydantic import BaseModel, Field

from a1.cfg_builder import ConstantExtractor


class TestConstantExtractor:
    """Test compile-time constant value extraction."""

    def test_literal_constants(self):
        """Test extraction of literal values."""
        code = """
x = 42
y = "hello"
z = True
w = None
"""
        tree = ast.parse(code)
        extractor = ConstantExtractor(tree)
        
        # Integer
        node = ast.parse("42", mode='eval').body
        value, is_const = extractor.extract_constant(node)
        assert is_const
        assert value == 42
        
        # String
        node = ast.parse('"hello"', mode='eval').body
        value, is_const = extractor.extract_constant(node)
        assert is_const
        assert value == "hello"
        
        # Boolean
        node = ast.parse('True', mode='eval').body
        value, is_const = extractor.extract_constant(node)
        assert is_const
        assert value is True
        
        # None
        node = ast.parse('None', mode='eval').body
        value, is_const = extractor.extract_constant(node)
        assert is_const
        assert value is None
    
    def test_variable_assignment(self):
        """Test extraction of values assigned to variables."""
        code = """
interface = "Gi1/0"
vlan = 100
"""
        tree = ast.parse(code)
        extractor = ConstantExtractor(tree)
        
        # Extract 'interface' variable
        node = ast.parse('interface', mode='eval').body
        value, is_const = extractor.extract_constant(node)
        assert is_const
        assert value == "Gi1/0"
        
        # Extract 'vlan' variable
        node = ast.parse('vlan', mode='eval').body
        value, is_const = extractor.extract_constant(node)
        assert is_const
        assert value == 100
    
    def test_arithmetic_on_constants(self):
        """Test extraction of arithmetic expressions on constants."""
        code = """
x = 10
"""
        tree = ast.parse(code)
        extractor = ConstantExtractor(tree)
        
        # Addition
        node = ast.parse('2 + 3', mode='eval').body
        value, is_const = extractor.extract_constant(node)
        assert is_const
        assert value == 5
        
        # Subtraction
        node = ast.parse('10 - 3', mode='eval').body
        value, is_const = extractor.extract_constant(node)
        assert is_const
        assert value == 7
        
        # Multiplication
        node = ast.parse('4 * 5', mode='eval').body
        value, is_const = extractor.extract_constant(node)
        assert is_const
        assert value == 20
        
        # String concatenation
        node = ast.parse('"hello" + " " + "world"', mode='eval').body
        value, is_const = extractor.extract_constant(node)
        assert is_const
        assert value == "hello world"
    
    def test_list_literal(self):
        """Test extraction of list literals."""
        code = ""
        tree = ast.parse(code)
        extractor = ConstantExtractor(tree)
        
        node = ast.parse('[1, 2, 3]', mode='eval').body
        value, is_const = extractor.extract_constant(node)
        assert is_const
        assert value == [1, 2, 3]
        
        # List with mixed types
        node = ast.parse('[1, "hello", True]', mode='eval').body
        value, is_const = extractor.extract_constant(node)
        assert is_const
        assert value == [1, "hello", True]
    
    def test_dict_literal(self):
        """Test extraction of dict literals."""
        code = ""
        tree = ast.parse(code)
        extractor = ConstantExtractor(tree)
        
        node = ast.parse('{"a": 1, "b": 2}', mode='eval').body
        value, is_const = extractor.extract_constant(node)
        assert is_const
        assert value == {"a": 1, "b": 2}
    
    def test_subscript_access(self):
        """Test extraction of subscript access on constants."""
        code = """
my_list = [10, 20, 30]
my_dict = {"key": "value"}
"""
        tree = ast.parse(code)
        extractor = ConstantExtractor(tree)
        
        # List subscript
        node = ast.parse('my_list[1]', mode='eval').body
        value, is_const = extractor.extract_constant(node)
        assert is_const
        assert value == 20
        
        # Dict subscript
        node = ast.parse('my_dict["key"]', mode='eval').body
        value, is_const = extractor.extract_constant(node)
        assert is_const
        assert value == "value"
    
    def test_chained_variable_assignment(self):
        """Test extraction through chain of variable assignments."""
        code = """
a = "Gi1/0"
b = a
c = b
"""
        tree = ast.parse(code)
        extractor = ConstantExtractor(tree)
        
        # Extract 'c' which should resolve to "Gi1/0"
        node = ast.parse('c', mode='eval').body
        value, is_const = extractor.extract_constant(node)
        assert is_const
        assert value == "Gi1/0"
    
    def test_non_constant_variable(self):
        """Test that undefined variables are not considered constant."""
        code = """
x = some_function()
"""
        tree = ast.parse(code)
        extractor = ConstantExtractor(tree)
        
        # 'unknown_var' is not defined
        node = ast.parse('unknown_var', mode='eval').body
        value, is_const = extractor.extract_constant(node)
        assert not is_const
        assert value is None
    
    def test_function_call_not_constant(self):
        """Test that function calls are not considered constant."""
        code = ""
        tree = ast.parse(code)
        extractor = ConstantExtractor(tree)
        
        node = ast.parse('get_value()', mode='eval').body
        value, is_const = extractor.extract_constant(node)
        assert not is_const
        assert value is None
    
    def test_unary_operations(self):
        """Test extraction of unary operations on constants."""
        code = ""
        tree = ast.parse(code)
        extractor = ConstantExtractor(tree)
        
        # Negation
        node = ast.parse('-5', mode='eval').body
        value, is_const = extractor.extract_constant(node)
        assert is_const
        assert value == -5
        
        # Logical not
        node = ast.parse('not True', mode='eval').body
        value, is_const = extractor.extract_constant(node)
        assert is_const
        assert value is False
    
    def test_f_string_with_constants(self):
        """Test extraction of f-strings with constant values."""
        code = """
name = "Alice"
age = 30
"""
        tree = ast.parse(code)
        extractor = ConstantExtractor(tree)
        
        # f-string with variables
        node = ast.parse('f"Hello, {name}! You are {age}."', mode='eval').body
        value, is_const = extractor.extract_constant(node)
        assert is_const
        assert value == "Hello, Alice! You are 30."
    
    def test_variable_assigned_to_expression(self):
        """Test variables assigned to constant expressions."""
        code = """
base = 100
offset = 50
total = base + offset
"""
        tree = ast.parse(code)
        extractor = ConstantExtractor(tree)
        
        node = ast.parse('total', mode='eval').body
        value, is_const = extractor.extract_constant(node)
        assert is_const
        assert value == 150
    
    def test_tuple_literal(self):
        """Test extraction of tuple literals."""
        code = ""
        tree = ast.parse(code)
        extractor = ConstantExtractor(tree)
        
        node = ast.parse('(1, 2, 3)', mode='eval').body
        value, is_const = extractor.extract_constant(node)
        assert is_const
        assert value == (1, 2, 3)
    
    def test_set_literal(self):
        """Test extraction of set literals."""
        code = ""
        tree = ast.parse(code)
        extractor = ConstantExtractor(tree)
        
        node = ast.parse('{1, 2, 3}', mode='eval').body
        value, is_const = extractor.extract_constant(node)
        assert is_const
        assert value == {1, 2, 3}
    
    def test_input_schema_access(self):
        """Test that input.field is recognized as potentially constant."""
        
        class TestInput(BaseModel):
            vlan_id: int = 100  # Default value
        
        code = """
config = await configure(vlan=input.vlan_id)
"""
        tree = ast.parse(code)
        extractor = ConstantExtractor(tree, input_schema=TestInput)
        
        # 'input' is recognized
        node = ast.parse('input', mode='eval').body
        value, is_const = extractor.extract_constant(node)
        assert is_const
        assert value == ('__INPUT__', 'input')
        
        # 'input.vlan_id' should be recognized as knowable
        node = ast.parse('input.vlan_id', mode='eval').body
        value, is_const = extractor.extract_constant(node)
        assert is_const
        # Should either return the default or mark as input field
        assert value == 100 or value[0] == '__INPUT_FIELD__'
