"""
Control Flow Graph builder for Python code analysis.

Builds CFG from AST to enable:
- Loop depth calculation
- Tool call ordering validation
- Cost estimation based on execution paths
"""

import ast
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BasicBlock:
    """Represents a basic block in the control flow graph."""

    bid: int
    stmts: list[ast.AST] = field(default_factory=list)
    calls: list[tuple[str, ast.Call, int]] = field(
        default_factory=list
    )  # (function_name, call_node, comprehension_depth)
    prev: list[int] = field(default_factory=list)
    next: list[int] = field(default_factory=list)

    def is_empty(self) -> bool:
        return len(self.stmts) == 0


class CFGBuilder(ast.NodeVisitor):
    """Build control flow graph from AST, handling all Python constructs."""

    def __init__(self):
        self.block_counter = 0
        self.blocks: dict[int, BasicBlock] = {}
        self.edges: dict[tuple[int, int], ast.AST | None] = {}
        self.curr_block: BasicBlock | None = None
        self.start_block: BasicBlock | None = None
        self.loop_stack: list[BasicBlock] = []
        self.in_comprehension = False
        self.comprehension_depth = 0  # Track nesting of comprehensions

    def new_block(self) -> BasicBlock:
        """Create a new basic block."""
        self.block_counter += 1
        block = BasicBlock(self.block_counter)
        self.blocks[self.block_counter] = block
        return block

    def add_edge(self, from_bid: int, to_bid: int, condition: ast.AST | None = None):
        """Add an edge between blocks."""
        self.blocks[from_bid].next.append(to_bid)
        self.blocks[to_bid].prev.append(from_bid)
        self.edges[(from_bid, to_bid)] = condition
        return self.blocks[to_bid]

    def add_stmt(self, block: BasicBlock, stmt: ast.AST):
        """Add statement to block."""
        block.stmts.append(stmt)

    def build(self, tree: ast.AST) -> tuple[BasicBlock, dict[int, BasicBlock]]:
        """Build CFG from AST."""
        self.curr_block = self.new_block()
        self.start_block = self.curr_block
        self.visit(tree)
        return self.start_block, self.blocks

    def extract_calls_from_node(self, node: ast.AST):
        """Recursively extract function calls from any AST node."""
        if isinstance(node, ast.Await):
            if isinstance(node.value, ast.Call):
                func_name = self.get_function_name(node.value.func)
                if func_name:
                    self.curr_block.calls.append((func_name, node.value, self.comprehension_depth))
        elif isinstance(node, ast.Call):
            func_name = self.get_function_name(node.func)
            if func_name:
                self.curr_block.calls.append((func_name, node, self.comprehension_depth))
            # Check arguments for nested calls
            for arg in node.args:
                self.extract_calls_from_node(arg)
            for keyword in node.keywords:
                self.extract_calls_from_node(keyword.value)
        elif isinstance(node, ast.BinOp):
            self.extract_calls_from_node(node.left)
            self.extract_calls_from_node(node.right)
        elif isinstance(node, ast.UnaryOp):
            self.extract_calls_from_node(node.operand)
        elif isinstance(node, ast.Compare):
            self.extract_calls_from_node(node.left)
            for comp in node.comparators:
                self.extract_calls_from_node(comp)
        elif isinstance(node, ast.BoolOp):
            for value in node.values:
                self.extract_calls_from_node(value)
        elif isinstance(node, ast.IfExp):
            # Handle ternary expressions
            self.extract_calls_from_node(node.test)
            self.extract_calls_from_node(node.body)
            self.extract_calls_from_node(node.orelse)
        elif isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            for elt in node.elts:
                self.extract_calls_from_node(elt)
        elif isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if k:
                    self.extract_calls_from_node(k)
                self.extract_calls_from_node(v)
        elif isinstance(node, ast.Subscript):
            self.extract_calls_from_node(node.value)
            self.extract_calls_from_node(node.slice)
        elif isinstance(node, ast.Attribute):
            self.extract_calls_from_node(node.value)
        elif isinstance(node, ast.ListComp):
            self.visit_ListComp(node)
        elif isinstance(node, ast.SetComp):
            self.visit_SetComp(node)
        elif isinstance(node, ast.DictComp):
            self.visit_DictComp(node)
        elif isinstance(node, ast.GeneratorExp):
            self.visit_GeneratorExp(node)

    def visit_Expr(self, node: ast.Expr):
        """Visit expression statement - check for function calls."""
        self.extract_calls_from_node(node.value)
        self.add_stmt(self.curr_block, node)

    def visit_Assign(self, node: ast.Assign):
        """Visit assignment - check if RHS is a function call."""
        self.extract_calls_from_node(node.value)
        self.add_stmt(self.curr_block, node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        """Visit annotated assignment."""
        if node.value:
            self.extract_calls_from_node(node.value)
        self.add_stmt(self.curr_block, node)

    def visit_AugAssign(self, node: ast.AugAssign):
        """Visit augmented assignment (+=, -=, etc.)."""
        self.extract_calls_from_node(node.value)
        self.add_stmt(self.curr_block, node)

    def visit_Return(self, node: ast.Return):
        """Visit return statement."""
        if node.value:
            self.extract_calls_from_node(node.value)
        self.add_stmt(self.curr_block, node)
        # Create new unreachable block after return
        self.curr_block = self.new_block()

    def visit_If(self, node: ast.If):
        """Visit if statement."""
        self.extract_calls_from_node(node.test)
        self.add_stmt(self.curr_block, node)

        # Create afterif block
        afterif_block = self.new_block()

        # Create if body block
        if_block = self.add_edge(self.curr_block.bid, self.new_block().bid, node.test)

        # Handle else
        if node.orelse:
            else_block = self.add_edge(self.curr_block.bid, self.new_block().bid)
            self.curr_block = else_block
            for stmt in node.orelse:
                self.visit(stmt)
            if not self.curr_block.next:
                self.add_edge(self.curr_block.bid, afterif_block.bid)
        else:
            self.add_edge(self.curr_block.bid, afterif_block.bid)

        # Handle if body
        self.curr_block = if_block
        for stmt in node.body:
            self.visit(stmt)
        if not self.curr_block.next:
            self.add_edge(self.curr_block.bid, afterif_block.bid)

        self.curr_block = afterif_block

    def visit_For(self, node: ast.For):
        """Visit for loop."""
        self.extract_calls_from_node(node.iter)

        loop_block = self.new_block()
        self.add_edge(self.curr_block.bid, loop_block.bid)
        self.curr_block = loop_block
        self.add_stmt(self.curr_block, node)

        after_loop = self.new_block()
        self.add_edge(self.curr_block.bid, after_loop.bid)

        body_block = self.add_edge(self.curr_block.bid, self.new_block().bid)
        self.loop_stack.append(after_loop)
        self.curr_block = body_block

        for stmt in node.body:
            self.visit(stmt)
        if not self.curr_block.next:
            self.add_edge(self.curr_block.bid, loop_block.bid)

        # Handle else clause
        if node.orelse:
            else_block = self.new_block()
            self.add_edge(loop_block.bid, else_block.bid)
            self.curr_block = else_block
            for stmt in node.orelse:
                self.visit(stmt)
            if not self.curr_block.next:
                self.add_edge(self.curr_block.bid, after_loop.bid)

        self.loop_stack.pop()
        self.curr_block = after_loop

    def visit_While(self, node: ast.While):
        """Visit while loop."""
        self.extract_calls_from_node(node.test)

        loop_block = self.new_block()
        self.add_edge(self.curr_block.bid, loop_block.bid)
        self.curr_block = loop_block
        self.add_stmt(self.curr_block, node)

        after_loop = self.new_block()
        self.add_edge(self.curr_block.bid, after_loop.bid)

        body_block = self.add_edge(self.curr_block.bid, self.new_block().bid, node.test)
        self.loop_stack.append(after_loop)
        self.curr_block = body_block

        for stmt in node.body:
            self.visit(stmt)
        if not self.curr_block.next:
            self.add_edge(self.curr_block.bid, loop_block.bid)

        # Handle else clause
        if node.orelse:
            else_block = self.new_block()
            self.add_edge(loop_block.bid, else_block.bid)
            self.curr_block = else_block
            for stmt in node.orelse:
                self.visit(stmt)
            if not self.curr_block.next:
                self.add_edge(self.curr_block.bid, after_loop.bid)

        self.loop_stack.pop()
        self.curr_block = after_loop

    def visit_Try(self, node: ast.Try):
        """Visit try/except."""
        self.add_stmt(self.curr_block, node)

        after_try = self.new_block()

        # Try body
        try_block = self.add_edge(self.curr_block.bid, self.new_block().bid)
        self.curr_block = try_block
        for stmt in node.body:
            self.visit(stmt)
        if not self.curr_block.next:
            self.add_edge(self.curr_block.bid, after_try.bid)

        # Exception handlers
        for handler in node.handlers:
            handler_block = self.add_edge(try_block.bid, self.new_block().bid)
            self.curr_block = handler_block
            for stmt in handler.body:
                self.visit(stmt)
            if not self.curr_block.next:
                self.add_edge(self.curr_block.bid, after_try.bid)

        # Else clause
        if node.orelse:
            else_block = self.add_edge(try_block.bid, self.new_block().bid)
            self.curr_block = else_block
            for stmt in node.orelse:
                self.visit(stmt)
            if not self.curr_block.next:
                self.add_edge(self.curr_block.bid, after_try.bid)

        # Finally clause
        if node.finalbody:
            finally_block = self.add_edge(after_try.bid, self.new_block().bid)
            self.curr_block = finally_block
            for stmt in node.finalbody:
                self.visit(stmt)
            after_finally = self.new_block()
            if not self.curr_block.next:
                self.add_edge(self.curr_block.bid, after_finally.bid)
            self.curr_block = after_finally
        else:
            self.curr_block = after_try

    def visit_With(self, node: ast.With):
        """Visit with statement."""
        for item in node.items:
            self.extract_calls_from_node(item.context_expr)
        self.add_stmt(self.curr_block, node)
        for stmt in node.body:
            self.visit(stmt)

    def visit_Break(self, node: ast.Break):
        """Visit break statement."""
        if self.loop_stack:
            self.add_edge(self.curr_block.bid, self.loop_stack[-1].bid)
        self.curr_block = self.new_block()

    def visit_Continue(self, node: ast.Continue):
        """Visit continue statement."""
        # Continue jumps back to loop guard (not implemented fully)
        self.curr_block = self.new_block()

    def visit_ListComp(self, node: ast.ListComp):
        """Visit list comprehension - extract calls but don't create loop blocks."""
        old_comp = self.in_comprehension
        self.in_comprehension = True
        num_generators = len(node.generators)
        self.comprehension_depth += num_generators

        self.extract_calls_from_node(node.elt)
        for gen in node.generators:
            self.extract_calls_from_node(gen.iter)
            for if_clause in gen.ifs:
                self.extract_calls_from_node(if_clause)

        self.comprehension_depth -= num_generators
        self.in_comprehension = old_comp

    def visit_SetComp(self, node: ast.SetComp):
        """Visit set comprehension - extract calls but don't create loop blocks."""
        old_comp = self.in_comprehension
        self.in_comprehension = True
        num_generators = len(node.generators)
        self.comprehension_depth += num_generators

        self.extract_calls_from_node(node.elt)
        for gen in node.generators:
            self.extract_calls_from_node(gen.iter)
            for if_clause in gen.ifs:
                self.extract_calls_from_node(if_clause)

        self.comprehension_depth -= num_generators
        self.in_comprehension = old_comp

    def visit_DictComp(self, node: ast.DictComp):
        """Visit dict comprehension - extract calls but don't create loop blocks."""
        old_comp = self.in_comprehension
        self.in_comprehension = True
        num_generators = len(node.generators)
        self.comprehension_depth += num_generators

        self.extract_calls_from_node(node.key)
        self.extract_calls_from_node(node.value)
        for gen in node.generators:
            self.extract_calls_from_node(gen.iter)
            for if_clause in gen.ifs:
                self.extract_calls_from_node(if_clause)

        self.comprehension_depth -= num_generators
        self.in_comprehension = old_comp

    def visit_GeneratorExp(self, node: ast.GeneratorExp):
        """Visit generator expression - extract calls but don't create loop blocks."""
        old_comp = self.in_comprehension
        self.in_comprehension = True
        num_generators = len(node.generators)
        self.comprehension_depth += num_generators

        self.extract_calls_from_node(node.elt)
        for gen in node.generators:
            self.extract_calls_from_node(gen.iter)
            for if_clause in gen.ifs:
                self.extract_calls_from_node(if_clause)

        self.comprehension_depth -= num_generators
        self.in_comprehension = old_comp

    def get_function_name(self, node: ast.AST) -> str | None:
        """Extract function name from call node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        return None


class ConstantExtractor:
    """
    Extract compile-time knowable constant values from AST nodes.
    
    Traces through variable assignments, attribute access, and simple expressions
    to determine if a value is constant at compile time.
    
    Handles:
    - Literals (42, "hello", True, None)
    - Variables assigned to constants
    - Attribute access on constants (obj.field.subfield)
    - Function parameters (if we know input schema structure)
    - Simple arithmetic on constants
    - List/dict/set literals with constant elements
    """
    
    def __init__(self, tree: ast.AST, input_schema=None):
        """
        Initialize constant extractor.
        
        Args:
            tree: AST of the code to analyze
            input_schema: Optional Pydantic model representing function input
                         (allows resolving input.field to schema default values)
        """
        self.tree = tree
        self.input_schema = input_schema
        
        # Build variable assignment map: var_name -> value_node
        self.assignments: dict[str, ast.AST] = {}
        self._build_assignment_map(tree)
    
    def _build_assignment_map(self, node: ast.AST):
        """Build map of variable assignments in the code."""
        for stmt in ast.walk(node):
            # Simple assignment: x = value
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        self.assignments[target.id] = stmt.value
            
            # Annotated assignment: x: int = value
            elif isinstance(stmt, ast.AnnAssign):
                if isinstance(stmt.target, ast.Name) and stmt.value:
                    self.assignments[stmt.target.id] = stmt.value
    
    def extract_constant(self, node: ast.AST) -> tuple[Any, bool]:
        """
        Extract constant value from AST node if knowable at compile time.
        
        Args:
            node: AST node to extract value from
        
        Returns:
            Tuple of (value, is_constant)
            - value: The constant value (None if not constant)
            - is_constant: True if value is knowable at compile time
        """
        # Direct constants
        if isinstance(node, ast.Constant):
            return node.value, True
        
        # Python <3.8 compatibility - these are deprecated but still used in some contexts
        # Check for Constant first (preferred), then fall back to legacy nodes
        if hasattr(ast, 'Num') and isinstance(node, ast.Num):
            return node.n, True
        if hasattr(ast, 'Str') and isinstance(node, ast.Str):
            return node.s, True
        if hasattr(ast, 'NameConstant') and isinstance(node, ast.NameConstant):
            return node.value, True
        if hasattr(ast, 'Bytes') and isinstance(node, ast.Bytes):
            return node.s, True
        
        # Variable reference - look up assignment
        if isinstance(node, ast.Name):
            var_name = node.id
            
            # Check if it's a function parameter referencing input schema
            if var_name == 'input' and self.input_schema:
                # Return marker that this is the input object
                return ('__INPUT__', var_name), True
            
            # Look up variable assignment
            if var_name in self.assignments:
                return self.extract_constant(self.assignments[var_name])
            
            # Unknown variable
            return None, False
        
        # Attribute access: obj.field.subfield
        if isinstance(node, ast.Attribute):
            base_value, is_const = self.extract_constant(node.value)
            
            if not is_const:
                return None, False
            
            # Handle access on input schema
            if isinstance(base_value, tuple) and base_value[0] == '__INPUT__':
                # This is input.field - check if we can resolve it
                if self.input_schema and hasattr(self.input_schema, 'model_fields'):
                    field_name = node.attr
                    if field_name in self.input_schema.model_fields:
                        field_info = self.input_schema.model_fields[field_name]
                        # If field has a default value, return it
                        if hasattr(field_info, 'default') and field_info.default is not ...:
                            return field_info.default, True
                        # Mark as input.field access (compile-time knowable structure)
                        return ('__INPUT_FIELD__', base_value[1], node.attr), True
                
                return ('__INPUT_FIELD__', base_value[1], node.attr), True
            
            # Handle nested attribute access on constants
            if isinstance(base_value, dict):
                # Dictionary attribute access
                if node.attr in base_value:
                    return base_value[node.attr], True
                return None, False
            
            # Can't resolve attribute on non-dict constant
            return None, False
        
        # List literal: [1, 2, 3]
        if isinstance(node, ast.List):
            elements = []
            for elt in node.elts:
                val, is_const = self.extract_constant(elt)
                if not is_const:
                    return None, False
                elements.append(val)
            return elements, True
        
        # Tuple literal: (1, 2, 3)
        if isinstance(node, ast.Tuple):
            elements = []
            for elt in node.elts:
                val, is_const = self.extract_constant(elt)
                if not is_const:
                    return None, False
                elements.append(val)
            return tuple(elements), True
        
        # Set literal: {1, 2, 3}
        if isinstance(node, ast.Set):
            elements = []
            for elt in node.elts:
                val, is_const = self.extract_constant(elt)
                if not is_const:
                    return None, False
                elements.append(val)
            return set(elements), True
        
        # Dict literal: {'a': 1, 'b': 2}
        if isinstance(node, ast.Dict):
            result = {}
            for k, v in zip(node.keys, node.values):
                if k is None:  # **kwargs in dict
                    return None, False
                
                key_val, key_const = self.extract_constant(k)
                val_val, val_const = self.extract_constant(v)
                
                if not (key_const and val_const):
                    return None, False
                
                result[key_val] = val_val
            return result, True
        
        # Binary operations on constants: 2 + 3, "hello" + "world"
        if isinstance(node, ast.BinOp):
            left, left_const = self.extract_constant(node.left)
            right, right_const = self.extract_constant(node.right)
            
            if not (left_const and right_const):
                return None, False
            
            # Evaluate simple operations
            try:
                if isinstance(node.op, ast.Add):
                    return left + right, True
                elif isinstance(node.op, ast.Sub):
                    return left - right, True
                elif isinstance(node.op, ast.Mult):
                    return left * right, True
                elif isinstance(node.op, ast.Div):
                    return left / right, True
                elif isinstance(node.op, ast.FloorDiv):
                    return left // right, True
                elif isinstance(node.op, ast.Mod):
                    return left % right, True
                elif isinstance(node.op, ast.Pow):
                    return left ** right, True
            except Exception:
                # Evaluation failed
                return None, False
            
            return None, False
        
        # Unary operations: -5, +5, not True
        if isinstance(node, ast.UnaryOp):
            operand, is_const = self.extract_constant(node.operand)
            
            if not is_const:
                return None, False
            
            try:
                if isinstance(node.op, ast.UAdd):
                    return +operand, True
                elif isinstance(node.op, ast.USub):
                    return -operand, True
                elif isinstance(node.op, ast.Not):
                    return not operand, True
                elif isinstance(node.op, ast.Invert):
                    return ~operand, True
            except Exception:
                return None, False
            
            return None, False
        
        # Subscript: list[0], dict['key']
        if isinstance(node, ast.Subscript):
            value, value_const = self.extract_constant(node.value)
            index, index_const = self.extract_constant(node.slice)
            
            if not (value_const and index_const):
                return None, False
            
            try:
                return value[index], True
            except Exception:
                return None, False
        
        # f-string (JoinedStr) - can be constant if all parts are constant
        if isinstance(node, ast.JoinedStr):
            parts = []
            for value in node.values:
                if isinstance(value, ast.Constant):
                    parts.append(str(value.value))
                elif isinstance(value, ast.FormattedValue):
                    val, is_const = self.extract_constant(value.value)
                    if not is_const:
                        return None, False
                    # Apply format spec if present
                    if value.format_spec:
                        return None, False  # Too complex for now
                    parts.append(str(val))
                else:
                    return None, False
            return ''.join(parts), True
        
        # Not a compile-time constant
        return None, False


__all__ = ["CFGBuilder", "BasicBlock", "ConstantExtractor"]
