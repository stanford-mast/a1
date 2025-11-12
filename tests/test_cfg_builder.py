"""
Tests for CFG builder module.
"""

import ast

from a1.cfg_builder import BasicBlock, CFGBuilder


class TestCFGBuilder:
    """Test CFG construction."""

    def test_simple_sequence(self):
        """Test CFG for simple sequence of statements."""
        code = """
x = 1
y = 2
z = 3
"""
        cfg = CFGBuilder()
        start, blocks = cfg.build(ast.parse(code))

        assert start is not None
        assert len(start.stmts) == 3
        assert len(start.next) == 0  # No successors

    def test_if_statement(self):
        """Test CFG for if statement."""
        code = """
if x > 0:
    y = 1
else:
    y = 2
z = 3
"""
        cfg = CFGBuilder()
        start, blocks = cfg.build(ast.parse(code))

        # Should have: start (if), then, else, after
        assert len(blocks) >= 4
        assert len(start.next) >= 2  # if and else branches

    def test_for_loop(self):
        """Test CFG for for loop."""
        code = """
for i in range(10):
    x = i * 2
y = x + 1
"""
        cfg = CFGBuilder()
        start, blocks = cfg.build(ast.parse(code))

        # Should have: start, loop header, loop body, after loop
        assert len(blocks) >= 4

    def test_while_loop(self):
        """Test CFG for while loop."""
        code = """
while x > 0:
    x = x - 1
y = x
"""
        cfg = CFGBuilder()
        start, blocks = cfg.build(ast.parse(code))

        # Should have: start, loop header, loop body, after loop
        assert len(blocks) >= 4

    def test_function_calls(self):
        """Test extracting function calls."""
        code = """
x = await tool_a(1, 2)
y = tool_b(x)
"""
        cfg = CFGBuilder()
        start, blocks = cfg.build(ast.parse(code))

        # Should extract both calls
        all_calls = []
        for block in blocks.values():
            all_calls.extend(block.calls)

        assert len(all_calls) == 2
        assert all_calls[0][0] == "tool_a"  # function name
        assert all_calls[1][0] == "tool_b"

    def test_comprehension_depth(self):
        """Test comprehension depth tracking."""
        code = """
result = [await tool(x) for x in items]
"""
        cfg = CFGBuilder()
        start, blocks = cfg.build(ast.parse(code))

        # Find the tool call
        calls = []
        for block in blocks.values():
            calls.extend(block.calls)

        assert len(calls) == 1
        func_name, call_node, comp_depth = calls[0]
        assert func_name == "tool"
        assert comp_depth == 1  # One level of comprehension

    def test_nested_comprehension(self):
        """Test nested comprehension depth."""
        code = """
result = [await tool(x, y) for x in items for y in x.subitems]
"""
        cfg = CFGBuilder()
        start, blocks = cfg.build(ast.parse(code))

        calls = []
        for block in blocks.values():
            calls.extend(block.calls)

        assert len(calls) == 1
        func_name, call_node, comp_depth = calls[0]
        assert comp_depth == 2  # Two generators = depth 2

    def test_try_except(self):
        """Test CFG for try/except."""
        code = """
try:
    x = risky()
except Exception:
    x = 0
finally:
    cleanup()
"""
        cfg = CFGBuilder()
        start, blocks = cfg.build(ast.parse(code))

        # Should have: start, try body, except handler, finally, after
        assert len(blocks) >= 5

    def test_nested_loops(self):
        """Test CFG for nested loops."""
        code = """
for i in range(10):
    for j in range(5):
        x = await tool(i, j)
"""
        cfg = CFGBuilder()
        start, blocks = cfg.build(ast.parse(code))

        # Should have multiple blocks for nested structure
        assert len(blocks) >= 5

        # Find tool call (not range calls)
        calls = []
        for block in blocks.values():
            for func_name, call_node, comp_depth in block.calls:
                if func_name == "tool":
                    calls.append((func_name, call_node, comp_depth))

        assert len(calls) == 1
        assert calls[0][0] == "tool"
        # Comprehension depth is 0 because these are for loops, not comprehensions
        assert calls[0][2] == 0


class TestBasicBlock:
    """Test BasicBlock class."""

    def test_basic_block_creation(self):
        """Test creating a basic block."""
        block = BasicBlock(bid=1)
        assert block.bid == 1
        assert block.is_empty()
        assert len(block.calls) == 0

    def test_basic_block_with_statements(self):
        """Test basic block with statements."""
        block = BasicBlock(bid=1)
        stmt = ast.parse("x = 1").body[0]
        block.stmts.append(stmt)

        assert not block.is_empty()
        assert len(block.stmts) == 1

    def test_basic_block_edges(self):
        """Test basic block edges."""
        block1 = BasicBlock(bid=1)
        block2 = BasicBlock(bid=2)

        block1.next.append(block2.bid)
        block2.prev.append(block1.bid)

        assert block2.bid in block1.next
        assert block1.bid in block2.prev
