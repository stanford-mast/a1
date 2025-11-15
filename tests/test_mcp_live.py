"""
Test MCP integration with REAL MCP servers.

This test uses actual MCP servers to test ToolSet.from_mcp_servers()
and tool invocation.
"""

import tempfile
from pathlib import Path

import pytest

from a1 import ToolSet


class TestMCPLiveIntegration:
    """Test MCP integration with real MCP servers."""

    @pytest.mark.asyncio
    async def test_load_tools_from_mcp_filesystem_server(self):
        """Test loading tools from the filesystem MCP server."""
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as tmpdir:
            # Resolve the real path (handles macOS symlinks like /var -> /private/var)
            tmpdir_resolved = str(Path(tmpdir).resolve())
            # Configure the filesystem MCP server
            config = {
                "mcpServers": {
                    "filesystem": {
                        "command": "npx",
                        "args": [
                            "-y",
                            "@modelcontextprotocol/server-filesystem",
                            tmpdir_resolved,  # Limit to temp directory for safety
                        ],
                    }
                }
            }

            # Load tools from MCP server
            toolset = await ToolSet.from_mcp_servers(config)

            assert toolset is not None
            assert len(toolset.tools) > 0

            tool_names = [tool.name for tool in toolset.tools]
            print(f"\n✓ Loaded {len(toolset.tools)} tools from filesystem MCP server")
            print(f"  Tools: {tool_names}")

            # Verify we have expected filesystem tools
            assert any("read" in name.lower() for name in tool_names), "Should have read tool"
            assert any("write" in name.lower() for name in tool_names), "Should have write tool"

    @pytest.mark.asyncio
    async def test_invoke_mcp_tool_read_file(self):
        """Test invoking MCP tool to read a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Resolve the real path (handles macOS symlinks like /var -> /private/var)
            tmpdir_resolved = str(Path(tmpdir).resolve())
            # Create a test file
            test_file = Path(tmpdir_resolved) / "test.txt"
            test_content = "Hello from MCP test!"
            test_file.write_text(test_content)

            # Configure MCP server
            config = {
                "mcpServers": {
                    "filesystem": {
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-filesystem", tmpdir_resolved],
                    }
                }
            }

            # Load tools
            toolset = await ToolSet.from_mcp_servers(config)

            # Find read_text_file tool (not deprecated read_file)
            read_tool = None
            for tool in toolset.tools:
                if "read_text_file" == tool.name.lower():
                    read_tool = tool
                    break

            assert read_tool is not None, "read_text_file tool not found"
            print(f"\n✓ Found read tool: {read_tool.name}")

            # Invoke the tool
            result = await read_tool(path=str(test_file))

            print(f"  Read file: {test_file}")
            print(f"  Result: {result}")

            assert result is not None
            assert hasattr(result, "content"), "Result should have content attribute"
            assert hasattr(result, "isError"), "Result should have isError attribute"
            assert result.isError is False, "Should not have error"
            assert test_content in str(result.content), "Content should match file contents"

            print(f"✓ MCP read_text_file tool works! Content: {str(result.content)[:50]}...")

    @pytest.mark.asyncio
    async def test_invoke_mcp_tool_write_file(self):
        """Test invoking MCP tool to write a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Resolve the real path (handles macOS symlinks like /var -> /private/var)
            tmpdir_resolved = str(Path(tmpdir).resolve())
            # Configure MCP server
            config = {
                "mcpServers": {
                    "filesystem": {
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-filesystem", tmpdir_resolved],
                    }
                }
            }

            # Load tools
            toolset = await ToolSet.from_mcp_servers(config)

            # Find write_file tool
            write_tool = None
            for tool in toolset.tools:
                if "write" in tool.name.lower() and "file" in tool.name.lower():
                    write_tool = tool
                    break

            assert write_tool is not None, "write_file tool not found"
            print(f"\n✓ Found write tool: {write_tool.name}")

            # Invoke the tool
            test_file = Path(tmpdir_resolved) / "output.txt"
            test_content = "Content written by MCP tool!"
            result = await write_tool(path=str(test_file), content=test_content)

            print(f"  Wrote file: {test_file}")
            print(f"  Result: {result}")

            assert result is not None
            assert result.isError is False, "Should not have error"

            # Verify file was created
            assert test_file.exists(), "File should exist"
            actual_content = test_file.read_text()
            assert actual_content == test_content, "File content should match"

            print("✓ MCP write_file tool works! File created successfully")

    @pytest.mark.asyncio
    async def test_invoke_mcp_tool_list_directory(self):
        """Test invoking MCP tool to list directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Resolve the real path (handles macOS symlinks like /var -> /private/var)
            tmpdir_resolved = str(Path(tmpdir).resolve())
            # Create some test files
            (Path(tmpdir_resolved) / "file1.txt").write_text("test1")
            (Path(tmpdir_resolved) / "file2.txt").write_text("test2")
            (Path(tmpdir_resolved) / "file3.md").write_text("test3")

            # Configure MCP server
            config = {
                "mcpServers": {
                    "filesystem": {
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-filesystem", tmpdir_resolved],
                    }
                }
            }

            # Load tools
            toolset = await ToolSet.from_mcp_servers(config)

            # Find list_directory or similar tool
            list_tool = None
            for tool in toolset.tools:
                tool_name_lower = tool.name.lower()
                if ("list" in tool_name_lower and "dir" in tool_name_lower) or ("directory" in tool_name_lower):
                    list_tool = tool
                    break

            if list_tool:  # This tool might not exist in all MCP filesystem versions
                print(f"\n✓ Found list tool: {list_tool.name}")

                # Invoke the tool
                result = await list_tool(path=tmpdir_resolved)

                print(f"  Listed directory: {tmpdir_resolved}")
                print(f"  Result: {result}")

                assert result is not None
                assert result.isError is False

                print("✓ MCP list_directory tool works!")
            else:
                print("\n⚠ list_directory tool not available in this MCP server version")

    @pytest.mark.asyncio
    async def test_multiple_mcp_servers(self):
        """Test loading tools from multiple MCP servers."""
        with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
            # Resolve the real paths (handles macOS symlinks like /var -> /private/var)
            tmpdir1_resolved = str(Path(tmpdir1).resolve())
            tmpdir2_resolved = str(Path(tmpdir2).resolve())
            # Configure two different filesystem servers
            config = {
                "mcpServers": {
                    "filesystem1": {
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-filesystem", tmpdir1_resolved],
                    },
                    "filesystem2": {
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-filesystem", tmpdir2_resolved],
                    },
                }
            }

            # Load tools from both servers
            toolset = await ToolSet.from_mcp_servers(config)

            assert toolset is not None
            # We should have tools from both servers
            # Each filesystem server provides multiple tools, so we should have many
            assert len(toolset.tools) >= 2, "Should have tools from both servers"

            print("\n✓ Loaded tools from multiple MCP servers")
            print(f"  Total tools: {len(toolset.tools)}")
            print(f"  Description: {toolset.description}")

            # Verify the description mentions both servers
            assert "filesystem1" in toolset.description
            assert "filesystem2" in toolset.description

    @pytest.mark.asyncio
    async def test_mcp_tool_metadata(self):
        """Test that MCP tools have correct metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Resolve the real path (handles macOS symlinks like /var -> /private/var)
            tmpdir_resolved = str(Path(tmpdir).resolve())
            config = {
                "mcpServers": {
                    "filesystem": {
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-filesystem", tmpdir_resolved],
                    }
                }
            }

            toolset = await ToolSet.from_mcp_servers(config)

            print("\n✓ MCP Toolset metadata:")
            print(f"  Name: {toolset.name}")
            print(f"  Description: {toolset.description}")

            # Check each tool has required metadata
            for tool in toolset.tools:
                assert tool.name is not None
                assert tool.description is not None or tool.description == ""
                assert tool.input_schema is not None
                assert tool.output_schema is not None
                print(f"  - {tool.name}: {tool.description[:60] if tool.description else 'no description'}...")

    @pytest.mark.asyncio
    async def test_mcp_tool_round_trip(self):
        """Test writing and reading back a file via MCP tools."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Resolve the real path (handles macOS symlinks like /var -> /private/var)
            tmpdir_resolved = str(Path(tmpdir).resolve())
            config = {
                "mcpServers": {
                    "filesystem": {
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-filesystem", tmpdir_resolved],
                    }
                }
            }

            toolset = await ToolSet.from_mcp_servers(config)

            # Find write and read_text_file tools
            write_tool = next(
                (t for t in toolset.tools if "write" in t.name.lower() and "file" in t.name.lower()), None
            )
            read_tool = next((t for t in toolset.tools if t.name.lower() == "read_text_file"), None)

            assert write_tool is not None
            assert read_tool is not None

            # Write a file
            test_file = Path(tmpdir_resolved) / "roundtrip.txt"
            test_content = "Round-trip test content!"
            write_result = await write_tool(path=str(test_file), content=test_content)
            assert write_result.isError is False

            # Read it back
            read_result = await read_tool(path=str(test_file))
            assert read_result.isError is False
            assert test_content in str(read_result.content)

            print("\n✓ MCP round-trip test passed!")
            print(f"  Wrote: {test_content}")
            print(f"  Read back: {str(read_result.content)[:50]}...")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
