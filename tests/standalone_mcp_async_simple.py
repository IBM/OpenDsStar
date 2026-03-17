"""
Simple test to verify async MCP tool support without full agent import.
"""

import importlib.util
import sys
from pathlib import Path

# Get the project root (parent of tests directory)
project_root = Path(__file__).parent.parent

# Add src to path
sys.path.insert(0, str(project_root / "src"))

# Import directly from module files to avoid import chain issues
spec_env = importlib.util.spec_from_file_location(
    "ds_star_execute_env",
    project_root / "src" / "agents" / "ds_star" / "ds_star_execute_env.py",
)
ds_star_execute_env = importlib.util.module_from_spec(spec_env)
spec_env.loader.exec_module(ds_star_execute_env)

spec_state = importlib.util.spec_from_file_location(
    "ds_star_state",
    project_root / "src" / "agents" / "ds_star" / "ds_star_state.py",
)
ds_star_state = importlib.util.module_from_spec(spec_state)
spec_state.loader.exec_module(ds_star_state)

execute_user_code = ds_star_execute_env.execute_user_code
DSState = ds_star_state.DSState
CodeMode = ds_star_state.CodeMode


# Mock async tool that simulates MCP tool behavior
class MockMCPTool:
    """Mock MCP tool with ainvoke method."""

    def __init__(self, name: str, operation):
        self.name = name
        self.description = f"Mock MCP tool: {name}"
        self.operation = operation

    async def ainvoke(self, a: int, b: int) -> str:
        """Async invoke method like LangChain MCP tools."""
        result = self.operation(a, b)
        return f"Result: {result}"


def test_mcp_async_tools():
    """Test that call_tool can handle async MCP tools."""
    print("Testing MCP async tool support...")
    print("=" * 60)

    # Create mock MCP tools
    tools = {
        "multiply": MockMCPTool("multiply", lambda a, b: a * b),
        "add": MockMCPTool("add", lambda a, b: a + b),
    }

    # Create minimal state
    state = DSState(
        user_query="Test MCP async tools",
        tools={},
        code_mode=CodeMode.STEPWISE,
        steps=[],
    )

    # Test 1: Async MCP tool (multiply)
    print("\n1. Testing async MCP tool (multiply 15 * 23)...")
    code1 = """
result = call_tool("multiply", a=15, b=23)
outputs["multiply_result"] = result
print(f"Multiply result: {result}")
"""
    logs1, outputs1 = execute_user_code(code1, state, tools, timeout=10)
    print(f"Logs: {logs1}")
    print(f"Outputs: {outputs1}")

    if "_error" in outputs1:
        print(f"❌ FAILED: {outputs1['_error']}")
        return False

    if "multiply_result" not in outputs1:
        print("❌ FAILED: No multiply_result in outputs")
        return False

    if "345" not in outputs1["multiply_result"]:
        print(f"❌ FAILED: Expected 345 in result, got: {outputs1['multiply_result']}")
        return False

    print("✅ Async multiply tool works!")

    # Test 2: Async MCP tool (add)
    print("\n2. Testing async MCP tool (add 345 + 42)...")
    code2 = """
result = call_tool("add", a=345, b=42)
outputs["add_result"] = result
print(f"Add result: {result}")
"""
    logs2, outputs2 = execute_user_code(code2, state, tools, timeout=10)
    print(f"Logs: {logs2}")
    print(f"Outputs: {outputs2}")

    if "_error" in outputs2:
        print(f"❌ FAILED: {outputs2['_error']}")
        return False

    if "add_result" not in outputs2:
        print("❌ FAILED: No add_result in outputs")
        return False

    if "387" not in outputs2["add_result"]:
        print(f"❌ FAILED: Expected 387 in result, got: {outputs2['add_result']}")
        return False

    print("✅ Async add tool works!")

    # Test 3: Multiple async tool calls
    print("\n3. Testing multiple async MCP tool calls...")
    code3 = """
r1 = call_tool("multiply", a=15, b=23)
r2 = call_tool("add", a=345, b=42)
outputs["combined"] = f"multiply: {r1}, add: {r2}"
print(f"Combined results: {outputs['combined']}")
"""
    logs3, outputs3 = execute_user_code(code3, state, tools, timeout=10)
    print(f"Logs: {logs3}")
    print(f"Outputs: {outputs3}")

    if "_error" in outputs3:
        print(f"❌ FAILED: {outputs3['_error']}")
        return False

    if "combined" not in outputs3:
        print("❌ FAILED: No combined in outputs")
        return False

    result = outputs3["combined"]
    if "345" not in result or "387" not in result:
        print(f"❌ FAILED: Expected both 345 and 387 in result, got: {result}")
        return False

    print("✅ Multiple async tool calls work!")

    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED! MCP async tool support is working!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_mcp_async_tools()
    sys.exit(0 if success else 1)
