"""
Test Field constraint validation in BaseVerify.
"""

import ast

import pytest
from pydantic import BaseModel, Field

from a1.codecheck import BaseVerify
from a1.models import Agent, Tool, tool


class TestFieldConstraints:
    """Test compile-time validation of Field constraints on constant inputs."""

    def test_pattern_validation_success(self):
        """Test that valid pattern passes validation."""
        
        # Define tool with pattern constraint
        class ConfigInput(BaseModel):
            interface: str = Field(pattern=r'^Gi\d+/\d+$')
        
        @tool(name="configure", description="Configure interface")
        async def configure(interface: str) -> str:
            return f"Configured {interface}"
        
        # Manually set input schema
        configure.input_schema = ConfigInput
        
        agent = Agent(
            name="test",
            input_schema=BaseModel,
            output_schema=BaseModel,
            tools=[configure],
        )
        
        # Code with valid constant input
        code = '''
result = await configure(interface="Gi1/0")
'''
        
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, agent)
        
        assert is_valid, f"Should pass validation: {error}"
    
    def test_pattern_validation_failure(self):
        """Test that invalid pattern fails validation."""
        
        class ConfigInput(BaseModel):
            interface: str = Field(pattern=r'^Gi\d+/\d+$')
        
        @tool(name="configure", description="Configure interface")
        async def configure(interface: str) -> str:
            return f"Configured {interface}"
        
        configure.input_schema = ConfigInput
        
        agent = Agent(
            name="test",
            input_schema=BaseModel,
            output_schema=BaseModel,
            tools=[configure],
        )
        
        # Code with INVALID constant input (eth0 doesn't match Gi\d+/\d+)
        code = '''
result = await configure(interface="eth0")
'''
        
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, agent)
        
        assert not is_valid, "Should fail validation for invalid pattern"
        assert "configure" in error
        assert "interface" in error.lower() or "pattern" in error.lower() or "string_pattern_mismatch" in error.lower()
    
    def test_numeric_bounds_validation_success(self):
        """Test that values within bounds pass validation."""
        
        class VlanInput(BaseModel):
            vlan_id: int = Field(ge=1, le=4094)
        
        @tool(name="configure_vlan", description="Configure VLAN")
        async def configure_vlan(vlan_id: int) -> str:
            return f"VLAN {vlan_id}"
        
        configure_vlan.input_schema = VlanInput
        
        agent = Agent(
            name="test",
            input_schema=BaseModel,
            output_schema=BaseModel,
            tools=[configure_vlan],
        )
        
        code = '''
result = await configure_vlan(vlan_id=100)
'''
        
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, agent)
        
        assert is_valid, f"Should pass validation: {error}"
    
    def test_numeric_bounds_validation_failure_high(self):
        """Test that value exceeding maximum fails validation."""
        
        class VlanInput(BaseModel):
            vlan_id: int = Field(ge=1, le=4094)
        
        @tool(name="configure_vlan", description="Configure VLAN")
        async def configure_vlan(vlan_id: int) -> str:
            return f"VLAN {vlan_id}"
        
        configure_vlan.input_schema = VlanInput
        
        agent = Agent(
            name="test",
            input_schema=BaseModel,
            output_schema=BaseModel,
            tools=[configure_vlan],
        )
        
        # Value exceeds maximum
        code = '''
result = await configure_vlan(vlan_id=5000)
'''
        
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, agent)
        
        assert not is_valid, "Should fail validation for value exceeding maximum"
        assert "configure_vlan" in error
    
    def test_numeric_bounds_validation_failure_low(self):
        """Test that value below minimum fails validation."""
        
        class VlanInput(BaseModel):
            vlan_id: int = Field(ge=1, le=4094)
        
        @tool(name="configure_vlan", description="Configure VLAN")
        async def configure_vlan(vlan_id: int) -> str:
            return f"VLAN {vlan_id}"
        
        configure_vlan.input_schema = VlanInput
        
        agent = Agent(
            name="test",
            input_schema=BaseModel,
            output_schema=BaseModel,
            tools=[configure_vlan],
        )
        
        code = '''
result = await configure_vlan(vlan_id=0)
'''
        
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, agent)
        
        assert not is_valid, "Should fail validation for value below minimum"
        assert "configure_vlan" in error
    
    def test_multiple_constraints(self):
        """Test validation with multiple Field constraints."""
        
        class RouterInput(BaseModel):
            interface: str = Field(pattern=r'^Gi\d+/\d+$')
            vlan: int = Field(ge=1, le=4094)
            description: str = Field(min_length=5, max_length=100)
        
        @tool(name="configure_router", description="Configure router")
        async def configure_router(**kwargs) -> str:
            return "Configured"
        
        configure_router.input_schema = RouterInput
        
        agent = Agent(
            name="test",
            input_schema=BaseModel,
            output_schema=BaseModel,
            tools=[configure_router],
        )
        
        # Valid inputs
        code = '''
result = await configure_router(
    interface="Gi1/0",
    vlan=100,
    description="Primary uplink interface"
)
'''
        
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, agent)
        
        assert is_valid, f"Should pass validation: {error}"
    
    def test_non_constant_inputs_ignored(self):
        """Test that non-constant inputs are not validated (can't check at compile time)."""
        
        class ConfigInput(BaseModel):
            interface: str = Field(pattern=r'^Gi\d+/\d+$')
        
        @tool(name="configure", description="Configure interface")
        async def configure(interface: str) -> str:
            return f"Configured {interface}"
        
        configure.input_schema = ConfigInput
        
        agent = Agent(
            name="test",
            input_schema=BaseModel,
            output_schema=BaseModel,
            tools=[configure],
        )
        
        # Non-constant input (variable)
        code = '''
iface = get_interface()  # Runtime value, can't validate at compile time
result = await configure(interface=iface)
'''
        
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, agent)
        
        # Should pass because we can't validate runtime values
        assert is_valid, f"Should pass for non-constant inputs: {error}"
