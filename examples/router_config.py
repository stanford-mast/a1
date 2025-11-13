"""
Router Configuration Agent Example

Demonstrates using a1 for complex configuration generation with:
1. Many tools with large enum parameters (interface names, VLANs, etc.)
2. Ordering constraints between commands (dependencies)
3. SemanticGenerate for tool filtering and enum reduction
4. OrderingVerify for dependency checking

Use Case: Generate Cisco router configurations with proper command ordering.
"""

import asyncio
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from a1 import (
    Agent,
    LLM,
    Done,
    Runtime,
    Strategy,
    tool,
)
from a1.extra_strategies import ReduceAndGenerate, CheckOrdering
from a1 import EM


# ==================== Configuration Models ====================

class InterfaceType(str, Enum):
    """Types of network interfaces (simulating large enum)"""
    GIGABIT_ETHERNET = "GigabitEthernet"
    TEN_GIGABIT_ETHERNET = "TenGigabitEthernet"
    FAST_ETHERNET = "FastEthernet"
    SERIAL = "Serial"
    LOOPBACK = "Loopback"


# Simulate large enum: In real world, could have 100+ interfaces
# For demo, we'll show the pattern with just a few
class InterfaceName(str, Enum):
    """Interface names - in production could be 100-1000 values"""
    GIGABIT_0_0 = "GigabitEthernet0/0"
    GIGABIT_0_1 = "GigabitEthernet0/1"
    GIGABIT_0_2 = "GigabitEthernet0/2"
    GIGABIT_1_0 = "GigabitEthernet1/0"
    GIGABIT_1_1 = "GigabitEthernet1/1"
    TEN_GIG_0_0 = "TenGigabitEthernet0/0"
    TEN_GIG_0_1 = "TenGigabitEthernet0/1"
    FAST_ETH_0_0 = "FastEthernet0/0"
    FAST_ETH_0_1 = "FastEthernet0/1"
    LOOPBACK_0 = "Loopback0"
    # ... imagine 90+ more entries


class VLANId(int, Enum):
    """VLAN IDs - in production could be 1-4096"""
    VLAN_1 = 1
    VLAN_10 = 10
    VLAN_20 = 20
    VLAN_30 = 30
    VLAN_100 = 100
    VLAN_200 = 200
    # ... imagine 100+ more entries


class InterfaceConfig(BaseModel):
    """Configuration result for an interface"""
    interface: str
    status: str
    ip_address: str | None = None
    vlan_id: int | None = None


class VLANConfig(BaseModel):
    """Configuration result for a VLAN"""
    vlan_id: int
    name: str
    status: str


class RouterConfigOutput(BaseModel):
    """Final router configuration"""
    interfaces: list[InterfaceConfig]
    vlans: list[VLANConfig]
    config_summary: str


class ConfigRequest(BaseModel):
    """User's configuration request"""
    task: str = Field(description="Natural language description of configuration needed")


# ==================== Router Configuration Tools ====================
# Each tool represents a router command
# These have DEPENDENCIES - some must be called before others

@tool(
    name="enable_interface",
    description="Enable a network interface. MUST be called before configuring IP or VLAN on that interface."
)
async def enable_interface(interface: InterfaceName) -> InterfaceConfig:
    """Enable an interface - prerequisite for all other interface commands"""
    return InterfaceConfig(
        interface=interface.value,
        status="enabled"
    )


@tool(
    name="set_interface_ip",
    description="Set IP address on an interface. Requires interface to be enabled first."
)
async def set_interface_ip(
    interface: InterfaceName,
    ip_address: str,
    subnet_mask: str = "255.255.255.0"
) -> InterfaceConfig:
    """Configure IP address - depends on enable_interface"""
    return InterfaceConfig(
        interface=interface.value,
        status="configured",
        ip_address=f"{ip_address}/{subnet_mask}"
    )


@tool(
    name="create_vlan",
    description="Create a new VLAN. MUST be called before assigning ports to this VLAN."
)
async def create_vlan(vlan_id: VLANId, name: str) -> VLANConfig:
    """Create VLAN - prerequisite for assign_interface_to_vlan"""
    return VLANConfig(
        vlan_id=vlan_id.value,
        name=name,
        status="active"
    )


@tool(
    name="assign_interface_to_vlan",
    description="Assign an interface to a VLAN. Requires VLAN to exist and interface to be enabled."
)
async def assign_interface_to_vlan(
    interface: InterfaceName,
    vlan_id: VLANId
) -> InterfaceConfig:
    """Assign interface to VLAN - depends on create_vlan and enable_interface"""
    return InterfaceConfig(
        interface=interface.value,
        status="assigned",
        vlan_id=vlan_id.value
    )


@tool(
    name="configure_interface_speed",
    description="Configure interface speed. Requires interface to be enabled first."
)
async def configure_interface_speed(
    interface: InterfaceName,
    speed: Literal["10", "100", "1000", "10000"]
) -> InterfaceConfig:
    """Set interface speed - depends on enable_interface"""
    return InterfaceConfig(
        interface=interface.value,
        status=f"speed_{speed}"
    )


@tool(
    name="configure_interface_duplex",
    description="Configure interface duplex mode. Requires interface to be enabled first."
)
async def configure_interface_duplex(
    interface: InterfaceName,
    duplex: Literal["half", "full", "auto"]
) -> InterfaceConfig:
    """Set duplex mode - depends on enable_interface"""
    return InterfaceConfig(
        interface=interface.value,
        status=f"duplex_{duplex}"
    )


# ==================== Main Example ====================

async def main():
    """
    Demonstrate router configuration with:
    - Multiple tools with large enums
    - Dependency constraints between commands
    - Semantic filtering of tools
    - CFG-based ordering verification
    """
    
    print("=" * 80)
    print("Router Configuration Agent with Ordering Constraints")
    print("=" * 80)
    
    # Define ordering rules: (prerequisite, dependent)
    # These encode the dependency graph
    ordering_rules = [
        ("enable_interface", "set_interface_ip"),
        ("enable_interface", "assign_interface_to_vlan"),
        ("enable_interface", "configure_interface_speed"),
        ("enable_interface", "configure_interface_duplex"),
        ("create_vlan", "assign_interface_to_vlan"),
    ]
    
    print("\nOrdering Rules (Dependencies):")
    for prereq, dependent in ordering_rules:
        print(f"  • {prereq} → {dependent}")
    
    # Create agent with ReduceAndGenerate and CheckOrdering
    agent = Agent(
        name="router_config_agent",
        description="Generate Cisco router configurations with proper command ordering",
        input_schema=ConfigRequest,
        output_schema=RouterConfigOutput,
        tools=[
            enable_interface,
            set_interface_ip,
            create_vlan,
            assign_interface_to_vlan,
            configure_interface_speed,
            configure_interface_duplex,
            LLM("gpt-4.1-mini"),
            Done(),
        ],
    )
    
    # Use custom strategy with semantic filtering and ordering verification
    strategy = Strategy(
        generate=ReduceAndGenerate(
            em_tool=EM(),
            llm_tool=LLM("gpt-4.1-mini"),
            max_tools=50,  # Filter to top 50 most relevant tools
            max_enum_size=100,  # Reduce enums to top 100 values
        ),
        verify=[
            CheckOrdering(rules=ordering_rules),  # Check dependencies
        ],
        max_iterations=5,
    )
    
    # Create runtime
    runtime = Runtime(strategy=strategy)
    
    # Test case: Configuration that requires proper ordering
    print("\n" + "=" * 80)
    print("Test Case: Configure VLAN 10 on GigabitEthernet0/0")
    print("=" * 80)
    print("\nExpected call sequence:")
    print("  1. enable_interface(GigabitEthernet0/0)")
    print("  2. create_vlan(10, 'Engineering')")
    print("  3. assign_interface_to_vlan(GigabitEthernet0/0, 10)")
    
    # This should work - proper ordering
    try:
        result = await runtime.jit(
            agent,
            task="Configure VLAN 10 named 'Engineering' on interface GigabitEthernet0/0"
        )
        print("\n✅ Configuration successful!")
        print(f"\nResult: {result}")
    except Exception as e:
        import traceback
        print(f"\n❌ Configuration failed: {e}")
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Test Case: Configure IP address on interface")
    print("=" * 80)
    print("\nExpected call sequence:")
    print("  1. enable_interface(GigabitEthernet0/1)")
    print("  2. set_interface_ip(GigabitEthernet0/1, '192.168.1.1')")
    
    # This should also work
    try:
        result = await runtime.jit(
            agent,
            task="Set IP address 192.168.1.1 on interface GigabitEthernet0/1"
        )
        print("\n✅ Configuration successful!")
        print(f"\nResult: {result}")
    except Exception as e:
        print(f"\n❌ Configuration failed: {e}")
    
    print("\n" + "=" * 80)
    print("Example complete!")
    print("=" * 80)
    print("\nKey Features Demonstrated:")
    print("  ✓ Large enums in tool parameters (InterfaceName, VLANId)")
    print("  ✓ Semantic filtering to reduce tool set size")
    print("  ✓ Dependency constraints between commands")
    print("  ✓ CFG-based ordering verification")
    print("  ✓ Automatic enum reduction using embeddings")


if __name__ == "__main__":
    asyncio.run(main())
