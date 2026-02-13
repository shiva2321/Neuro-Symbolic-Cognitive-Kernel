"""
Tests for ModuleRegistry - Module Discovery and Registration System
====================================================================

Tests the automatic module discovery, validation, and instantiation system
for external cognitive modules.

Part of Task 9: Module SDK Package (NSCK V2 Substrate).
"""

import pytest
import sys
from pathlib import Path
import tempfile
import shutil

# Add parent directories to path

from python.module_registry import ModuleRegistry, get_registry
from python.core.reasoning.global_workspace import WorkspaceModule, Coalition
import numpy as np
from typing import Optional, Dict, Any


# Test fixtures - Valid and invalid modules

class ValidModule1(WorkspaceModule):
    """A valid test module."""
    __author__ = "Test Suite"
    __version__ = "1.0"
    
    def propose(self, state_hv: np.ndarray) -> Optional[Coalition]:
        return None
    
    def update(self, feedback_hv: np.ndarray, reward: float, info: Dict):
        pass
    
    def receive_broadcast(self, content: Any):
        pass
    
    def get_telemetry(self) -> Dict[str, Any]:
        return {"test": True}


class ValidModule2(WorkspaceModule):
    """Another valid module with constructor parameters."""
    
    def __init__(self, threshold: float = 0.5, name: str = "module2"):
        self.threshold = threshold
        self.name = name
    
    def propose(self, state_hv: np.ndarray) -> Optional[Coalition]:
        return None
    
    def update(self, feedback_hv: np.ndarray, reward: float, info: Dict):
        pass
    
    def receive_broadcast(self, content: Any):
        pass
    
    def get_telemetry(self) -> Dict[str, Any]:
        return {"threshold": self.threshold, "name": self.name}


class InvalidModule:
    """Invalid - doesn't inherit from WorkspaceModule."""
    
    def propose(self, state_hv: np.ndarray):
        return None


# Test: Manual Registration

def test_register_valid_module():
    """Test registering a valid module manually."""
    registry = ModuleRegistry()
    registry.register(ValidModule1)
    
    modules = registry.get_modules()
    assert ValidModule1 in modules
    assert len(modules) == 1


def test_register_with_metadata():
    """Test registering with custom metadata."""
    registry = ModuleRegistry()
    registry.register(
        ValidModule1,
        name="CustomName",
        metadata={"author": "TestSuite", "version": "2.0"}
    )
    
    module_info = registry.list_modules()[0]
    assert module_info["name"] == "CustomName"
    assert module_info["metadata"]["author"] == "TestSuite"


def test_register_invalid_module_raises_error():
    """Test that registering invalid module raises TypeError."""
    registry = ModuleRegistry()
    
    with pytest.raises(TypeError):
        registry.register(InvalidModule)


def test_register_duplicate_module():
    """Test registering same module multiple times."""
    registry = ModuleRegistry()
    registry.register(ValidModule1, name="module1")
    registry.register(ValidModule1, name="module1_duplicate")
    
    modules = registry.list_modules()
    assert len(modules) == 2


# Test: Module Discovery

def test_discover_modules_from_directory():
    """Test automatic module discovery from directory (handles import errors gracefully)."""
    # Create temporary directory with module file (will fail to import but shouldn't crash)
    with tempfile.TemporaryDirectory() as tmpdir:
        module_file = Path(tmpdir) / "test_module.py"
        module_file.write_text("""
# This module will fail to import due to missing dependencies
from non_existent_module import Something

class TestClass:
    pass
""")
        
        registry = ModuleRegistry()
        # Should not crash even though import fails
        registry.discover_modules(tmpdir, recursive=False)
        
        # Should have 0 modules (import failed)
        modules = registry.get_modules()
        assert len(modules) == 0
        # But it shouldn't have crashed!


def test_discover_skips_private_files():
    """Test that discovery skips files starting with underscore."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create files
        (Path(tmpdir) / "_private.py").write_text("# Private file")
        (Path(tmpdir) / "test_module.py").write_text("# Test file")
        
        registry = ModuleRegistry()
        registry.discover_modules(tmpdir)
        
        # Should skip both (no valid modules)
        assert len(registry.get_modules()) == 0


def test_discover_nonexistent_directory():
    """Test discovery on nonexistent directory."""
    registry = ModuleRegistry()
    registry.discover_modules("/nonexistent/path")
    
    # Should not raise error, just warn
    assert len(registry.get_modules()) == 0


# Test: Module Validation

def test_validation_rejects_non_class():
    """Test that validation rejects non-class objects."""
    registry = ModuleRegistry()
    
    assert not registry._is_valid_module("not_a_class")
    assert not registry._is_valid_module(123)
    assert not registry._is_valid_module(None)


def test_validation_rejects_base_class():
    """Test that validation rejects WorkspaceModule itself."""
    registry = ModuleRegistry()
    
    assert not registry._is_valid_module(WorkspaceModule)


def test_validation_accepts_valid_subclass():
    """Test that validation accepts proper subclass."""
    registry = ModuleRegistry()
    
    assert registry._is_valid_module(ValidModule1)
    assert registry._is_valid_module(ValidModule2)


# Test: Module Instantiation

def test_instantiate_without_parameters():
    """Test instantiating module with no __init__ parameters."""
    registry = ModuleRegistry()
    registry.register(ValidModule1)
    
    instance = registry.instantiate(ValidModule1)
    
    assert isinstance(instance, ValidModule1)
    assert hasattr(instance, "propose")


def test_instantiate_with_parameters():
    """Test instantiating module with dependency injection."""
    registry = ModuleRegistry()
    registry.register(ValidModule2)
    
    instance = registry.instantiate(
        ValidModule2,
        threshold=0.8,
        name="custom_name"
    )
    
    assert isinstance(instance, ValidModule2)
    assert instance.threshold == 0.8
    assert instance.name == "custom_name"


def test_instantiate_with_partial_parameters():
    """Test instantiation with some parameters missing (use defaults)."""
    registry = ModuleRegistry()
    registry.register(ValidModule2)
    
    instance = registry.instantiate(ValidModule2, threshold=0.9)
    
    assert instance.threshold == 0.9
    assert instance.name == "module2"  # Default value


def test_instantiate_warns_on_missing_required_parameter():
    """Test that instantiation warns when required parameter missing."""
    class ModuleWithRequired(WorkspaceModule):
        def __init__(self, required_param):
            self.required_param = required_param
        
        def propose(self, state_hv: np.ndarray) -> Optional[Coalition]:
            return None
        def update(self, feedback_hv: np.ndarray, reward: float, info: Dict):
            pass
        def receive_broadcast(self, content: Any):
            pass
        def get_telemetry(self) -> Dict[str, Any]:
            return {}
    
    registry = ModuleRegistry()
    registry.register(ModuleWithRequired)
    
    # Should warn but still attempt instantiation
    # (which will fail with TypeError from Python)
    with pytest.raises(TypeError):
        registry.instantiate(ModuleWithRequired)


# Test: Module Queries

def test_get_module_by_name():
    """Test retrieving specific module by name."""
    registry = ModuleRegistry()
    registry.register(ValidModule1, name="test_module")
    
    module_class = registry.get_module("test_module")
    
    assert module_class == ValidModule1


def test_get_module_nonexistent_name():
    """Test querying nonexistent module returns None."""
    registry = ModuleRegistry()
    
    module_class = registry.get_module("nonexistent")
    
    assert module_class is None


def test_list_modules_with_metadata():
    """Test listing modules includes metadata."""
    registry = ModuleRegistry()
    registry.register(ValidModule1, metadata={"test": "data"})
    registry.register(ValidModule2, metadata={"test": "data2"})
    
    modules_list = registry.list_modules()
    
    assert len(modules_list) == 2
    assert all("name" in m for m in modules_list)
    assert all("class" in m for m in modules_list)
    assert all("metadata" in m for m in modules_list)


# Test: Clear Registry

def test_clear_removes_all_modules():
    """Test that clear() removes all registered modules."""
    registry = ModuleRegistry()
    registry.register(ValidModule1)
    registry.register(ValidModule2)
    
    assert len(registry.get_modules()) == 2
    
    registry.clear()
    
    assert len(registry.get_modules()) == 0


# Test: Global Registry Singleton

def test_get_registry_returns_same_instance():
    """Test that get_registry() returns singleton."""
    registry1 = get_registry()
    registry2 = get_registry()
    
    assert registry1 is registry2


def test_global_registry_persists_registrations():
    """Test that global registry maintains state."""
    registry = get_registry()
    registry.clear()  # Clear any previous state
    
    registry.register(ValidModule1, name="global_test")
    
    # Get registry again
    registry2 = get_registry()
    
    assert len(registry2.get_modules()) == 1
    assert registry2.get_module("global_test") == ValidModule1


# Test: Integration with SDK Examples

def test_discover_sdk_examples():
    """Test discovering actual SDK example modules."""
    sdk_path = Path(__file__).parent.parent / "nsck_sdk"
    
    if sdk_path.exists():
        registry = ModuleRegistry()
        registry.discover_modules(str(sdk_path), recursive=False)
        
        modules = registry.list_modules()
        module_names = [m["name"] for m in modules]
        
        # Should find our 3 example modules
        assert any("SecurityMonitor" in name for name in module_names)
        assert any("CustomPlanner" in name for name in module_names)
        assert any("DomainExpert" in name for name in module_names)


# Test: Module Metadata Extraction

def test_metadata_extraction_from_docstring():
    """Test automatic metadata extraction from module attributes."""
    registry = ModuleRegistry()
    registry.register(ValidModule1)
    
    modules = registry.list_modules()
    metadata = modules[0]["metadata"]
    
    assert metadata["author"] == "Test Suite"
    assert metadata["version"] == "1.0"
    assert "docstring" in metadata


def test_metadata_extraction_handles_missing_attributes():
    """Test metadata extraction when __author__/__version__ missing."""
    registry = ModuleRegistry()
    registry.register(ValidModule2)
    
    modules = registry.list_modules()
    metadata = modules[0]["metadata"]
    
    # Should have docstring but not author/version
    assert "docstring" in metadata
    assert "author" not in metadata
    assert "version" not in metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
