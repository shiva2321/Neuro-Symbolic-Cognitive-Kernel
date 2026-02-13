"""
NSCK Module Registry
====================

Dynamic module discovery and registration system for external cognitive modules.
Allows developers to build plugins without modifying CognitiveEngine.

Part of NSCK V2 substrate transformation (Task 9: Module SDK).
"""

import importlib
import importlib.util
import inspect
import sys
from typing import List, Dict, Type, Optional, Any
from pathlib import Path

from python.core.reasoning.global_workspace import WorkspaceModule


class ModuleRegistry:
    """
    Registry for discovering and instantiating external WorkspaceModules.
    
    Supports:
    - Automatic discovery from directories
    - Manual module registration
    - Module validation (WorkspaceModule compliance)
    - Instantiation with dependency injection
    
    Usage:
        registry = ModuleRegistry()
        registry.discover_modules("./custom_modules")
        
        # Register with CognitiveEngine
        for module_class in registry.get_modules():
            instance = registry.instantiate(module_class, brain_store=brain)
            engine.register_module(instance)
    """
    
    def __init__(self):
        self._modules: Dict[str, Type[WorkspaceModule]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
    
    def register(self, module_class: Type[WorkspaceModule], 
                 name: Optional[str] = None, 
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Manually register a module class.
        
        Args:
            module_class: WorkspaceModule subclass
            name: Optional module name (defaults to class name)
            metadata: Optional metadata dict (author, version, etc.)
            
        Raises:
            TypeError: If module_class is not a WorkspaceModule subclass
            
        Example:
            registry.register(
                MyCustomModule,
                metadata={"author": "John Doe", "version": "1.0"}
            )
        """
        if not self._is_valid_module(module_class):
            raise TypeError(
                f"{module_class.__name__} must be a subclass of WorkspaceModule "
                f"with all required methods implemented"
            )
        
        module_name = name or module_class.__name__
        
        # Auto-extract metadata if not provided
        if metadata is None:
            metadata = {}
        
        # Add docstring
        if "docstring" not in metadata:
            metadata["docstring"] = module_class.__doc__ or ""
        
        # Add author/version if available
        if hasattr(module_class, "__author__") and "author" not in metadata:
            metadata["author"] = module_class.__author__
        if hasattr(module_class, "__version__") and "version" not in metadata:
            metadata["version"] = module_class.__version__
        
        self._modules[module_name] = module_class
        self._metadata[module_name] = metadata
        
        print(f"✅ Registered module: {module_name}")
    
    def discover_modules(self, directory: str, recursive: bool = True):
        """
        Automatically discover modules in a directory.
        
        Scans Python files for WorkspaceModule subclasses and registers them.
        
        Args:
            directory: Path to directory containing module files
            recursive: If True, scan subdirectories recursively
            
        Example:
            registry.discover_modules("./plugins")
            registry.discover_modules("./custom_modules", recursive=False)
        """
        directory = Path(directory)
        if not directory.exists():
            print(f"⚠️  Directory not found: {directory}")
            return
        
        pattern = "**/*.py" if recursive else "*.py"
        python_files = list(directory.glob(pattern))
        
        discovered_count = 0
        for filepath in python_files:
            if filepath.name.startswith("_") or filepath.name.startswith("test_"):
                continue  # Skip private files and tests
            
            try:
                modules_found = self._load_modules_from_file(filepath)
                discovered_count += modules_found
            except Exception as e:
                print(f"⚠️  Error loading {filepath.name}: {e}")
        
        print(f"✅ Discovered {discovered_count} modules from {directory}")
    
    def _load_modules_from_file(self, filepath: Path) -> int:
        """Load all WorkspaceModule subclasses from a Python file."""
        # Create module spec and load
        module_name = filepath.stem
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        if spec is None or spec.loader is None:
            return 0
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        # Scan for WorkspaceModule subclasses
        count = 0
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (obj != WorkspaceModule and 
                issubclass(obj, WorkspaceModule) and 
                self._is_valid_module(obj)):
                
                # Extract metadata from docstring or class attributes
                metadata = {
                    "source_file": str(filepath),
                    "docstring": obj.__doc__ or ""
                }
                
                if hasattr(obj, "__author__"):
                    metadata["author"] = obj.__author__
                if hasattr(obj, "__version__"):
                    metadata["version"] = obj.__version__
                
                self.register(obj, name=name, metadata=metadata)
                count += 1
        
        return count
    
    def _is_valid_module(self, module_class: Type) -> bool:
        """
        Check if a class is a valid WorkspaceModule.
        
        Validates:
        - Is a subclass of WorkspaceModule
        - Implements all required abstract methods
        """
        if not inspect.isclass(module_class):
            return False
        
        if not issubclass(module_class, WorkspaceModule):
            return False
        
        # Check that it's not the abstract base class itself
        if module_class is WorkspaceModule:
            return False
        
        # Check that all abstract methods are implemented
        required_methods = ["propose", "update", "receive_broadcast", "get_telemetry"]
        for method_name in required_methods:
            if not hasattr(module_class, method_name):
                return False
            
            method = getattr(module_class, method_name)
            if getattr(method, "__isabstractmethod__", False):
                return False  # Method is still abstract
        
        return True
    
    def get_modules(self) -> List[Type[WorkspaceModule]]:
        """Get all registered module classes."""
        return list(self._modules.values())
    
    def get_module(self, name: str) -> Optional[Type[WorkspaceModule]]:
        """Get a specific module class by name."""
        return self._modules.get(name)
    
    def list_modules(self) -> List[Dict[str, Any]]:
        """
        List all registered modules with their metadata.
        
        Returns:
            List of dicts with keys: name, class, metadata
        """
        return [
            {
                "name": name,
                "class": module_class,
                "metadata": self._metadata.get(name, {})
            }
            for name, module_class in self._modules.items()
        ]
    
    def instantiate(self, module_class: Type[WorkspaceModule], **kwargs) -> WorkspaceModule:
        """
        Instantiate a module with dependency injection.
        
        Inspects the module's __init__ signature and provides matching kwargs.
        
        Args:
            module_class: WorkspaceModule class to instantiate
            **kwargs: Dependencies to inject (e.g., brain_store=..., config=...)
            
        Returns:
            Instantiated module
            
        Example:
            brain = BrainStore("brain.db")
            module = registry.instantiate(
                SecurityMonitor,
                brain_store=brain,
                threshold=0.8
            )
        """
        # Get __init__ signature
        sig = inspect.signature(module_class.__init__)
        
        # Match kwargs to parameters
        init_kwargs = {}
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            
            if param_name in kwargs:
                init_kwargs[param_name] = kwargs[param_name]
            elif param.default is not inspect.Parameter.empty:
                # Has default value, don't need to provide
                pass
            else:
                # Required parameter not provided
                print(f"⚠️  Warning: Required parameter '{param_name}' not provided for {module_class.__name__}")
        
        return module_class(**init_kwargs)
    
    def clear(self):
        """Clear all registered modules."""
        self._modules.clear()
        self._metadata.clear()


# Singleton instance for global access
_registry = ModuleRegistry()


def get_registry() -> ModuleRegistry:
    """Get the global module registry instance."""
    return _registry
