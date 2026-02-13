"""
Integration tests for external plugin discovery and execution.

Validates that third-party WorkspaceModules can be discovered, instantiated,
registered with GlobalWorkspace, and participate in competition without core changes.
"""
import sys
from pathlib import Path
import numpy as np

# Add project paths for plugin imports
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from python.module_registry import ModuleRegistry
from global_workspace import GlobalWorkspace, Coalition


def _write_plugin_file(tmp_path: Path) -> Path:
    plugin_code = "\n".join([
        "from global_workspace import WorkspaceModule, Coalition",
        "import numpy as np",
        "",
        "class ExternalPlanner(WorkspaceModule):",
        "    'External plugin planner module.'",
        "    __author__ = 'Plugin Dev'",
        "    __version__ = '0.1'",
        "",
        "    def __init__(self, threshold: float = 0.7, name: str = 'external_planner'):",
        "        self.threshold = threshold",
        "        self.name = name",
        "        self.last_broadcast = None",
        "",
        "    def propose(self, state_hv: np.ndarray):",
        "        if state_hv is None:",
        "            return None",
        "        return Coalition(",
        "            source=self.name,",
        "            content='ACTION_RIGHT',",
        "            base_salience=self.threshold,",
        "            relevance=0.1,",
        "            affect_match=0.0,",
        "            sender_confidence=0.8,",
        "        )",
        "",
        "    def update(self, feedback_hv: np.ndarray, reward: float, info: dict):",
        "        return None",
        "",
        "    def receive_broadcast(self, content):",
        "        self.last_broadcast = content",
        "",
        "    def get_telemetry(self):",
        "        return {",
        "            'active': True,",
        "            'name': self.name,",
        "            'threshold': self.threshold,",
        "        }",
    ])
    plugin_file = tmp_path / "external_planner.py"
    plugin_file.write_text(plugin_code)
    return plugin_file


def test_external_plugin_discovery_and_competition(tmp_path: Path):
    """Verify external plugin can be discovered, instantiated, and compete."""
    _write_plugin_file(tmp_path)

    registry = ModuleRegistry()
    registry.discover_modules(str(tmp_path), recursive=False)

    module_class = registry.get_module("ExternalPlanner")
    assert module_class is not None, "Plugin module should be discovered"

    instance = registry.instantiate(module_class, threshold=0.9, name="external_planner")
    assert instance.threshold == 0.9
    assert instance.name == "external_planner"

    workspace = GlobalWorkspace(attention_threshold=0.2)
    workspace.register_module(instance.name, instance)

    state_hv = np.zeros(8, dtype=np.int8)
    proposal = instance.propose(state_hv)
    assert isinstance(proposal, Coalition)

    winner = workspace.compete([proposal])
    assert winner is not None
    assert workspace.current_winner == "external_planner"
    assert instance.last_broadcast == "ACTION_RIGHT"


def test_external_plugin_metadata(tmp_path: Path):
    """Verify plugin metadata is recorded on registration."""
    _write_plugin_file(tmp_path)

    registry = ModuleRegistry()
    registry.discover_modules(str(tmp_path), recursive=False)

    info = registry.list_modules()
    assert len(info) == 1
    metadata = info[0]["metadata"]

    assert metadata.get("author") == "Plugin Dev"
    assert metadata.get("version") == "0.1"
    assert "External plugin planner module" in (metadata.get("docstring") or "")
