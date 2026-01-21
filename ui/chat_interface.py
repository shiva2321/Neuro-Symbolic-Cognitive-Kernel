"""
NCGN Chat Interface - Interactive Query Terminal

Provides a terminal-based chat interface for interacting
with the trained NCGN system.

Features:
- Natural language questions
- Command-based controls
- State visualization (text-based)
- Session history
- Dialogue state management (Phase 3)

Usage:
    python -m ui.chat_interface
"""

import os
import sys
from typing import Optional, List

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ChatInterface:
    """
    Interactive terminal interface for NCGN queries.
    
    Now integrates with DialogueManager for stateful conversation.
    
    Commands:
        :help       - Show help
        :state      - Show current state
        :inject X   - Inject energy into node X
        :step       - Run one tick
        :run N      - Run N ticks
        :train      - Run training session
        :visualize  - Open web dashboard
        :teach      - Switch to teaching mode
        :read FILE  - Read and stage file content
        :quit       - Exit
    """
    
    HELP_TEXT = """
╭──────────────────────────────────────────────────────────────╮
│          NCGN Chat Interface (Phase 3)               │
├──────────────────────────────────────────────────────────────┤
│  Teach knowledge:                                     │
│    • Dogs eat meat.                                   │
│    • Birds fly.                                       │
│    • Penguins are birds.                              │
│                                                       │
│  Ask questions:                                       │
│    • What does dog eat?                               │
│    • Is metal edible?                                 │
│    • Why can't dog eat metal?                         │
│                                                       │
│  Commands:                                            │
│    :help          Show this help                      │
│    :state         Show current graph state            │
│    :inject <node> Inject energy into a node           │
│    :step          Run one tick                        │
│    :run <n>       Run n ticks                         │
│    :nodes         List all nodes                      │
│    :read <file>   Read and stage file content         │
│    :dialogue      Show dialogue state                 │
│    :clear         Clear conversation                  │
│    :quit          Exit                                │
╰──────────────────────────────────────────────────────────────╯
"""
    
    def __init__(self, memory=None, engine=None, controller=None, query_engine=None):
        self.memory = memory
        self.engine = engine
        self.controller = controller
        self.query_engine = query_engine
        
        self.conversation: List[tuple] = []
        self.running = False
        
        # Phase 3: DialogueManager for stateful conversation
        self.dialogue_manager = None
    
    def setup_demo_system(self):
        """Set up a demo system if none provided."""
        from core.memory import GraphMemory
        from core.system1 import System1Engine
        from core.system2 import System2Controller
        from core.query_engine import QueryEngine
        
        self.memory = GraphMemory()
        self.engine = System1Engine(self.memory, k_winners=10)
        self.controller = System2Controller(self.memory)
        
        # Add demo knowledge
        animals = ["dog", "cat", "rabbit", "bird", "cow"]
        foods = ["meat", "fish", "carrot", "seed", "grass"]
        
        for animal in animals:
            self.memory.add_node(animal, threshold=0.5)
        
        for food in foods:
            self.memory.add_node(food, threshold=0.5, novelty_score=0.1)
        
        # Add edible/inedible property nodes
        self.memory.add_node("edible", threshold=0.5)
        self.memory.add_node("inedible", threshold=0.5)
        
        # Associations
        pairs = [("dog", "meat"), ("cat", "fish"), ("rabbit", "carrot"), 
                 ("bird", "seed"), ("cow", "grass")]
        
        for animal, food in pairs:
            self.memory.add_synapse(animal, food, type="eats", weight=0.9, confidence=0.9)
        
        # Food -> edible
        for food in foods:
            self.memory.add_synapse(food, "edible", type="is", weight=0.85, confidence=0.9)
            self.controller.set_property(food, "is_edible", True)
        
        # Inedible items
        inedibles = ["metal", "plastic", "rock", "glass"]
        for item in inedibles:
            self.memory.add_node(item, threshold=0.5, novelty_score=0.1)
            self.memory.add_synapse(item, "inedible", type="is", weight=0.9, confidence=0.95)
            self.controller.set_property(item, "is_edible", False)
        
        # Set up schema
        from core.memory import EventSchema
        schema = EventSchema.from_dict({
            "id": "schema_eat",
            "action": "eat",
            "confidence": 0.95,
            "roles": {"agent": "animate_object", "target": "edible_object"},
            "constraints": {"target": ["is_edible"]}
        })
        self.controller.add_schema(schema)
        
        # Animate properties
        for animal in animals:
            self.controller.set_property(animal, "is_animate", True)
        
        # Create query engine
        self.query_engine = QueryEngine(self.memory, self.controller)
        
        # Phase 3: Initialize DialogueManager
        try:
            from cortex.dialogue import DialogueManager
            self.dialogue_manager = DialogueManager(
                memory=self.memory,
                engine=self.engine,
                controller=self.controller,
                query_engine=self.query_engine
            )
        except ImportError:
            self.dialogue_manager = None
        
        print("✓ Demo system initialized with animal-food knowledge")
        if self.dialogue_manager:
            print("✓ DialogueManager enabled (Phase 3)")
    
    def print_header(self):
        """Print welcome header."""
        print("\n" + "═" * 60)
        print("   🧠 NCGN - Neuromorphic Cognitive Graph Network")
        print("       Interactive Query Interface")
        print("═" * 60)
        print("   Type ':help' for commands or ask a question")
        print("═" * 60 + "\n")
    
    def format_result(self, result) -> str:
        """Format a query result for display."""
        lines = []
        
        if result.success:
            lines.append(f"\n💡 {result.answer}")
            
            if result.confidence > 0:
                conf_bar = "█" * int(result.confidence * 10) + "░" * (10 - int(result.confidence * 10))
                lines.append(f"   Confidence: [{conf_bar}] {result.confidence:.1%}")
            
            if result.evidence:
                lines.append("\n   Evidence:")
                for ev in result.evidence[:3]:
                    lines.append(f"   • {ev}")
            
            if result.reasoning_path:
                lines.append("\n   Reasoning path:")
                for src, rel, tgt in result.reasoning_path[:5]:
                    lines.append(f"   {src} -[{rel}]→ {tgt}")
        else:
            lines.append(f"\n❌ {result.answer}")
        
        return "\n".join(lines)
    
    def handle_command(self, cmd: str) -> Optional[str]:
        """Handle a command (starting with :)."""
        parts = cmd[1:].strip().split(maxsplit=1)
        command = parts[0].lower() if parts else ""
        args = parts[1] if len(parts) > 1 else ""
        
        if command == "help":
            return self.HELP_TEXT
        
        elif command == "state":
            active = self.memory.get_active_nodes()
            if not active:
                return "No active nodes."
            
            lines = [f"\n📊 Current State (Tick {self.engine.current_tick})"]
            lines.append(f"   Surprise: {self.engine.surprise_level:.3f}")
            lines.append(f"   System 2: {'Triggered' if self.engine.system2_triggered else 'Idle'}")
            lines.append(f"\n   Active nodes ({len(active)}):")
            
            for node_id in sorted(active):
                node = self.memory.get_node(node_id)
                energy_bar = "█" * int(node.energy * 10)
                lines.append(f"   {node_id:12} [{energy_bar:10}] E={node.energy:.3f}")
            
            return "\n".join(lines)
        
        elif command == "nodes":
            total = self.memory.node_count()
            nodes = sorted(self.memory.nodes.keys())
            
            lines = [f"\n📋 All Nodes ({total} total)"]
            for i in range(0, len(nodes), 5):
                chunk = nodes[i:i+5]
                lines.append("   " + ", ".join(chunk))
            
            return "\n".join(lines)
        
        elif command == "inject":
            if not args:
                return "Usage: :inject <node_id> [energy]"
            
            parts = args.split()
            node_id = parts[0]
            energy = float(parts[1]) if len(parts) > 1 else 1.0
            
            self.engine.inject_energy(node_id, energy)
            return f"✓ Injected {energy} energy into '{node_id}'"
        
        elif command == "step":
            completed = self.engine.tick()
            return f"✓ Tick {self.engine.current_tick} completed" if completed else "System paused"
        
        elif command == "run":
            n = int(args) if args else 10
            completed = self.engine.run(n)
            return f"✓ Ran {completed} ticks (now at tick {self.engine.current_tick})"
        
        elif command == "related":
            if not args:
                return "Usage: :related <node_id>"
            
            related = self.query_engine.get_related(args.strip(), max_depth=2)
            if not related:
                return f"No nodes related to '{args}'"
            
            lines = [f"\n🔗 Nodes related to '{args}':"]
            for node_id, score in list(related.items())[:10]:
                score_bar = "█" * int(score * 10)
                lines.append(f"   {node_id:12} [{score_bar:10}] {score:.2f}")
            
            return "\n".join(lines)
        
        elif command == "dialogue":
            if self.dialogue_manager:
                state = self.dialogue_manager.get_state_indicator()
                ctx = self.dialogue_manager.context
                lines = [
                    f"\n📍 Dialogue State: {state}",
                    f"   Clarifications asked: {ctx.clarifications_asked}",
                    f"   Exceptions learned: {ctx.exceptions_learned}",
                ]
                if ctx.pending_triple:
                    lines.append(f"   Pending: {ctx.pending_triple}")
                if ctx.staging_buffer:
                    lines.append(f"   Staging: {ctx.staging_buffer}")
                return "\n".join(lines)
            else:
                return "DialogueManager not available"
        
        elif command == "read":
            if not args:
                return "Usage: :read <filename>"
            if self.dialogue_manager:
                return self.dialogue_manager._start_file_ingestion(args)
            else:
                return "DialogueManager not available for file ingestion"
        
        elif command == "clear":
            self.conversation.clear()
            if self.dialogue_manager:
                self.dialogue_manager.context.history.clear()
            return "✓ Conversation cleared"
        
        elif command in ("quit", "exit", "q"):
            self.running = False
            return "Goodbye! 👋"
        
        else:
            return f"Unknown command: {command}. Type :help for commands."
    
    def run(self):
        """Run the interactive chat loop with DialogueManager integration."""
        if not self.memory:
            self.setup_demo_system()
        
        self.print_header()
        self.running = True
        
        while self.running:
            try:
                # Show state indicator if DialogueManager enabled
                if self.dialogue_manager:
                    state_indicator = self.dialogue_manager.get_state_indicator()
                    prompt = f"[{state_indicator}] NCGN> "
                else:
                    prompt = "NCGN> "
                
                user_input = input(prompt).strip()
                
                if not user_input:
                    continue
                
                if user_input.startswith(":"):
                    response = self.handle_command(user_input)
                elif self.dialogue_manager:
                    # Use DialogueManager for stateful processing
                    response, new_state = self.dialogue_manager.process_input(user_input)
                    self.conversation.append((user_input, response))
                else:
                    # Fallback to query engine
                    result = self.query_engine.ask(user_input)
                    response = self.format_result(result)
                    self.conversation.append((user_input, result))
                
                print(response)
                print()
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except EOFError:
                break
            except Exception as e:
                print(f"\nError: {e}")


def main():
    """Run the chat interface."""
    chat = ChatInterface()
    chat.run()


if __name__ == "__main__":
    main()
