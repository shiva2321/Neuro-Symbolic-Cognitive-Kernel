"""
NCGN Dialogue Module - Conversation State Machine

Manages dialogue flow with proper state tracking, replacing
the stateless chat loop with context-aware interaction.

Key States:
- IDLE: Waiting for user input
- PROCESSING: Running System 1 cycles
- CLARIFICATION_PENDING: System 2 triggered surprise, waiting for explanation
- CONFIRMATION_PENDING: Waiting for user to approve staged content

The Contradiction Loop:
1. User: "Dogs eat metal"
2. System 1 predicts "Meat"
3. Bridge: Surprise > threshold
4. System 2: Schema violation (metal ≠ edible)
5. State → CLARIFICATION_PENDING
6. Output: "My physics say metal isn't edible. Why do you say this?"
7. User: "It's a robot dog"
8. Analysis: "Robot Dog" = subclass proposal
9. System 2 decides: Create (Robot Dog) --[is_a]--> (Dog)
10. Add exception: Eat(Robot Dog, Metal) = True
11. State → IDLE

CRITICAL: User answers are PROPOSALS, not truth. System 2 decides.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Callable, Any
from enum import Enum
import sys
import os

# Add parent path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .ingestion import Triple, RelationType, text_to_triples, DocumentReader
from .staging import StagingBuffer, ConflictInfo, ConflictType, MergeResult


class DialogueState(Enum):
    """States of the dialogue state machine."""
    IDLE = "idle"                              # Waiting for user input
    PROCESSING = "processing"                  # Running System 1 cycles
    CLARIFICATION_PENDING = "clarification"    # Waiting for user explanation
    CONFIRMATION_PENDING = "confirmation"      # Waiting for commit approval
    FILE_REVIEW = "file_review"               # Reviewing staged file content


class ProposalType(Enum):
    """Types of proposals derived from user explanations."""
    NEW_SUBCLASS = "new_subclass"       # "It's a robot dog" -> Robot Dog is_a Dog
    EXCEPTION = "exception"             # "Penguins are an exception"
    NEW_PROPERTY = "new_property"       # "Robot dogs are mechanical"
    NEW_SCHEMA = "new_schema"           # Define new action schema
    REJECTION = "rejection"             # User refuses to clarify
    UNKNOWN = "unknown"                 # Could not parse explanation


@dataclass
class Proposal:
    """
    A proposal derived from user explanation.
    
    User answers become proposals that System 2 evaluates.
    They do NOT automatically become truth.
    """
    proposal_type: ProposalType
    subject: str
    predicate: str
    object: str
    original_triple: Optional[Triple] = None  # What caused the conflict
    explanation: str = ""                      # User's raw explanation
    confidence: float = 0.3                    # Slightly higher than sensor output
    
    def to_triple(self) -> Triple:
        """Convert proposal to a triple for staging."""
        return Triple(
            subject=self.subject,
            predicate=self.predicate,
            object=self.object,
            relation_type=RelationType.IS_A if self.proposal_type == ProposalType.NEW_SUBCLASS else RelationType.ASSOCIATES,
            confidence=self.confidence,
            source_text=self.explanation
        )


@dataclass
class DialogueContext:
    """Current conversation context for state tracking."""
    state: DialogueState = DialogueState.IDLE
    
    # Pending items waiting for user action
    pending_triple: Optional[Triple] = None       # Triple causing clarification
    pending_violation: Optional[str] = None       # Why it was flagged
    pending_diagnosis: Optional[Any] = None       # System2 DiagnosisResult
    
    # Staging buffer for file ingestion
    staging_buffer: Optional[StagingBuffer] = None
    
    # Conversation history
    history: List[Tuple[str, str]] = field(default_factory=list)  # (user, system)
    
    # Statistics
    clarifications_asked: int = 0
    exceptions_learned: int = 0
    
    def add_exchange(self, user_input: str, system_response: str):
        """Record a conversation exchange."""
        self.history.append((user_input, system_response))
        # Keep history bounded
        if len(self.history) > 50:
            self.history = self.history[-50:]
    
    def get_recent_context(self, n: int = 3) -> str:
        """Get recent conversation for context."""
        recent = self.history[-n:]
        lines = []
        for user, system in recent:
            lines.append(f"User: {user}")
            lines.append(f"System: {system}")
        return "\n".join(lines)


class DialogueManager:
    """
    State machine for dialogue flow.
    
    Replaces stateless chat loop with context-aware interaction.
    Handles the Contradiction Loop and file ingestion workflow.
    """
    
    def __init__(
        self,
        memory=None,         # GraphMemory
        engine=None,         # System1Engine
        controller=None,     # System2Controller
        query_engine=None    # QueryEngine
    ):
        self.memory = memory
        self.engine = engine
        self.controller = controller
        self.query_engine = query_engine
        
        self.context = DialogueContext()
        self.doc_reader = DocumentReader()
        
        # Callbacks for UI integration
        self._on_state_change: Optional[Callable[[DialogueState], None]] = None
        self._on_surprise: Optional[Callable[[float], None]] = None
    
    def set_state_callback(self, callback: Callable[[DialogueState], None]):
        """Set callback for state changes."""
        self._on_state_change = callback
    
    def set_surprise_callback(self, callback: Callable[[float], None]):
        """Set callback for surprise events."""
        self._on_surprise = callback
    
    def _change_state(self, new_state: DialogueState):
        """Change state and notify listeners."""
        old_state = self.context.state
        self.context.state = new_state
        
        if self._on_state_change and old_state != new_state:
            self._on_state_change(new_state)
    
    def process_input(self, user_input: str) -> Tuple[str, DialogueState]:
        """
        Process user input based on current state.
        
        This is the main entry point for dialogue processing.
        
        Args:
            user_input: Raw user input string
        
        Returns:
            Tuple of (response_text, new_state)
        """
        user_input = user_input.strip()
        
        if not user_input:
            return "I'm listening...", self.context.state
        
        # Route based on current state
        if self.context.state == DialogueState.IDLE:
            response = self._handle_idle_input(user_input)
        
        elif self.context.state == DialogueState.CLARIFICATION_PENDING:
            response = self._handle_clarification(user_input)
        
        elif self.context.state == DialogueState.CONFIRMATION_PENDING:
            response = self._handle_confirmation(user_input)
        
        elif self.context.state == DialogueState.FILE_REVIEW:
            response = self._handle_file_review(user_input)
        
        else:
            response = self._handle_idle_input(user_input)
        
        # Record exchange
        self.context.add_exchange(user_input, response)
        
        return response, self.context.state
    
    def _handle_idle_input(self, user_input: str) -> str:
        """Handle input when in IDLE state."""
        lower_input = user_input.lower()
        
        # Check for file ingestion request
        if lower_input.startswith("read ") or lower_input.startswith("load "):
            return self._start_file_ingestion(user_input[5:].strip())
        
        # Check for teaching statement
        if self._is_teaching_statement(user_input):
            return self._process_teaching(user_input)
        
        # Check for query
        if self._is_question(user_input):
            return self._process_query(user_input)
        
        # Default: try to extract knowledge
        return self._process_teaching(user_input)
    
    def _is_teaching_statement(self, text: str) -> bool:
        """Check if input looks like a teaching statement."""
        # Teaching patterns
        patterns = [
            " is ", " are ", " was ", " were ",
            " eats ", " eat ", " flies ", " fly ",
            " has ", " have ", " can ", " cannot ",
            " do ", " does ", " don't ", " doesn't "
        ]
        lower = text.lower()
        return any(p in lower for p in patterns)
    
    def _is_question(self, text: str) -> bool:
        """Check if input is a question."""
        question_starters = ["what", "why", "how", "is", "are", "can", "does", "do", "?"]
        lower = text.lower().strip()
        return lower.endswith("?") or any(lower.startswith(q) for q in question_starters)
    
    def _process_teaching(self, user_input: str) -> str:
        """Process a teaching statement."""
        # Extract triples
        triples = text_to_triples(user_input)
        
        if not triples:
            return "I couldn't understand that as knowledge. Could you rephrase?"
        
        # Check each triple against System 2
        responses = []
        
        for triple in triples:
            result = self._evaluate_triple(triple)
            responses.append(result)
        
        return "\n".join(responses)
    
    def _evaluate_triple(self, triple: Triple) -> str:
        """
        Evaluate a triple against existing knowledge.
        
        This is where the Contradiction Loop starts.
        """
        if not self.controller:
            # No System 2 - just accept
            self._learn_triple(triple)
            return f"✓ Learned: {triple.subject} {triple.predicate} {triple.object}"
        
        # Import System 2 types
        from core.system2 import Triple as S2Triple, DiagnosisType
        
        # Create System 2 triple
        s2_triple = S2Triple(
            agent=triple.subject,
            action=triple.predicate,
            object=triple.object
        )
        
        # Get diagnosis
        diagnosis = self.controller.diagnose(s2_triple)
        
        if diagnosis.diagnosis_type == DiagnosisType.NO_ISSUE:
            # Clean - learn it
            self._learn_triple(triple)
            return f"✓ Learned: {triple.subject} {triple.predicate} {triple.object}"
        
        elif diagnosis.diagnosis_type == DiagnosisType.CONSTRAINT_VIOLATION:
            # Conflict detected - enter CLARIFICATION_PENDING
            self.context.pending_triple = triple
            self.context.pending_violation = diagnosis.explanation
            self.context.pending_diagnosis = diagnosis
            self.context.clarifications_asked += 1
            self._change_state(DialogueState.CLARIFICATION_PENDING)
            
            return (f"🤔 My knowledge says: {diagnosis.explanation}\n"
                   f"   Why do you say \"{triple.subject} {triple.predicate} {triple.object}\"?")
        
        elif diagnosis.diagnosis_type == DiagnosisType.MISSING_KNOWLEDGE:
            # Need more info
            self.context.pending_triple = triple
            self.context.pending_violation = diagnosis.explanation
            self._change_state(DialogueState.CLARIFICATION_PENDING)
            
            missing = ", ".join(diagnosis.missing_properties)
            return (f"❓ I need to know: {missing}\n"
                   f"   Can you tell me about {triple.object}?")
        
        else:
            # Unknown schema - tentatively accept
            self._learn_triple(triple)
            return f"✓ Learned (new pattern): {triple.subject} {triple.predicate} {triple.object}"
    
    def _handle_clarification(self, user_input: str) -> str:
        """
        Handle user explanation for a contradiction.
        
        User answers are PROPOSALS, not automatic truth.
        System 2 decides whether to accept.
        """
        lower = user_input.lower()
        
        # Check for rejection
        if lower in ("nevermind", "forget it", "cancel", "skip", "no"):
            self._change_state(DialogueState.IDLE)
            return "Okay, I'll discard that statement."
        
        # Analyze the explanation
        proposal = self._analyze_explanation(user_input)
        
        if proposal.proposal_type == ProposalType.NEW_SUBCLASS:
            # User defined a new subclass
            return self._handle_subclass_proposal(proposal)
        
        elif proposal.proposal_type == ProposalType.EXCEPTION:
            # User declared an exception
            return self._handle_exception_proposal(proposal)
        
        elif proposal.proposal_type == ProposalType.NEW_PROPERTY:
            # User provided a property value
            return self._handle_property_proposal(proposal)
        
        else:
            # Could not understand - ask again
            return ("I'm not sure I understand. Could you explain differently?\n"
                   f"   For example: 'It's a special kind of {self.context.pending_triple.subject}' or\n"
                   f"   '{self.context.pending_triple.subject} is an exception'")
    
    def _analyze_explanation(self, explanation: str) -> Proposal:
        """
        Analyze user's explanation to determine proposal type.
        
        Patterns:
        - "It's a robot dog" -> NEW_SUBCLASS (robot dog is_a dog)
        - "X is an exception" -> EXCEPTION
        - "X is mechanical" -> NEW_PROPERTY
        """
        lower = explanation.lower()
        pending = self.context.pending_triple
        
        # Pattern: "It's a X" or "It is a X"
        if "it's a " in lower or "it is a " in lower:
            # Extract the new class
            if "it's a " in lower:
                new_class = lower.split("it's a ", 1)[1].strip()
            else:
                new_class = lower.split("it is a ", 1)[1].strip()
            
            # Remove trailing punctuation
            new_class = new_class.rstrip(".,!?")
            
            return Proposal(
                proposal_type=ProposalType.NEW_SUBCLASS,
                subject=new_class,
                predicate="is_a",
                object=pending.subject if pending else "",
                original_triple=pending,
                explanation=explanation
            )
        
        # Pattern: "X is an exception" or "exception"
        if "exception" in lower:
            return Proposal(
                proposal_type=ProposalType.EXCEPTION,
                subject=pending.subject if pending else "",
                predicate=pending.predicate if pending else "",
                object=pending.object if pending else "",
                original_triple=pending,
                explanation=explanation
            )
        
        # Pattern: Property assignment "X is Y"
        if " is " in lower and pending:
            parts = lower.split(" is ", 1)
            if len(parts) == 2:
                prop_value = parts[1].strip().rstrip(".,!?")
                return Proposal(
                    proposal_type=ProposalType.NEW_PROPERTY,
                    subject=pending.object,  # Apply to the object
                    predicate="has_property",
                    object=prop_value,
                    original_triple=pending,
                    explanation=explanation
                )
        
        return Proposal(
            proposal_type=ProposalType.UNKNOWN,
            subject="",
            predicate="",
            object="",
            original_triple=pending,
            explanation=explanation
        )
    
    def _handle_subclass_proposal(self, proposal: Proposal) -> str:
        """
        Handle a new subclass proposal.
        
        Example: "It's a robot dog" creates (robot dog) --[is_a]--> (dog)
        and allows the original action for the subclass.
        """
        pending = self.context.pending_triple
        new_class = proposal.subject
        parent_class = proposal.object
        
        # Learn the subclass relationship
        subclass_triple = Triple(
            subject=new_class,
            predicate="is_a",
            object=parent_class,
            relation_type=RelationType.IS_A,
            confidence=0.4,  # Higher than sensor, but not absolute
            source_text=proposal.explanation
        )
        self._learn_triple(subclass_triple)
        
        # Allow the original action for this subclass (as exception)
        if pending:
            exception_triple = Triple(
                subject=new_class,
                predicate=pending.predicate,
                object=pending.object,
                relation_type=RelationType.ACTION,
                confidence=0.4,
                source_text=f"Exception: {proposal.explanation}"
            )
            self._learn_triple(exception_triple)
            
            # Set property if applicable
            if self.controller:
                # Mark this specific subject as allowed
                self.controller.set_property(
                    new_class,
                    f"can_{pending.predicate}_{pending.object}",
                    True
                )
        
        self.context.exceptions_learned += 1
        self._change_state(DialogueState.IDLE)
        
        return (f"✓ Understood. I've learned:\n"
               f"   • {new_class} is a type of {parent_class}\n"
               f"   • {new_class} can {pending.predicate} {pending.object}")
    
    def _handle_exception_proposal(self, proposal: Proposal) -> str:
        """Handle an exception declaration."""
        pending = self.context.pending_triple
        
        if pending:
            # Mark this as an explicit exception
            pending.confidence = 0.5  # Boost since user confirmed
            self._learn_triple(pending)
            
            # Record exception in System 2
            if self.controller:
                self.controller.set_property(
                    pending.subject,
                    f"exception_{pending.predicate}_{pending.object}",
                    True
                )
        
        self.context.exceptions_learned += 1
        self._change_state(DialogueState.IDLE)
        
        return (f"✓ Noted as exception: {pending.subject} {pending.predicate} {pending.object}\n"
               f"   I've updated my knowledge to allow this case.")
    
    def _handle_property_proposal(self, proposal: Proposal) -> str:
        """Handle a new property proposal."""
        if self.controller:
            self.controller.set_property(
                proposal.subject,
                proposal.object,
                True
            )
        
        # Also learn as triple
        prop_triple = Triple(
            subject=proposal.subject,
            predicate="has_property",
            object=proposal.object,
            relation_type=RelationType.PROPERTY,
            confidence=0.4,
            source_text=proposal.explanation
        )
        self._learn_triple(prop_triple)
        
        # Re-evaluate pending if we have it
        if self.context.pending_triple:
            pending = self.context.pending_triple
            result = self._evaluate_triple(pending)
            
            if self.context.state == DialogueState.IDLE:
                return f"✓ Now I know {proposal.subject} is {proposal.object}.\n{result}"
            else:
                return f"✓ Noted: {proposal.subject} is {proposal.object}. Need more info..."
        
        self._change_state(DialogueState.IDLE)
        return f"✓ Learned: {proposal.subject} is {proposal.object}"
    
    def _learn_triple(self, triple: Triple):
        """Commit a triple to memory."""
        if not self.memory:
            return
        
        # Add nodes
        if not self.memory.has_node(triple.subject):
            self.memory.add_node(triple.subject, novelty_score=0.5)
        if not self.memory.has_node(triple.object):
            self.memory.add_node(triple.object, novelty_score=0.5)
        
        # Add synapse
        self.memory.add_synapse(
            triple.subject,
            triple.object,
            type=triple.predicate,
            weight=0.5,
            confidence=triple.confidence
        )
    
    def _process_query(self, user_input: str) -> str:
        """Process a query using the query engine."""
        if not self.query_engine:
            return "Query engine not available."
        
        result = self.query_engine.ask(user_input)
        return self._format_query_result(result)
    
    def _format_query_result(self, result) -> str:
        """Format query result for display."""
        if result.success:
            lines = [f"💡 {result.answer}"]
            if result.confidence > 0:
                conf_bar = "█" * int(result.confidence * 10) + "░" * (10 - int(result.confidence * 10))
                lines.append(f"   Confidence: [{conf_bar}] {result.confidence:.1%}")
            return "\n".join(lines)
        else:
            return f"❌ {result.answer}"
    
    def _start_file_ingestion(self, filename: str) -> str:
        """Start file ingestion workflow."""
        if not filename:
            return "Please specify a file to read. Example: read biology.txt"
        
        # Try to read and parse
        triples = self.doc_reader.parse_file(filename)
        
        if not triples:
            return f"Could not extract any knowledge from '{filename}'."
        
        # Create staging buffer
        self.context.staging_buffer = StagingBuffer()
        self.context.staging_buffer.source_file = filename
        
        # Add all triples with schema checking
        clean, flagged = self.context.staging_buffer.add_triples(
            triples,
            system2=self.controller,
            main_memory=self.memory
        )
        
        self._change_state(DialogueState.FILE_REVIEW)
        
        summary = self.context.staging_buffer.get_summary()
        
        return (f"📖 Read file: {filename}\n\n{summary}\n\n"
               "Type 'commit' to save clean items, 'review' to see details, or 'cancel' to discard.")
    
    def _handle_confirmation(self, user_input: str) -> str:
        """Handle confirmation for pending action."""
        lower = user_input.lower().strip()
        
        if lower in ("yes", "y", "confirm", "ok", "proceed"):
            # Confirm action
            self._change_state(DialogueState.IDLE)
            return "✓ Confirmed."
        
        elif lower in ("no", "n", "cancel", "abort"):
            self._change_state(DialogueState.IDLE)
            return "Cancelled."
        
        else:
            return "Please respond with 'yes' or 'no'."
    
    def _handle_file_review(self, user_input: str) -> str:
        """Handle file review state."""
        lower = user_input.lower().strip()
        buffer = self.context.staging_buffer
        
        if not buffer:
            self._change_state(DialogueState.IDLE)
            return "No staged content to review."
        
        if lower == "commit":
            result = buffer.merge_to_main(self.memory)
            self._change_state(DialogueState.IDLE)
            return f"✓ Committed: {result}"
        
        elif lower == "review":
            lines = ["📋 Staged content:"]
            for i, triple in enumerate(buffer.clean_triples[:10]):
                lines.append(f"   {i+1}. {triple}")
            if len(buffer.clean_triples) > 10:
                lines.append(f"   ... and {len(buffer.clean_triples) - 10} more")
            return "\n".join(lines)
        
        elif lower == "flagged":
            if not buffer.flagged_triples:
                return "No flagged items."
            lines = ["⚠️ Flagged items:"]
            for i, (triple, info) in enumerate(buffer.flagged_triples):
                lines.append(f"   {i+1}. {triple}")
                lines.append(f"      Reason: {info.reason}")
            return "\n".join(lines)
        
        elif lower == "cancel":
            buffer.clear()
            self._change_state(DialogueState.IDLE)
            return "Discarded staged content."
        
        elif lower.startswith("approve "):
            # Approve specific flagged items
            try:
                idx = int(lower.split()[1]) - 1
                if 0 <= idx < len(buffer.flagged_triples):
                    triple, _ = buffer.flagged_triples[idx]
                    triple.flagged = False
                    buffer.clean_triples.append(triple)
                    buffer.flagged_triples.pop(idx)
                    return f"✓ Approved: {triple}"
                else:
                    return f"Invalid index. Use 1-{len(buffer.flagged_triples)}"
            except (ValueError, IndexError):
                return "Usage: approve <number>"
        
        else:
            return ("Commands: 'commit', 'review', 'flagged', 'approve <n>', or 'cancel'")
    
    def get_state_indicator(self) -> str:
        """Get a visual indicator for current state."""
        indicators = {
            DialogueState.IDLE: "💭 IDLE",
            DialogueState.PROCESSING: "⚙️ PROCESSING",
            DialogueState.CLARIFICATION_PENDING: "❓ CLARIFICATION",
            DialogueState.CONFIRMATION_PENDING: "⏳ CONFIRMATION",
            DialogueState.FILE_REVIEW: "📖 FILE REVIEW",
        }
        return indicators.get(self.context.state, str(self.context.state))
    
    def process_file_content(self, content: str, filename: str = "uploaded.txt") -> Tuple[str, DialogueState]:
        """
        Process file content directly (for dashboard upload).
        
        Args:
            content: Raw text content of the file
            filename: Optional filename for metadata
        
        Returns:
            Tuple of (response_text, new_state)
        """
        if not content.strip():
            return "No content provided.", self.context.state
        
        # Parse the content
        triples = self.doc_reader.parse_text(content)
        
        if not triples:
            return f"Could not extract any knowledge from the content.", self.context.state
        
        # Create staging buffer
        self.context.staging_buffer = StagingBuffer()
        self.context.staging_buffer.source_file = filename
        
        # Add all triples with schema checking
        clean, flagged = self.context.staging_buffer.add_triples(
            triples,
            system2=self.controller,
            main_memory=self.memory
        )
        
        self._change_state(DialogueState.FILE_REVIEW)
        
        summary = self.context.staging_buffer.get_summary()
        
        return (f"📖 Processed: {filename}\n\n{summary}", self.context.state)


# CLI for testing
if __name__ == "__main__":
    print("=" * 60)
    print("NCGN Dialogue Manager - Interactive Test")
    print("=" * 60)
    print("Type statements or questions. Type 'quit' to exit.")
    print("=" * 60)
    
    manager = DialogueManager()
    
    while True:
        try:
            state_indicator = manager.get_state_indicator()
            user_input = input(f"\n[{state_indicator}] You> ").strip()
            
            if user_input.lower() in ('quit', 'exit', ':q'):
                print("Goodbye!")
                break
            
            response, new_state = manager.process_input(user_input)
            print(f"\nNCGN> {response}")
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")
