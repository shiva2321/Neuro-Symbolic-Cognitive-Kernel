"""
VSA Left-Corner Parser
======================

Implements a biologically plausible Left-Corner Parser using Vector Symbolic Architectures.
Uses VSA buffers (STACK, TREE) and Neural Integrator dynamics (approximated).

Architecture:
- Input Buffer: Holds current word vector.
- Stack Buffer: Holds predicted syntactic goals.
- Tree Buffer: Holds partial/complete parse tree.

Algorithm: Left-Corner Parsing (Shift-Predict-Reduce).
"""

from typing import List, Tuple, Dict, Optional
import python.core.vsa.hypervec_shim as hv
# from python.core.vsa.semantic_memory import SemanticMemory

class LeftCornerParser:
    def __init__(self):
        # 1. Initialize Buffers (Neural Integrators)
        self.stack = hv.HyperVector(seed=0) # Identity/Empty? Or Random?
        self.tree = hv.HyperVector(seed=0)  # Empty tree
        
        # 2. Vocabulary & Roles
        # In a real system, these come from Semantic Memory.
        # Here we define the functional vectors for the parser mechanics.
        self.vectors = {
            "PUSH_OP": hv.HyperVector(seed=9999),  # Operation to push stack down
            "POP_OP": hv.HyperVector(seed=9999).permute_inverse(0), # Just inverse of push? NO.
            # Push logic: Stack = Stack * PUSH_OP + NewItem
            # Pop logic:  Stack = Stack * inv(PUSH_OP) - NewItem? 
            # HRR Stacks are tricky. 
            # Standard HRR Stack: S_t = S_{t-1} * PUSH_KEY + Item.
            # Pop: Item = S_t; S_{t-1} = (S_t - Item) * inv(PUSH_KEY).
             
            "ROOT": hv.HyperVector(seed=1000),      # Top of stack symbol
            
            # Roles
            "SUBJECT": hv.HyperVector(seed=2001),
            "ACTION": hv.HyperVector(seed=2002),
            "OBJECT": hv.HyperVector(seed=2003),
            
            # Categories (POS)
            "DT": hv.HyperVector(seed=3001),
            "NN": hv.HyperVector(seed=3002),
            "VB": hv.HyperVector(seed=3003),
            "WP": hv.HyperVector(seed=3004), # Wh-Pronoun (Who, What)
            
            # Phrase Types
            "S": hv.HyperVector(seed=4001),
            "NP": hv.HyperVector(seed=4002),
            "VP": hv.HyperVector(seed=4003),
            "Q": hv.HyperVector(seed=4004), # Question
        }
        
        self.symbolic_stack: List[str] = []
        self.constituent = hv.HyperVector(seed=0)

    def reset(self) -> None:
        """Reset parser buffers to initial state (call before each new sentence)."""
        self.stack = hv.HyperVector(seed=0)
        self.tree = hv.HyperVector(seed=0)
        self.symbolic_stack = []
        self.constituent = hv.HyperVector(seed=0)

    def get_vector(self, word: str) -> hv.HyperVector:
        """Look up or create a deterministic hypervector for a word or tag.

        Known structural tokens (roles, POS tags, etc.) are looked up from
        ``self.vectors``.  Unknown surface words are assigned a fresh HV
        seeded by the word's hash — deterministic across calls."""
        if word in self.vectors:
            return self.vectors[word]
        # Deterministic seed from word string so same word always → same HV.
        seed = abs(hash(word)) % (2 ** 31)
        v = hv.HyperVector(seed=seed)
        self.vectors[word] = v
        return v

    def parse_step(self, word: str, tag: str) -> str:
        """
        Execute one step of the parser state machine.
        """
        # Mapping NLTK tags to VSA coarse tags if needed
        if tag in ["WP", "WRB"]: tag = "WP"
        
        stack_top = self.symbolic_stack[-1] if self.symbolic_stack else "EMPTY"
        action = "UNKNOWN"
        
        # RULE 0: QUESTION INITIATION (Wh-Word)
        # If Input=WP (What) and Stack=Empty -> Predict Question (Q)
        if tag == "WP" and stack_top == "EMPTY":
            action = "SHIFT_WH"
            self.symbolic_stack.append("Q")
            # Usually strict LC: Q -> WP VP (What is...) or Q -> WP SQ (What did...)
            # We predict VP/SQ next.
            self.symbolic_stack.append("VP") # Expect verb next (is)
            
        # RULE 1: SHIFT (Start of Sentence / NP)
        elif tag == "DT" and stack_top == "EMPTY":
            action = "SHIFT_INIT"
            self.symbolic_stack.append("S") 
            self.symbolic_stack.append("NP")

        # ... (Existing Rules) ...
        # RULE 2: PREDICT (Inside NP)
        elif tag == "NN" and stack_top == "NP":
             action = "PREDICT_NP_DONE"
             self.symbolic_stack.pop()
        
        # RULE 3: VERB (Transitive/Intransitive)
        elif tag == "VB" and (stack_top == "S" or stack_top == "VP"): # Handle VP from Question too
             # If we are in Q -> VP, we just found the V.
             if stack_top == "VP":
                 # Pop VP, predict NP (Target of Question)
                 # "What(WP) is(VB) X(NP)?"
                 self.symbolic_stack.pop() # VP found (it was just the verb 'is')
                 self.symbolic_stack.append("NP_OBJ") # Expect the target
                 action = "SHIFT_COPULA" # Copula 'is'
             else:
                 # Standard S -> NP VP
                 self.symbolic_stack.append("NP_OBJ")
                 action = "SHIFT_VP"

        # ... (Object Rules) ...
        elif tag == "NN" and stack_top == "NP_OBJ":
            action = "REDUCE_OBJECT"
            self.symbolic_stack.pop()

        # Execute VSA Binding
        v_word = self.get_vector(word)
        v_node = v_word.xor(self.get_vector(tag))
        
        if action == "SHIFT_WH":
            # Bind "What" to PROBE/QUERY role
            # Tree = What * QUERY
            v_query_role = self.get_vector("QUERY") # We need to add this key or use 'SUBJECT'?
            # Let's use 'SUBJECT' for now as placeholder for 'Query Target' ?? 
            # No, 'What' is the operator. 
            # Let's add QUERY role dynamically if missing
            if "QUERY" not in self.vectors: self.vectors["QUERY"] = hv.HyperVector(seed=8888)
            
            self.tree = v_node.xor(self.vectors["QUERY"])
            
        elif action == "SHIFT_COPULA":
            # "Is" - usually ignored in VSA as it's just a relation binder
            # But we can bind it as ACTION
            v_action = self.get_vector("ACTION")
            part = v_node.xor(v_action)
            self.tree = self.tree.bundle(part)
        
        elif action == "SHIFT_INIT":
            self.constituent = v_node
            
        elif action == "PREDICT_NP_DONE":
            self.constituent = self.constituent.bundle(v_node)
            
        elif action == "SHIFT_VP":
             # Subject -> AGENT
             subject_cluster = self.constituent
             v_agent_role = self.get_vector("AGENT")
             part1 = subject_cluster.xor(v_agent_role)
             
             # Verb -> ACTION
             v_action_role = self.get_vector("ACTION")
             part2 = v_node.xor(v_action_role)
             
             self.tree = part1.bundle(part2)
             self.constituent = hv.HyperVector(seed=0)

        elif action == "REDUCE_OBJECT":
            self.constituent = self.constituent.bundle(v_node)
            # Object -> THEME
            # If we are in a Question "What is X?", X is the THEME/TARGET
            # Tree = (What*QUERY) + (Is*ACTION)
            # Now add (X*THEME)
            part3 = self.constituent.xor(self.get_vector("OBJECT"))
            self.tree = self.tree.bundle(part3)
            self.constituent = hv.HyperVector(seed=0)
            
        return action
            
        return action

    def parse_sentence(self, tokens: List[Tuple[str, str]]):
        self.reset()
        try:
            print(f"Parsing: {tokens}")
        except UnicodeEncodeError:
            pass # Skip print on unicode error
        for word, tag in tokens:
            act = self.parse_step(word, tag)
            try:
                print(f"  Word: {word:10} Tag: {tag:5} -> Action: {act}")
            except UnicodeEncodeError:
                pass
            
        return self.tree
