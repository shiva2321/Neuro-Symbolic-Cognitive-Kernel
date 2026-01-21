"""
NCGN Query Engine - Natural Language Query Processing

Processes questions about the cognitive network:
- "What does X eat?" - Association queries
- "Is Y edible?" - Property queries  
- "Why can't X do Y?" - Explanation queries
- Traces reasoning paths through the graph

Provides explanations of the system's knowledge and reasoning.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum


class QueryType(Enum):
    """Types of queries the engine can handle."""
    ASSOCIATION = "association"      # What does X relate to?
    PROPERTY = "property"            # Does X have property Y?
    EXPLANATION = "explanation"      # Why is X related to Y?
    STATE = "state"                  # What is currently active?
    PREDICTION = "prediction"        # What will happen if X?
    CONSTRAINT = "constraint"        # Can X do Y?
    UNKNOWN = "unknown"


@dataclass
class QueryResult:
    """Result of a query."""
    query_type: QueryType
    success: bool
    answer: str
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.0
    related_nodes: List[str] = field(default_factory=list)
    reasoning_path: List[Tuple[str, str, str]] = field(default_factory=list)  # (source, relation, target)


@dataclass
class ExplanationPath:
    """A path of reasoning from source to target."""
    start: str
    end: str
    path: List[Tuple[str, str, str]]  # (node, relation, next_node)
    total_confidence: float
    explanation: str


class QueryParser:
    """
    Parses natural language queries into structured form.
    
    Handles patterns like:
    - "What does X eat/do/like?"
    - "Is X edible/animate/property?"
    - "Why can't X eat Y?"
    - "What is active/firing?"
    """
    
    # Query patterns
    PATTERNS = {
        # Association queries
        r"what\s+does\s+(\w+)\s+(\w+)\??": ("association", ["subject", "relation"]),
        r"what\s+(\w+)\s+does\s+(\w+)\s+(\w+)\??": ("association", ["filler", "subject", "relation"]),
        r"(\w+)\s+eats?\s+what\??": ("association", ["subject"]),
        
        # Property queries
        r"is\s+(\w+)\s+(edible|animate|alive|dangerous)\??": ("property", ["subject", "property"]),
        r"does\s+(\w+)\s+have\s+(\w+)\??": ("property", ["subject", "property"]),
        r"can\s+(\w+)\s+(\w+)\??": ("property", ["subject", "action"]),
        
        # Explanation queries
        r"why\s+can'?t?\s+(\w+)\s+(\w+)\s+(\w+)\??": ("explanation", ["agent", "action", "object"]),
        r"why\s+is\s+(\w+)\s+(\w+)\??": ("explanation", ["subject", "predicate"]),
        r"explain\s+(\w+)\s*(?:to|->|→)?\s*(\w+)": ("explanation", ["source", "target"]),
        
        # State queries
        r"what\s+is\s+(active|firing|energized)\??": ("state", ["state_type"]),
        r"show\s+(state|active|nodes)": ("state", ["state_type"]),
        r"current\s+state": ("state", []),
        
        # Prediction queries
        r"what\s+happens?\s+if\s+(\w+)\s*(?:is|gets)?\s*(activated|stimulated)?\??": ("prediction", ["subject"]),
        r"predict\s+(\w+)": ("prediction", ["subject"]),
    }
    
    def parse(self, query: str) -> Tuple[QueryType, Dict[str, str]]:
        """
        Parse a natural language query.
        
        Returns:
            Tuple of (query_type, extracted_parameters)
        """
        query = query.lower().strip()
        
        for pattern, (query_type, param_names) in self.PATTERNS.items():
            match = re.match(pattern, query, re.IGNORECASE)
            if match:
                params = {}
                for i, name in enumerate(param_names):
                    if i < len(match.groups()):
                        params[name] = match.group(i + 1)
                return QueryType(query_type), params
        
        # Try to extract any mentioned nodes
        words = query.split()
        return QueryType.UNKNOWN, {"raw": query, "words": words}


class QueryEngine:
    """
    Processes queries against the NCGN knowledge graph.
    
    Supports:
    - Association queries (What does X eat?)
    - Property queries (Is X edible?)
    - Explanation queries (Why can't X eat Y?)
    - State queries (What is active?)
    """
    
    def __init__(self, memory, controller):
        """
        Initialize query engine.
        
        Args:
            memory: GraphMemory instance
            controller: System2Controller instance
        """
        self.memory = memory
        self.controller = controller
        self.parser = QueryParser()
        
        # Query history for context
        self.history: List[Tuple[str, QueryResult]] = []
        self.max_history = 50
    
    def ask(self, question: str) -> QueryResult:
        """
        Process a natural language question.
        
        Args:
            question: The question to answer
        
        Returns:
            QueryResult with answer and evidence
        """
        query_type, params = self.parser.parse(question)
        
        result = None
        
        if query_type == QueryType.ASSOCIATION:
            result = self._handle_association_query(params)
        elif query_type == QueryType.PROPERTY:
            result = self._handle_property_query(params)
        elif query_type == QueryType.EXPLANATION:
            result = self._handle_explanation_query(params)
        elif query_type == QueryType.STATE:
            result = self._handle_state_query(params)
        elif query_type == QueryType.PREDICTION:
            result = self._handle_prediction_query(params)
        else:
            result = self._handle_unknown_query(params)
        
        # Store in history
        self.history.append((question, result))
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        return result
    
    def _handle_association_query(self, params: Dict[str, str]) -> QueryResult:
        """Handle 'What does X eat/do?' queries."""
        subject = params.get("subject", "")
        relation = params.get("relation", "eats")
        
        if not subject:
            return QueryResult(
                query_type=QueryType.ASSOCIATION,
                success=False,
                answer="Please specify a subject (e.g., 'What does dog eat?')"
            )
        
        # Get node and its outgoing edges
        node = self.memory.get_node(subject)
        if not node:
            return QueryResult(
                query_type=QueryType.ASSOCIATION,
                success=False,
                answer=f"I don't know about '{subject}'."
            )
        
        # Find matching edges
        outgoing = self.memory.get_outgoing(subject)
        matches = []
        
        for synapse in outgoing:
            if relation in synapse.type or synapse.type in relation:
                matches.append((synapse.target_id, synapse.weight, synapse.confidence))
        
        if not matches:
            # Try any outgoing edge
            for synapse in outgoing:
                matches.append((synapse.target_id, synapse.weight, synapse.confidence))
        
        if matches:
            # Sort by confidence * weight
            matches.sort(key=lambda x: x[1] * x[2], reverse=True)
            
            best = matches[0]
            evidence = [
                f"{subject} → {target} (weight: {w:.2f}, confidence: {c:.2f})"
                for target, w, c in matches[:3]
            ]
            
            return QueryResult(
                query_type=QueryType.ASSOCIATION,
                success=True,
                answer=f"Based on learned associations, {subject} {relation} {best[0]}.",
                evidence=evidence,
                confidence=best[2],
                related_nodes=[m[0] for m in matches],
                reasoning_path=[(subject, relation, best[0])]
            )
        
        return QueryResult(
            query_type=QueryType.ASSOCIATION,
            success=False,
            answer=f"I don't know what {subject} {relation}."
        )
    
    def _handle_property_query(self, params: Dict[str, str]) -> QueryResult:
        """Handle 'Is X edible?' queries."""
        subject = params.get("subject", "")
        property_name = params.get("property", params.get("action", ""))
        
        if not subject:
            return QueryResult(
                query_type=QueryType.PROPERTY,
                success=False,
                answer="Please specify a subject."
            )
        
        # Normalize property name
        if not property_name.startswith("is_"):
            property_name = f"is_{property_name}"
        
        # Check System 2 properties
        value = self.controller.get_property(subject, property_name)
        
        if value is not None:
            answer = f"Yes, {subject} is {property_name.replace('is_', '')}." if value else \
                     f"No, {subject} is not {property_name.replace('is_', '')}."
            return QueryResult(
                query_type=QueryType.PROPERTY,
                success=True,
                answer=answer,
                confidence=1.0,
                evidence=[f"Property '{property_name}' for '{subject}' is {value}"],
                related_nodes=[subject]
            )
        
        # Check for property nodes in graph
        prop_node = property_name.replace("is_", "")
        if self.memory.get_synapse(subject, prop_node):
            synapse = self.memory.get_synapse(subject, prop_node)
            return QueryResult(
                query_type=QueryType.PROPERTY,
                success=True,
                answer=f"{subject} is associated with {prop_node} (confidence: {synapse.confidence:.2f}).",
                confidence=synapse.confidence,
                related_nodes=[subject, prop_node]
            )
        
        return QueryResult(
            query_type=QueryType.PROPERTY,
            success=False,
            answer=f"I don't know if {subject} is {property_name.replace('is_', '')}."
        )
    
    def _handle_explanation_query(self, params: Dict[str, str]) -> QueryResult:
        """Handle 'Why can't X eat Y?' queries."""
        agent = params.get("agent", params.get("source", ""))
        action = params.get("action", "eat")
        obj = params.get("object", params.get("target", ""))
        
        if not agent or not obj:
            return QueryResult(
                query_type=QueryType.EXPLANATION,
                success=False,
                answer="Please specify both agent and object (e.g., 'Why can't dog eat metal?')"
            )
        
        # Check for schema violation
        from .system2 import Triple, DiagnosisType
        
        triple = Triple(agent=agent, action=action, object=obj)
        diagnosis = self.controller.diagnose(triple)
        
        if diagnosis.diagnosis_type == DiagnosisType.CONSTRAINT_VIOLATION:
            violations = ", ".join(diagnosis.violated_constraints)
            return QueryResult(
                query_type=QueryType.EXPLANATION,
                success=True,
                answer=f"{agent.capitalize()} cannot {action} {obj} because it violates constraint(s): {violations}.",
                evidence=[
                    f"Schema '{action}' requires target to satisfy: {violations}",
                    f"{obj} does not satisfy these constraints"
                ],
                confidence=diagnosis.confidence,
                related_nodes=[agent, action, obj],
                reasoning_path=[
                    (agent, "wants_to", action),
                    (action, "requires", violations),
                    (obj, "violates", violations)
                ]
            )
        
        elif diagnosis.diagnosis_type == DiagnosisType.MISSING_KNOWLEDGE:
            missing = ", ".join(diagnosis.missing_properties)
            return QueryResult(
                query_type=QueryType.EXPLANATION,
                success=True,
                answer=f"I'm not sure if {agent} can {action} {obj}. Missing knowledge: {missing}.",
                evidence=[f"Unknown properties for '{obj}': {missing}"],
                confidence=0.3,
                related_nodes=[agent, action, obj]
            )
        
        # No violation detected
        return QueryResult(
            query_type=QueryType.EXPLANATION,
            success=True,
            answer=f"There's no known reason why {agent} cannot {action} {obj}.",
            confidence=0.5,
            related_nodes=[agent, action, obj]
        )
    
    def _handle_state_query(self, params: Dict[str, str]) -> QueryResult:
        """Handle 'What is active?' queries."""
        active_nodes = self.memory.get_active_nodes()
        
        if not active_nodes:
            return QueryResult(
                query_type=QueryType.STATE,
                success=True,
                answer="No nodes are currently active."
            )
        
        # Get energies
        node_info = []
        for node_id in sorted(active_nodes):
            node = self.memory.get_node(node_id)
            if node:
                node_info.append(f"{node_id} (E={node.energy:.3f})")
        
        return QueryResult(
            query_type=QueryType.STATE,
            success=True,
            answer=f"Currently active nodes ({len(active_nodes)}): {', '.join(node_info)}",
            related_nodes=list(active_nodes),
            evidence=[f"Total nodes: {self.memory.node_count()}", 
                     f"Total edges: {self.memory.edge_count()}"]
        )
    
    def _handle_prediction_query(self, params: Dict[str, str]) -> QueryResult:
        """Handle 'What happens if X is activated?' queries."""
        subject = params.get("subject", "")
        
        if not subject or not self.memory.has_node(subject):
            return QueryResult(
                query_type=QueryType.PREDICTION,
                success=False,
                answer=f"I don't know about '{subject}'."
            )
        
        # Find all reachable nodes from subject
        outgoing = self.memory.get_outgoing(subject)
        
        if not outgoing:
            return QueryResult(
                query_type=QueryType.PREDICTION,
                success=True,
                answer=f"If {subject} is activated, no other nodes will be directly affected (no outgoing connections).",
                related_nodes=[subject]
            )
        
        predictions = []
        reasoning = []
        
        for synapse in outgoing:
            target = synapse.target_id
            predicted_energy = synapse.weight  # Simplified prediction
            predictions.append(f"{target} (E≈{predicted_energy:.2f})")
            reasoning.append((subject, synapse.type, target))
        
        return QueryResult(
            query_type=QueryType.PREDICTION,
            success=True,
            answer=f"If {subject} is activated, the following nodes will receive energy: {', '.join(predictions)}",
            related_nodes=[subject] + [s.target_id for s in outgoing],
            reasoning_path=reasoning,
            confidence=0.7
        )
    
    def _handle_unknown_query(self, params: Dict[str, str]) -> QueryResult:
        """Handle queries that don't match known patterns."""
        words = params.get("words", [])
        raw = params.get("raw", "")
        
        # Try to find mentioned nodes
        found_nodes = []
        for word in words:
            if self.memory.has_node(word):
                found_nodes.append(word)
        
        if found_nodes:
            # Get info about found nodes
            info = []
            for node_id in found_nodes:
                node = self.memory.get_node(node_id)
                outgoing = self.memory.get_outgoing(node_id)
                info.append(f"{node_id}: E={node.energy:.2f}, {len(outgoing)} connections")
            
            return QueryResult(
                query_type=QueryType.UNKNOWN,
                success=True,
                answer=f"I found these nodes in your query: {', '.join(info)}. Please ask a more specific question.",
                related_nodes=found_nodes
            )
        
        return QueryResult(
            query_type=QueryType.UNKNOWN,
            success=False,
            answer="I couldn't understand that question. Try asking:\n" +
                   "- 'What does X eat?'\n" +
                   "- 'Is X edible?'\n" +
                   "- 'Why can't X eat Y?'\n" +
                   "- 'What is active?'"
        )
    
    def explain(self, node_id: str) -> ExplanationPath:
        """
        Generate an explanation of why a node has its current state.
        
        Traces back through incoming connections.
        """
        if not self.memory.has_node(node_id):
            return ExplanationPath(
                start="?",
                end=node_id,
                path=[],
                total_confidence=0.0,
                explanation=f"Node '{node_id}' not found."
            )
        
        # Trace back through incoming edges
        incoming = self.memory.get_incoming(node_id)
        
        if not incoming:
            return ExplanationPath(
                start=node_id,
                end=node_id,
                path=[],
                total_confidence=1.0,
                explanation=f"'{node_id}' has no incoming connections (it's a root node or input)."
            )
        
        # Build explanation path
        path = []
        sources = []
        total_conf = 0.0
        
        for source_id, synapse in incoming:
            path.append((source_id, synapse.type, node_id))
            sources.append(f"{source_id} ({synapse.type}, conf={synapse.confidence:.2f})")
            total_conf += synapse.confidence
        
        avg_conf = total_conf / len(incoming) if incoming else 0.0
        
        explanation = f"'{node_id}' receives input from: {', '.join(sources)}"
        
        return ExplanationPath(
            start=incoming[0][0] if incoming else node_id,
            end=node_id,
            path=path,
            total_confidence=avg_conf,
            explanation=explanation
        )
    
    def test_association(self, source: str, target: str) -> float:
        """
        Test the strength of association between two nodes.
        
        Returns:
            Association strength (0.0 to 1.0)
        """
        synapse = self.memory.get_synapse(source, target)
        if synapse:
            return synapse.weight * synapse.confidence
        
        # Check reverse
        synapse = self.memory.get_synapse(target, source)
        if synapse:
            return synapse.weight * synapse.confidence * 0.5  # Weaker for reverse
        
        return 0.0
    
    def get_related(self, node_id: str, max_depth: int = 2) -> Dict[str, float]:
        """
        Get all nodes related to a given node within max_depth hops.
        
        Returns:
            Dict mapping node_id to relevance score
        """
        if not self.memory.has_node(node_id):
            return {}
        
        related = {}
        visited = {node_id}
        frontier = [(node_id, 1.0)]  # (node, relevance)
        
        for depth in range(max_depth):
            next_frontier = []
            
            for current_id, relevance in frontier:
                # Outgoing
                for synapse in self.memory.get_outgoing(current_id):
                    target = synapse.target_id
                    if target not in visited:
                        score = relevance * synapse.weight * 0.7  # Decay with distance
                        related[target] = max(related.get(target, 0), score)
                        next_frontier.append((target, score))
                        visited.add(target)
                
                # Incoming
                for source_id, synapse in self.memory.get_incoming(current_id):
                    if source_id not in visited:
                        score = relevance * synapse.weight * 0.5  # Less for reverse
                        related[source_id] = max(related.get(source_id, 0), score)
                        next_frontier.append((source_id, score))
                        visited.add(source_id)
            
            frontier = next_frontier
        
        return dict(sorted(related.items(), key=lambda x: x[1], reverse=True))
