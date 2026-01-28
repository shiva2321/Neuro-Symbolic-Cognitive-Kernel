"""
NCGN v2.0 Natural Language Formatter

Peripheral module for translating Brain State into Human Text.
NO REASONING. Pure formatting.
Input: {active_concepts: {dog: 0.9}, decision: REJECT, conflict: {source: sun, target: west}}
Output: "I cannot accept that. My internal model strongly suggests the sun raises in the East."
"""

from typing import Dict, Any, Optional
from .config import Config, DEFAULT_CONFIG

class NaturalLanguageFormatter:
    def __init__(self, model_path: str, config: Config = DEFAULT_CONFIG):
        self.model_path = model_path
        self.config = config
        self._llm = None
        self._client = None
    
    def _ensure_loaded(self) -> None:
        """Lazy load the LLM."""
        if self._llm is not None:
            return
            
        try:
            from llama_cpp import Llama
            self._llm = Llama(
                model_path=self.model_path,
                n_ctx=self.config.llm_context_window,
                n_gpu_layers=self.config.llm_gpu_layers,
                chat_format="chatml",
                verbose=False,
            )
        except ImportError:
            return

    def format_response(self, brain_state: Dict[str, Any]) -> str:
        """
        Generate natural language response from brain state.
        """
        self._ensure_loaded()
        if not self._llm:
            return "Brain online. (LLM not loaded for formatting)"
            
        # Construct prompt based on state
        decision = brain_state.get("decision", "unknown")
        active = brain_state.get("active_concepts", {})
        conflict = brain_state.get("conflict", None)
        user_input = brain_state.get("user_input", "")
        
        # Format context
        context_str = ", ".join([f"{k}({v:.2f})" for k,v in list(active.items())[:10]])
        
        system_prompt = """You are the Voice of NCGN. 
You translate the brain's internal mathematical state into human language.
Reflect the system's decision and confidence accurately.

State interpretation:
- DECISION: ACCEPT -> Acknowledge the new knowledge.
- DECISION: REJECT -> Politely refuse, citing the conflict.
- DECISION: CURIOSITY -> Ask for clarification.

CRITICAL: Do NOT answer from your own knowledge. ONLY use the 'Active Context' provided. If the context is missing information, say you don't know. Do NOT hallucinate reasoning.
"""
        
        user_prompt = f"""User Input: "{user_input}"
Brain State:
- Decision: {decision}
- Active Context: {context_str}
{f"- Conflict: {conflict}" if conflict else ""}

Generate a single sentence response."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=100,
                temperature=0.7
            )
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            return f"Error formatting response: {e}"
