# NSCK AGI: Detailed Implementation Roadmap

**Purpose:** Technical implementation guide for achieving sentient AGI capabilities  
**Audience:** Developers, researchers, contributors  
**Timeline:** 24-60 months (2-5 years)

---

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Phase 1: Language & Communication](#phase-1-language--communication)
3. [Phase 2: Emotional & Social Intelligence](#phase-2-emotional--social-intelligence)
4. [Phase 3: Memory & Dreaming](#phase-3-memory--dreaming)
5. [Phase 4: Continual Learning](#phase-4-continual-learning)
6. [Phase 5: Consciousness & Self-Evolution](#phase-5-consciousness--self-evolution)
7. [Integration Strategy](#integration-strategy)
8. [Testing & Validation](#testing--validation)

---

## Architecture Overview

### Current Stack
```
┌─────────────────────────────────────────┐
│         Cognitive Engine                │  ← Decision Making
├─────────────────────────────────────────┤
│  Global Workspace  │  Metacognition     │  ← Integration Layer
├──────────────┬──────┴───────────────────┤
│ Perception   │ Memory  │ Learning       │  ← Cognitive Modules
├──────────────┼─────────┼────────────────┤
│   SNN (PyTorch)    │    VSA (Rust)     │  ← Neural + Symbolic
└────────────────────┴───────────────────┘
```

### Target Architecture (Post-Implementation)
```
┌─────────────────────────────────────────────────────┐
│              Meta-Cognitive Controller              │
│        (Self-Awareness + Attention Schema)          │
├─────────────────────────────────────────────────────┤
│              Global Workspace (Enhanced)            │
│  (Conscious Access + Broadcasting + IIT Phi)        │
├──────────┬──────────┬──────────┬───────────────────┤
│ Language │ Emotion  │ Social   │ Imagination       │
│ (LLM+VSA)│ (Affect) │ (ToM)    │ (World Model)     │
├──────────┴──────────┴──────────┴───────────────────┤
│  Episodic Memory  │  Semantic Memory  │  Procedural │
│  (with Sleep/Consolidation)                         │
├─────────────────────────────────────────────────────┤
│     Perception     │    Motor Control   │ Learning  │
│  (Multimodal Fusion)                                │
├──────────────┬──────────────────────────────────────┤
│  SNN (Plastic) │  VSA (Reasoning)  │  LLM (Language)│
└────────────────┴───────────────────────────────────┘
```

---

## Phase 1: Language & Communication (Months 1-12)

### Objective
Transform the agent from game-playing to natural language interaction.

### Current State
- ✅ `lingua_cortex.py`: Semantic Folding skeleton
- ⚠️ Limited to simple word vectors
- ❌ No real NLP pipeline

### Implementation Steps

#### 1.1 Semantic Folding Enhancement (Months 1-3)

**File:** `nsck-demo/python/lingua_cortex.py`

**Tasks:**
1. **Build Training Pipeline**
   ```python
   class SemanticFoldingTrainer:
       def __init__(self, grid_size=128, dim=10000):
           self.grid = np.zeros((grid_size, grid_size))
           self.word_to_coords = {}
           self.hypervec_dim = dim
       
       def train_on_corpus(self, corpus_path):
           # 1. Extract word co-occurrence statistics
           # 2. Map to 2D grid using t-SNE or UMAP
           # 3. Generate hypervectors from grid positions
           # 4. Store in codebook
           pass
   ```

2. **Dataset Preparation**
   - Download Wikipedia subset (10GB text)
   - Preprocess: tokenization, co-occurrence matrix
   - Store in efficient format (HDF5 or SQLite)

3. **Training Script**
   ```bash
   # New file: nsck-demo/python/train_semantic_folding.py
   python train_semantic_folding.py \
       --corpus data/wikipedia_subset.txt \
       --grid-size 256 \
       --output models/semantic_grid.pkl
   ```

**Validation:**
- Word similarity tasks (SimLex-999 benchmark)
- Target: >0.6 Spearman correlation

---

#### 1.2 LLM Integration (Months 3-6)

**New File:** `nsck-demo/python/language_module.py`

**Architecture:**
```python
class LanguageModule:
    def __init__(self):
        # Small LLM for language understanding
        self.llm = LlamaCppPython(
            model_path="models/phi-3-mini-4k-instruct.Q4_K_M.gguf",
            n_ctx=4096,
            n_threads=4
        )
        
        # VSA for grounding
        self.vsa_grounding = SymbolGrounder()
        
        # Semantic folding for concept space
        self.semantic_space = SemanticGrid.load("models/semantic_grid.pkl")
    
    def understand(self, text: str) -> Dict[str, Any]:
        """
        Convert text to grounded representation.
        
        Returns:
            - intent: What user wants
            - entities: Objects mentioned
            - predicates: Relations
            - grounded_hv: VSA hypervector of meaning
        """
        # 1. LLM extracts intent and entities
        structured_output = self.llm.extract_structured(text)
        
        # 2. Map to VSA space
        grounded_hv = self.ground_to_vsa(structured_output)
        
        # 3. Query semantic folding for related concepts
        related = self.semantic_space.find_neighbors(grounded_hv)
        
        return {
            "intent": structured_output.intent,
            "entities": structured_output.entities,
            "predicates": structured_output.relations,
            "grounded_hv": grounded_hv,
            "related_concepts": related
        }
    
    def generate(self, meaning_hv: HyperVector, style: str = "neutral") -> str:
        """
        Generate text from grounded meaning.
        """
        # 1. Decode VSA to symbolic form
        symbolic = self.vsa_grounding.decode(meaning_hv)
        
        # 2. LLM converts to natural language
        prompt = f"Express this meaning naturally: {symbolic}"
        return self.llm.generate(prompt)
```

**Integration with Cognitive Engine:**
```python
# In cognitive_engine.py
class CognitiveEngine:
    def __init__(self, ...):
        # Add language module
        self.language = LanguageModule()
    
    def process_instruction(self, text: str) -> str:
        """Process natural language instruction."""
        # 1. Understand input
        meaning = self.language.understand(text)
        
        # 2. Ground to current task
        grounded_state = self.ground_meaning(meaning, self.current_task)
        
        # 3. Execute or plan
        action = self.decide(grounded_state)
        
        # 4. Generate response
        response_hv = self.form_response(action)
        return self.language.generate(response_hv)
```

**Testing:**
```bash
# Interactive testing
python -m nsck_demo.python.chatbot
> "Move to the food"
Agent: I will navigate toward the food item.
> "Why did you do that?"
Agent: I moved toward the food because my energy was low (homeostatic drive).
```

---

#### 1.3 Dialogue System (Months 6-9)

**New File:** `nsck-demo/python/dialogue_manager.py`

**Features:**
1. **Context Tracking**
   - Multi-turn conversation memory
   - Anaphora resolution ("it", "that", "the same one")
   - Topic tracking

2. **Intent Recognition**
   - Commands: "Go to X", "Pick up Y"
   - Questions: "What is X?", "Why did you Z?"
   - Explanations: "Explain your reasoning"

3. **Response Generation**
   - Natural language explanations from rule traces
   - Clarification questions when uncertain
   - Proactive suggestions

**Example Interaction:**
```
Human: What do you see?
Agent: I see a 10x10 grid. There's food at position (7,3) and I'm at (2,2).