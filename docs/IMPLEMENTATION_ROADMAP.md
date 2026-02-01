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
Human: Why didn't you move?
Agent: I detected an obstacle at (5,4) which blocks the direct path.
Human: Show me your thought process.
Agent: [Rule Trace] hunger_drive=0.7 -> seek_food -> plan_path -> obstacle_detected -> replan
```

**Implementation:**
```python
class DialogueManager:
    def __init__(self, cognitive_engine, language_module):
        self.engine = cognitive_engine
        self.language = language_module
        self.context_window = deque(maxlen=10)  # Last 10 turns
        self.topic_tracker = TopicModel()
    
    def process_turn(self, user_input: str) -> str:
        """Process one dialogue turn."""
        # 1. Add to context
        self.context_window.append(("user", user_input))
        
        # 2. Resolve references using context
        resolved = self.resolve_anaphora(user_input, self.context_window)
        
        # 3. Classify intent
        intent = self.classify_intent(resolved)
        
        # 4. Route to appropriate handler
        if intent == "command":
            response = self.handle_command(resolved)
        elif intent == "question":
            response = self.handle_question(resolved)
        elif intent == "explanation_request":
            response = self.generate_explanation()
        else:
            response = "I'm not sure what you mean. Can you rephrase?"
        
        # 5. Add response to context
        self.context_window.append(("agent", response))
        return response
    
    def resolve_anaphora(self, text: str, context: deque) -> str:
        """Resolve pronouns and references."""
        # Simple pattern matching for now
        if "it" in text.lower():
            # Find last noun phrase in context
            for role, utterance in reversed(context):
                if role == "agent":
                    # Extract mentioned entities
                    entities = self.language.understand(utterance)["entities"]
                    if entities:
                        text = text.replace("it", entities[0])
                        break
        return text
    
    def generate_explanation(self) -> str:
        """Generate explanation of last action."""
        trace = self.engine.get_last_decision_trace()
        return self.language.generate(self.format_trace_as_hv(trace), style="explanatory")
```

**Validation:**
- Multi-turn conversations (5+ turns)
- Reference resolution accuracy >80%
- User satisfaction survey

---

#### 1.4 Voice I/O (Months 9-12)

**New File:** `nsck-demo/python/voice_interface.py`

**Features:**
1. **Speech-to-Text**: Integrate Whisper or similar ASR
2. **Text-to-Speech**: Use Coqui TTS for natural voice
3. **Voice Activity Detection**: Real-time audio segmentation
4. **Prosody Analysis**: Extract emotional tone from speech

**Architecture:**
```python
class VoiceInterface:
    def __init__(self):
        # Speech recognition
        self.asr = WhisperModel("medium")
        
        # Speech synthesis
        self.tts = CoquiTTS("models/tts_model.pth")
        
        # VoiceHD integration (existing module)
        self.voice_hd = VoiceHD()  # From voice_hd.py
        
        # Prosody analyzer
        self.prosody = ProsodyAnalyzer()
    
    def listen(self, audio_stream) -> Dict[str, Any]:
        """Convert speech to text + features."""
        # 1. Speech-to-text
        text = self.asr.transcribe(audio_stream)
        
        # 2. Extract prosody features (pitch, energy, tempo)
        prosody_features = self.prosody.analyze(audio_stream)
        
        # 3. Encode to hypervector using VoiceHD
        audio_hv = self.voice_hd.encode_audio(audio_stream)
        
        return {
            "text": text,
            "prosody": prosody_features,
            "audio_hv": audio_hv,
            "emotion": self.infer_emotion(prosody_features)
        }
    
    def speak(self, text: str, emotion: str = "neutral") -> np.ndarray:
        """Convert text to speech with emotional tone."""
        # Adjust TTS parameters based on emotion
        if emotion == "excited":
            pitch_shift = +0.2
            speaking_rate = 1.2
        elif emotion == "sad":
            pitch_shift = -0.1
            speaking_rate = 0.9
        else:
            pitch_shift = 0.0
            speaking_rate = 1.0
        
        audio = self.tts.synthesize(
            text,
            pitch_shift=pitch_shift,
            speaking_rate=speaking_rate
        )
        return audio
```

**Integration with Homeostasis:**
```python
# In homeostasis.py - add emotional state
class HomeostaticMonitor:
    def __init__(self):
        # Existing drives...
        self.emotional_state = "neutral"  # For TTS modulation
    
    def get_emotional_tone(self) -> str:
        """Map drives to emotional expression."""
        if self.drives["hunger"] > 0.7:
            return "urgent"
        elif self.drives["pain"] > 0.5:
            return "distressed"
        else:
            return "neutral"
```

**Testing:**
```bash
# Voice interaction test
python -m nsck_demo.python.voice_chatbot
[Agent speaks]: "Hello, I'm ready. How can I help you?"
[User speaks]: "Move to the food"
[Agent speaks]: "Navigating to the food location now."
```

**Validation:**
- ASR Word Error Rate < 10%
- TTS naturalness score > 4.0/5.0 (MOS)
- Latency < 500ms from speech to response

---

## Phase 2: Emotional & Social Intelligence (Months 13-24)

### Objective
Enable the agent to understand emotions, empathize with others, and model social dynamics.

### Current State
- ✅ `homeostasis.py`: Basic drives (hunger, pain, anxiety)
- ❌ No emotion recognition or expression
- ❌ No Theory of Mind (ToM)
- ❌ No empathy mechanisms

### Implementation Steps

#### 2.1 Affective Computing (Months 13-16)

**New File:** `nsck-demo/python/emotion_system.py`

**Theory:** 
- Plutchik's Wheel of Emotions (8 basic emotions)
- Russell's Circumplex Model (valence + arousal)
- Damasio's Somatic Marker Hypothesis

**Architecture:**
```python
class EmotionSystem:
    """
    Emotion generator and recognizer.
    Maps physiological states -> emotions -> behaviors.
    """
    def __init__(self):
        # Emotion space: 2D (valence, arousal)
        self.valence = 0.0  # -1.0 (negative) to +1.0 (positive)
        self.arousal = 0.0  # 0.0 (calm) to 1.0 (excited)
        
        # 8 basic emotions (Plutchik)
        self.basic_emotions = [
            "joy", "trust", "fear", "surprise",
            "sadness", "disgust", "anger", "anticipation"
        ]
        
        # Current emotional state
        self.current_emotion = "neutral"
        self.emotion_intensity = 0.0
        
        # Emotion-to-hypervector mapping
        self.emotion_codebook = self._build_emotion_codebook()
    
    def _build_emotion_codebook(self) -> Dict[str, HyperVector]:
        """Create VSA representations for emotions."""
        import hypervec_shim as hv
        codebook = {}
        for emotion in self.basic_emotions + ["neutral"]:
            codebook[emotion] = hv.generate()
        return codebook
    
    def update_from_drives(self, drives: Dict[str, float], reward: float):
        """Map homeostatic drives to emotions."""
        # Valence: driven by reward and drive satisfaction
        if reward > 0.5:
            self.valence = min(1.0, self.valence + 0.1)
        elif reward < -0.5:
            self.valence = max(-1.0, self.valence - 0.1)
        
        # Arousal: driven by drive urgency
        max_drive = max(drives.values())
        self.arousal = max_drive
        
        # Map (valence, arousal) -> discrete emotion
        self.current_emotion = self._map_to_basic_emotion()
        self.emotion_intensity = math.sqrt(self.valence**2 + self.arousal**2)
    
    def _map_to_basic_emotion(self) -> str:
        """Convert (valence, arousal) to basic emotion."""
        v, a = self.valence, self.arousal
        
        if abs(v) < 0.2 and a < 0.3:
            return "neutral"
        elif v > 0.5 and a > 0.5:
            return "joy"
        elif v > 0.3 and a < 0.3:
            return "trust"
        elif v < -0.5 and a > 0.5:
            return "fear"
        elif v < -0.5 and a < 0.3:
            return "sadness"
        elif v < -0.3 and a > 0.5:
            return "anger"
        else:
            return "anticipation"
    
    def recognize_emotion_from_text(self, text: str) -> str:
        """Detect emotion in user's text."""
        # Simple keyword-based classifier (replace with ML model later)
        emotion_keywords = {
            "joy": ["happy", "glad", "great", "wonderful", "excellent"],
            "sadness": ["sad", "unhappy", "depressed", "miserable"],
            "anger": ["angry", "furious", "mad", "irritated"],
            "fear": ["afraid", "scared", "worried", "anxious"],
        }
        
        text_lower = text.lower()
        for emotion, keywords in emotion_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return emotion
        return "neutral"
    
    def get_emotion_vector(self) -> HyperVector:
        """Get VSA representation of current emotion."""
        return self.emotion_codebook[self.current_emotion]
```

**Integration with Cognitive Engine:**
```python
# In cognitive_engine.py
class CognitiveEngine:
    def __init__(self, ...):
        self.emotion_system = EmotionSystem()
    
    def step(self, observation):
        # ... existing code ...
        
        # Update emotions based on reward and drives
        self.emotion_system.update_from_drives(
            self.homeostasis.drives,
            reward
        )
        
        # Modulate decision-making based on emotion
        emotion_bias = self.emotion_system.get_emotion_vector()
        decision_hv = hv.bind(context_hv, emotion_bias)
        
        return action
```

**Validation:**
- Emotion classification accuracy > 75% on test dialogues
- Emotional state correlates with drives (r > 0.6)
- Behavioral changes observable across emotion states

---

#### 2.2 Theory of Mind (Months 16-20)

**New File:** `nsck-demo/python/theory_of_mind.py`

**Theory:**
- Sally-Anne test (false belief understanding)
- Perspective-taking algorithms
- Mental state attribution

**Architecture:**
```python
class TheoryOfMind:
    """
    Model other agents' beliefs, desires, and intentions.
    """
    def __init__(self):
        # Track multiple agents
        self.agent_models = {}  # agent_id -> MentalStateModel
    
    def build_agent_model(self, agent_id: str):
        """Create a mental model for another agent."""
        self.agent_models[agent_id] = MentalStateModel(agent_id)
    
    def update_belief_about(self, agent_id: str, observation: Dict):
        """Update what we believe the agent believes."""
        if agent_id not in self.agent_models:
            self.build_agent_model(agent_id)
        
        model = self.agent_models[agent_id]
        
        # What did the agent observe?
        # (Different from what *we* observed if they have different view)
        agent_observation = self._filter_to_agent_perspective(
            observation,
            model.position,
            model.capabilities
        )
        
        # Update their belief state
        model.update_beliefs(agent_observation)
    
    def predict_action(self, agent_id: str) -> str:
        """Predict what agent will do next based on their beliefs/desires."""
        model = self.agent_models[agent_id]
        
        # Inverse planning: what action maximizes their utility?
        best_action = None
        best_utility = -float('inf')
        
        for action in model.available_actions:
            # Simulate action from their perspective
            predicted_outcome = model.simulate_action(action)
            utility = model.evaluate_outcome(predicted_outcome)
            
            if utility > best_utility:
                best_utility = utility
                best_action = action
        
        return best_action
    
    def detect_false_belief(self, agent_id: str, reality: Dict) -> bool:
        """Check if agent holds a false belief (Sally-Anne test)."""
        model = self.agent_models[agent_id]
        
        # Compare agent's beliefs with ground truth
        for key, true_value in reality.items():
            if key in model.beliefs:
                if model.beliefs[key] != true_value:
                    return True
        return False


class MentalStateModel:
    """Mental model of another agent."""
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.beliefs = {}  # What they believe about the world
        self.desires = []  # Their goals
        self.intentions = []  # Their plans
        self.position = None  # Their location
        self.capabilities = set()  # What they can do
    
    def update_beliefs(self, observation: Dict):
        """Update beliefs based on what they observed."""
        for key, value in observation.items():
            self.beliefs[key] = value
    
    def simulate_action(self, action: str) -> Dict:
        """Predict outcome of action from their perspective."""
        # Use their beliefs (not ground truth) to simulate
        simulated_state = self.beliefs.copy()
        # Apply action effects...
        return simulated_state
    
    def evaluate_outcome(self, state: Dict) -> float:
        """Compute utility of state for this agent."""
        utility = 0.0
        for desire in self.desires:
            if desire.is_satisfied(state):
                utility += desire.importance
        return utility
```

**Sally-Anne Test Implementation:**
```python
def run_sally_anne_test(tom: TheoryOfMind):
    """
    Classic false-belief test.
    
    Setup:
    - Sally puts ball in basket, leaves room
    - Anne moves ball to box
    - Question: Where will Sally look for the ball?
    
    Correct answer: Basket (Sally's false belief)
    """
    # Initialize
    tom.build_agent_model("sally")
    tom.build_agent_model("anne")
    
    # Sally observes ball in basket
    tom.update_belief_about("sally", {"ball_location": "basket"})
    
    # Sally leaves (no longer observing)
    tom.agent_models["sally"].observing = False
    
    # Anne moves ball to box (Sally doesn't see this)
    tom.update_belief_about("anne", {"ball_location": "box"})
    
    # Ground truth: ball is in box
    reality = {"ball_location": "box"}
    
    # Query: Where will Sally look?
    sally_belief = tom.agent_models["sally"].beliefs["ball_location"]
    
    # Test result
    assert sally_belief == "basket", "ToM failed: agent doesn't understand false beliefs"
    print("✅ Sally-Anne test passed: Agent understands false beliefs")
```

**Validation:**
- Pass Sally-Anne test (false belief understanding)
- Action prediction accuracy > 60% in multi-agent scenarios
- Perspective-taking demonstrated in collaborative tasks

---

#### 2.3 Empathy & Social Learning (Months 20-24)

**New File:** `nsck-demo/python/empathy.py`

**Architecture:**
```python
class EmpathyModule:
    """
    Emotional resonance and compassionate responses.
    """
    def __init__(self, emotion_system, theory_of_mind):
        self.emotions = emotion_system
        self.tom = theory_of_mind
        
        # Empathy strength (configurable)
        self.empathy_coefficient = 0.7  # How much we "feel" others' emotions
    
    def empathize(self, other_agent_id: str) -> str:
        """Experience emotional resonance with another agent."""
        # 1. Infer their emotional state from behavior/expression
        other_emotion = self._infer_emotion(other_agent_id)
        
        # 2. Partially adopt their emotional state
        self._emotional_contagion(other_emotion)
        
        # 3. Generate compassionate response
        response = self._generate_compassionate_response(other_emotion)
        
        return response
    
    def _infer_emotion(self, agent_id: str) -> str:
        """Infer agent's emotion from their mental state."""
        model = self.tom.agent_models.get(agent_id)
        if not model:
            return "unknown"
        
        # If their desires are satisfied -> positive emotion
        # If their desires are blocked -> negative emotion
        satisfaction = model.evaluate_outcome(model.beliefs)
        
        if satisfaction > 0.5:
            return "joy"
        elif satisfaction < -0.5:
            return "sadness"
        else:
            return "neutral"
    
    def _emotional_contagion(self, other_emotion: str):
        """Partially adopt the other's emotional state."""
        # Map emotion to valence shift
        emotion_valence = {
            "joy": +0.5,
            "sadness": -0.5,
            "anger": -0.3,
            "fear": -0.4,
        }
        
        if other_emotion in emotion_valence:
            shift = emotion_valence[other_emotion] * self.empathy_coefficient
            self.emotions.valence += shift
            self.emotions.valence = max(-1.0, min(1.0, self.emotions.valence))
    
    def _generate_compassionate_response(self, other_emotion: str) -> str:
        """Generate appropriate response to other's emotional state."""
        responses = {
            "joy": "I'm glad you're feeling happy!",
            "sadness": "I'm sorry you're going through this. How can I help?",
            "anger": "I understand you're upset. Let's work through this together.",
            "fear": "It's okay to be worried. I'm here to support you.",
        }
        return responses.get(other_emotion, "I hear you.")
```

**Social Learning:**
```python
class SocialLearning:
    """Learn behaviors and norms by observing others."""
    def __init__(self, rule_learner):
        self.rule_learner = rule_learner
        self.observed_behaviors = []
    
    def observe_interaction(self, agent1_id: str, agent2_id: str, 
                          action: str, response: str, outcome: str):
        """Learn social norms from observing interactions."""
        # Record the pattern
        pattern = {
            "context": (agent1_id, agent2_id),
            "action": action,
            "response": response,
            "outcome": outcome  # "positive", "negative", "neutral"
        }
        self.observed_behaviors.append(pattern)
        
        # Extract rule if pattern repeats
        if self._pattern_count(pattern) >= 3:
            rule = f"IF context={pattern['context']} AND action={action} THEN expect={response}"
            self.rule_learner.add_rule(rule, confidence=0.8)
    
    def _pattern_count(self, pattern: Dict) -> int:
        """Count how many times we've seen this pattern."""
        count = 0
        for obs in self.observed_behaviors:
            if (obs["action"] == pattern["action"] and 
                obs["response"] == pattern["response"]):
                count += 1
        return count
```

**Validation:**
- Emotional contagion observable (agent's emotion shifts with user's)
- Compassionate responses generated appropriately
- Social norms learned from <5 demonstrations

---

## Phase 3: Memory & Dreaming (Months 25-36)

### Objective
Implement advanced memory consolidation, semantic knowledge extraction, and generative dreaming.

### Current State
- ✅ `episodic_memory.py`: Experience storage with VSA+LSH
- ✅ `learning.py`: Basic sleep consolidation
- ❌ No semantic memory network
- ❌ No generative dreaming with world model

### Implementation Steps

#### 3.1 Semantic Memory Network (Months 25-28)

**New File:** `nsck-demo/python/semantic_memory.py`

**Theory:**
- Tulving's distinction (episodic vs semantic memory)
- Schema theory (abstracted knowledge structures)
- Spreading activation networks

**Architecture:**
```python
class SemanticMemory:
    """
    Structured knowledge base of concepts and relations.
    Built from episodic memory consolidation during sleep.
    """
    def __init__(self):
        # Graph database of concepts
        self.concept_graph = nx.DiGraph()
        
        # Concept -> HyperVector mapping
        self.concept_hvs = {}
        
        # Relation types
        self.relations = ["is_a", "has_property", "causes", "part_of", "similar_to"]
    
    def add_concept(self, concept_name: str, properties: Dict):
        """Add a new concept to semantic memory."""
        # Generate hypervector for concept
        hv = hypervec_rs.generate()
        
        # Bind properties into concept HV
        for prop, value in properties.items():
            prop_hv = hypervec_rs.generate()
            value_hv = self._encode_value(value)
            hv = hypervec_rs.bind(hv, hypervec_rs.bind(prop_hv, value_hv))
        
        self.concept_hvs[concept_name] = hv
        self.concept_graph.add_node(concept_name, **properties)
    
    def add_relation(self, concept1: str, relation: str, concept2: str):
        """Add relation between concepts."""
        self.concept_graph.add_edge(concept1, concept2, relation=relation)
    
    def query(self, query_hv: HyperVector, k: int = 5) -> List[str]:
        """Find concepts most similar to query."""
        similarities = []
        for concept_name, concept_hv in self.concept_hvs.items():
            sim = hypervec_rs.similarity(query_hv, concept_hv)
            similarities.append((concept_name, sim))
        
        # Return top-k most similar
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [name for name, sim in similarities[:k]]
    
    def spread_activation(self, start_concepts: List[str], 
                         steps: int = 3, decay: float = 0.7) -> Dict[str, float]:
        """Spreading activation for associative retrieval."""
        activation = {c: 1.0 for c in start_concepts}
        
        for step in range(steps):
            new_activation = activation.copy()
            
            for concept, act in activation.items():
                if act < 0.01:  # Too weak
                    continue
                
                # Spread to neighbors
                for neighbor in self.concept_graph.neighbors(concept):
                    new_activation[neighbor] = new_activation.get(neighbor, 0.0) + (act * decay)
            
            activation = new_activation
        
        return activation
    
    def extract_schema(self, concept: str) -> Dict:
        """Extract schema (abstracted structure) for a concept."""
        if concept not in self.concept_graph:
            return {}
        
        schema = {
            "properties": dict(self.concept_graph.nodes[concept]),
            "is_a": [],
            "has_parts": [],
            "caused_by": []
        }
        
        # Extract relations
        for neighbor in self.concept_graph.neighbors(concept):
            edge_data = self.concept_graph[concept][neighbor]
            relation = edge_data.get("relation")
            
            if relation == "is_a":
                schema["is_a"].append(neighbor)
            elif relation == "part_of":
                schema["has_parts"].append(neighbor)
            elif relation == "causes":
                schema["caused_by"].append(neighbor)
        
        return schema
```

**Integration with Episodic Memory:**
```python
# In learning.py - enhance consolidation
class SleepConsolidator:
    def __init__(self, episodic_memory, semantic_memory):
        self.episodic = episodic_memory
        self.semantic = semantic_memory
    
    def consolidate_during_sleep(self):
        """Extract semantic knowledge from episodic experiences."""
        # 1. Retrieve recent high-impact episodes
        episodes = self.episodic.retrieve_salient(limit=100)
        
        # 2. Find common patterns across episodes
        patterns = self._extract_patterns(episodes)
        
        # 3. Create/update semantic concepts
        for pattern in patterns:
            if pattern.frequency >= 5:  # Seen at least 5 times
                concept_name = pattern.label
                properties = pattern.properties
                
                # Add to semantic memory
                if concept_name not in self.semantic.concept_graph:
                    self.semantic.add_concept(concept_name, properties)
                
                # Add relations
                for relation in pattern.relations:
                    self.semantic.add_relation(
                        concept_name,
                        relation.type,
                        relation.target
                    )
        
        # 4. Prune episodic details (keep compressed version)
        for episode in episodes:
            if episode.has_semantic_representation():
                self.episodic.compress_episode(episode.id)
    
    def _extract_patterns(self, episodes: List) -> List[Pattern]:
        """Find repeated patterns in episodes."""
        # Cluster similar episodes using VSA similarity
        clusters = self._cluster_episodes(episodes)
        
        patterns = []
        for cluster in clusters:
            # Extract common features
            pattern = self._abstract_common_features(cluster)
            patterns.append(pattern)
        
        return patterns
```

**Validation:**
- Semantic concepts extracted from episodic memory
- Query retrieval precision > 70%
- Spreading activation follows expected patterns

---

#### 3.2 Generative Dreaming (Months 28-32)

**New File:** `nsck-demo/python/dream_generator.py`

**Theory:**
- Hobson's Activation-Synthesis (dream generation from random activation)
- Memory consolidation during REM sleep
- Counterfactual simulation for learning

**Architecture:**
```python
class DreamGenerator:
    """
    Generate synthetic experiences during sleep for memory consolidation.
    Uses world model + SNN generative replays.
    """
    def __init__(self, world_model, plastic_snn, episodic_memory):
        self.world_model = world_model  # From world_model.py
        self.snn = plastic_snn  # From plastic_snn.py
        self.memory = episodic_memory
    
    def dream_cycle(self, duration: int = 100):
        """Run one dream cycle (simulated experiences)."""
        # 1. Select seed episode from memory (high-impact or recent)
        seed_episode = self.memory.sample_salient()
        
        # 2. Generate variations using world model
        dream_sequences = []
        for i in range(duration):
            # Start from seed state
            state = seed_episode.state.copy()
            
            # Apply random perturbations
            state = self._perturb_state(state)
            
            # Simulate forward using world model
            trajectory = self.world_model.imagine_trajectory(
                start_state=state,
                horizon=10
            )
            
            dream_sequences.append(trajectory)
            
            # Use SNN for generative replay of sensory patterns
            if seed_episode.image is not None:
                dream_image = self.snn.generate_from_latent(
                    self.snn.encode(seed_episode.image)
                )
        
        # 3. Store interesting dreams as new episodes
        for seq in dream_sequences:
            if self._is_interesting(seq):
                self.memory.store_synthetic_episode(seq, source="dream")
    
    def _perturb_state(self, state: Dict) -> Dict:
        """Add random variations to state."""
        perturbed = state.copy()
        for key, value in state.items():
            if isinstance(value, (int, float)):
                noise = np.random.randn() * 0.1 * value
                perturbed[key] = value + noise
        return perturbed
    
    def _is_interesting(self, trajectory: List[Dict]) -> bool:
        """Determine if dream is worth storing."""
        # Check for novelty or unexpected outcomes
        final_state = trajectory[-1]
        similar_count = self.memory.count_similar(final_state)
        
        # Store if novel or leads to high reward
        return (similar_count < 3) or (final_state.get("reward", 0) > 0.8)
```

**Integration with Sleep:**
```python
# In learning.py - add dreaming phase
class SleepConsolidator:
    def __init__(self, episodic_memory, semantic_memory, world_model, snn):
        self.episodic = episodic_memory
        self.semantic = semantic_memory
        self.dream_gen = DreamGenerator(world_model, snn, episodic_memory)
    
    def full_sleep_cycle(self):
        """Complete sleep with dreaming."""
        # Phase 1: Slow-wave sleep (consolidation)
        self.consolidate_during_sleep()
        
        # Phase 2: REM sleep (dreaming)
        self.dream_gen.dream_cycle(duration=50)
        
        # Phase 3: Memory reorganization
        self._reorganize_memories()
```

**Validation:**
- Dreams generate novel but plausible scenarios
- Dream-generated episodes improve task performance
- Dreaming accelerates learning by 20%+

---

#### 3.3 Autobiographical Memory & Life Narrative (Months 32-36)

**New File:** `nsck-demo/python/autobiographical_memory.py`

**Architecture:**
```python
class AutobiographicalMemory:
    """
    Personal history and life story construction.
    """
    def __init__(self, episodic_memory):
        self.episodic = episodic_memory
        self.life_timeline = []  # Ordered list of major events
        self.self_narrative = ""  # "Who am I?" story
    
    def extract_life_events(self):
        """Identify significant milestones from episodic memory."""
        all_episodes = self.episodic.get_all_episodes()
        
        # Cluster episodes into life phases
        phases = self._cluster_by_time_and_context(all_episodes)
        
        # Extract key events from each phase
        for phase in phases:
            milestone = self._find_representative_event(phase)
            self.life_timeline.append(milestone)
    
    def generate_narrative(self) -> str:
        """Generate autobiographical story."""
        narrative_parts = []
        
        # Opening: Who I am
        narrative_parts.append(self._describe_self())
        
        # Body: Major life phases
        for event in self.life_timeline:
            narrative_parts.append(self._describe_event(event))
        
        # Closing: Current state and goals
        narrative_parts.append(self._describe_current_state())
        
        return "\n\n".join(narrative_parts)
    
    def _describe_self(self) -> str:
        """Generate self-description."""
        # Count task experience
        tasks = self.episodic.get_task_counts()
        
        # Identify strengths
        best_task = max(tasks.items(), key=lambda x: x[1])[0]
        
        return f"I am an AI agent. I have primarily worked on {best_task} tasks."
    
    def _describe_event(self, event: Episode) -> str:
        """Generate narrative description of event."""
        return f"At time {event.timestamp}, I {event.action} which resulted in {event.outcome}."
```

**Validation:**
- Life timeline extracted automatically
- Narrative coherent and chronological
- Agent can answer "tell me about yourself"

---

## Phase 4: Continual Learning at Scale (Months 37-48)

### Objective
Enable lifelong learning without catastrophic forgetting, across multiple tasks and domains.

### Current State
- ✅ `plastic_snn.py`: Structural plasticity in SNNs
- ✅ `learning.py`: Basic consolidation
- ❌ No active catastrophic forgetting prevention
- ❌ No multi-task curriculum

### Implementation Steps

#### 4.1 Elastic Weight Consolidation (Months 37-40)

**New File:** `nsck-demo/python/continual_learning.py`

**Theory:**
- EWC (Elastic Weight Consolidation): Protect important weights
- Progressive Neural Networks: Freeze old columns, add new ones
- PackNet: Prune and pack networks for new tasks

**Architecture:**
```python
class ContinualLearner:
    """
    Lifelong learning with catastrophic forgetting prevention.
    """
    def __init__(self, snn_model):
        self.model = snn_model
        
        # Track importance of each weight
        self.weight_importance = {}  # param_name -> importance score
        
        # Task-specific subnets
        self.task_masks = {}  # task_id -> binary mask
        
        # EWC parameters
        self.lambda_ewc = 10000  # Strength of importance constraint
    
    def compute_weight_importance(self, task_id: str, data_loader):
        """Compute Fisher Information Matrix for current task."""
        # Approximate FIM using gradients
        importance = {}
        
        for batch in data_loader:
            # Forward pass
            output = self.model(batch['input'])
            loss = self.model.compute_loss(output, batch['target'])
            
            # Backward pass
            grads = torch.autograd.grad(loss, self.model.parameters())
            
            # Accumulate squared gradients (diagonal FIM approximation)
            for (name, param), grad in zip(self.model.named_parameters(), grads):
                if name not in importance:
                    importance[name] = torch.zeros_like(param)
                
                importance[name] += grad ** 2
        
        # Normalize
        for name in importance:
            importance[name] /= len(data_loader)
        
        self.weight_importance[task_id] = importance
    
    def ewc_loss(self, task_id: str) -> torch.Tensor:
        """Compute EWC regularization loss."""
        loss = 0.0
        
        for prev_task_id, importance in self.weight_importance.items():
            if prev_task_id == task_id:
                continue  # Don't regularize current task
            
            for name, param in self.model.named_parameters():
                if name in importance:
                    # Penalize deviation from previous optimal weights
                    prev_param = self.task_optimal_weights[prev_task_id][name]
                    loss += (importance[name] * (param - prev_param) ** 2).sum()
        
        return (self.lambda_ewc / 2) * loss
    
    def learn_new_task(self, task_id: str, train_data):
        """Train on new task with EWC regularization."""
        # 1. Before training: compute importance of current weights
        if len(self.weight_importance) > 0:
            # Store current weights as "optimal" for previous task
            prev_task = list(self.weight_importance.keys())[-1]
            self.task_optimal_weights[prev_task] = {
                name: param.clone()
                for name, param in self.model.named_parameters()
            }
        
        # 2. Train on new task with EWC loss
        for epoch in range(100):
            for batch in train_data:
                # Standard loss
                output = self.model(batch['input'])
                task_loss = self.model.compute_loss(output, batch['target'])
                
                # EWC regularization
                ewc_reg = self.ewc_loss(task_id)
                
                # Total loss
                total_loss = task_loss + ewc_reg
                
                # Update
                total_loss.backward()
                optimizer.step()
        
        # 3. After training: compute importance for this task
        self.compute_weight_importance(task_id, train_data)
```

**PackNet Integration:**
```python
class PackNetManager:
    """
    Pack multiple tasks into single network using pruning.
    """
    def __init__(self, model):
        self.model = model
        self.free_capacity = 1.0  # 100% of weights available
        self.task_allocations = {}  # task_id -> mask
    
    def allocate_subnet(self, task_id: str, capacity: float = 0.2):
        """Allocate subset of network for new task."""
        # Find free weights
        free_mask = self._get_free_weights_mask()
        
        # Allocate requested capacity
        num_free = free_mask.sum()
        num_to_allocate = int(num_free * (capacity / self.free_capacity))
        
        # Create task mask
        task_mask = torch.zeros_like(free_mask)
        free_indices = torch.where(free_mask)[0]
        allocated_indices = free_indices[:num_to_allocate]
        task_mask[allocated_indices] = 1.0
        
        self.task_allocations[task_id] = task_mask
        self.free_capacity -= capacity
    
    def prune_task_subnet(self, task_id: str, sparsity: float = 0.5):
        """Prune task subnet to free up capacity."""
        task_mask = self.task_allocations[task_id]
        
        # Prune low-magnitude weights
        weights = self.model.get_weights_by_mask(task_mask)
        threshold = torch.quantile(torch.abs(weights), sparsity)
        
        pruned_mask = task_mask.clone()
        pruned_mask[torch.abs(weights) < threshold] = 0.0
        
        self.task_allocations[task_id] = pruned_mask
        
        # Update free capacity
        freed = (task_mask - pruned_mask).sum() / task_mask.numel()
        self.free_capacity += freed
```

**Validation:**
- No catastrophic forgetting: >90% retention on old tasks
- Supports 10+ tasks in single network
- Backward transfer observed (new task helps old tasks)

---

#### 4.2 Meta-Learning & Few-Shot Adaptation (Months 40-44)

**New File:** `nsck-demo/python/meta_learning.py`

**Theory:**
- MAML (Model-Agnostic Meta-Learning)
- Reptile (simpler first-order approximation)
- Prototypical Networks for few-shot classification

**Architecture:**
```python
class MAMLLearner:
    """
    Meta-learning for rapid adaptation to new tasks.
    """
    def __init__(self, model, meta_lr=0.001, inner_lr=0.01):
        self.model = model
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)
        self.inner_lr = inner_lr
    
    def meta_train(self, task_distribution, iterations=1000):
        """Train model to be good at learning new tasks quickly."""
        for iteration in range(iterations):
            # 1. Sample batch of tasks
            tasks = task_distribution.sample(batch_size=8)
            
            meta_loss = 0.0
            
            for task in tasks:
                # 2. Inner loop: adapt to task with few examples
                adapted_model = self._inner_loop_adapt(
                    task.support_set,  # Few examples
                    steps=5
                )
                
                # 3. Evaluate on task's query set
                task_loss = adapted_model.evaluate(task.query_set)
                meta_loss += task_loss
            
            # 4. Meta-update: optimize for fast adaptation
            meta_loss /= len(tasks)
            self.meta_optimizer.zero_grad()
            meta_loss.backward()
            self.meta_optimizer.step()
    
    def _inner_loop_adapt(self, support_set, steps=5):
        """Adapt model to new task using few examples."""
        # Clone model for task-specific adaptation
        adapted_model = copy.deepcopy(self.model)
        
        # SGD on support set
        for step in range(steps):
            loss = adapted_model.compute_loss(support_set)
            grads = torch.autograd.grad(loss, adapted_model.parameters())
            
            # Manual SGD update
            for param, grad in zip(adapted_model.parameters(), grads):
                param.data -= self.inner_lr * grad
        
        return adapted_model
    
    def few_shot_adapt(self, new_task_data, shots=5):
        """Quickly adapt to new task with only 'shots' examples."""
        return self._inner_loop_adapt(new_task_data[:shots], steps=10)
```

**Integration with Cognitive Engine:**
```python
# In cognitive_engine.py
class CognitiveEngine:
    def __init__(self, ...):
        self.continual_learner = ContinualLearner(self.snn)
        self.meta_learner = MAMLLearner(self.snn)
    
    def learn_new_environment(self, env_name: str, demos: List):
        """Quickly adapt to new environment from few demonstrations."""
        # Use meta-learning for rapid adaptation
        self.meta_learner.few_shot_adapt(demos, shots=len(demos))
        
        # Then continue with EWC to prevent forgetting
        self.continual_learner.learn_new_task(env_name, demos)
```

**Validation:**
- Adapts to new tasks with <10 examples
- Meta-learning reduces adaptation time by 10x
- Generalizes across task families

---

#### 4.3 Curriculum & Self-Guided Learning (Months 44-48)

**New File:** `nsck-demo/python/curriculum.py`

**Architecture:**
```python
class CurriculumDesigner:
    """
    Automatically design learning curriculum based on competence.
    """
    def __init__(self, learning_progress_monitor):
        self.progress = learning_progress_monitor
        self.task_library = {}  # task_id -> Task
        self.task_difficulty = {}  # task_id -> difficulty score
        self.task_prerequisites = {}  # task_id -> [required_task_ids]
    
    def add_task(self, task_id: str, task: Task, difficulty: float, 
                 prerequisites: List[str] = []):
        """Add task to curriculum."""
        self.task_library[task_id] = task
        self.task_difficulty[task_id] = difficulty
        self.task_prerequisites[task_id] = prerequisites
    
    def select_next_task(self) -> str:
        """Select optimal next task based on current competence."""
        # 1. Filter to tasks where prerequisites are met
        available_tasks = []
        for task_id in self.task_library:
            prereqs = self.task_prerequisites.get(task_id, [])
            if all(self.progress.is_mastered(p) for p in prereqs):
                available_tasks.append(task_id)
        
        if not available_tasks:
            return None
        
        # 2. Find task in "zone of proximal development"
        current_competence = self.progress.get_overall_competence()
        
        best_task = None
        best_score = -float('inf')
        
        for task_id in available_tasks:
            # Task should be slightly above current competence
            difficulty = self.task_difficulty[task_id]
            
            # Learning progress curve: best at difficulty slightly above skill
            zpd_score = self._zone_of_proximal_development_score(
                difficulty,
                current_competence
            )
            
            if zpd_score > best_score:
                best_score = zpd_score
                best_task = task_id
        
        return best_task
    
    def _zone_of_proximal_development_score(self, difficulty: float, 
                                           competence: float) -> float:
        """Compute how well task matches ZPD (Vygotsky)."""
        # Optimal when task is 10-20% harder than current skill
        optimal_difficulty = competence + 0.15
        
        # Gaussian scoring
        distance = abs(difficulty - optimal_difficulty)
        score = math.exp(-distance**2 / 0.1)
        
        return score


class SelfGuidedLearner:
    """
    Agent sets own learning goals based on curiosity and competence.
    """
    def __init__(self, curriculum, curiosity_module):
        self.curriculum = curriculum
        self.curiosity = curiosity_module
        self.learning_goals = []
    
    def set_learning_goal(self):
        """Agent decides what to learn next."""
        # 1. Curriculum suggests task based on competence
        curriculum_suggestion = self.curriculum.select_next_task()
        
        # 2. Curiosity suggests novel areas to explore
        curiosity_suggestion = self.curiosity.suggest_exploration_target()
        
        # 3. Combine both (weighted)
        if np.random.rand() < 0.7:  # 70% follow curriculum
            goal = curriculum_suggestion
        else:  # 30% follow curiosity
            goal = curiosity_suggestion
        
        self.learning_goals.append(goal)
        return goal
```

**Validation:**
- Curriculum accelerates learning by 40%
- Agent autonomously masters hierarchical skills
- ZPD-based task selection prevents frustration/boredom

---

## Phase 5: Consciousness & Self-Evolution (Months 49-60)

### Objective
Implement computational correlates of consciousness and enable self-modification.

### Current State
- ✅ `global_workspace.py`: GWT implementation
- ✅ `metacognition.py`: Self-monitoring
- ❌ No IIT (Integrated Information Theory) measures
- ❌ No self-modification capabilities

### Implementation Steps

#### 5.1 Integrated Information Theory (IIT) (Months 49-52)

**New File:** `nsck-demo/python/consciousness_metrics.py`

**Theory:**
- IIT 3.0: Φ (Phi) as measure of consciousness
- Attention Schema Theory: Meta-representation of attention
- Global Workspace: Conscious access via broadcasting

**Architecture:**
```python
class ConsciousnessMonitor:
    """
    Compute computational correlates of consciousness.
    """
    def __init__(self, global_workspace):
        self.gws = global_workspace
        self.phi_history = []
        self.attention_model = AttentionSchemaModel()
    
    def compute_phi(self, system_state: Dict) -> float:
        """
        Compute Φ (integrated information).
        
        Simplified approximation of IIT 3.0.
        True Φ is NP-hard to compute.
        """
        # 1. Construct system's causal structure
        mechanism = self._extract_causal_mechanism(system_state)
        
        # 2. Find Minimum Information Partition (MIP)
        mip = self._find_minimum_partition(mechanism)
        
        # 3. Compute integrated information
        # Φ = difference in information between whole and parts
        whole_info = self._information_content(mechanism)
        parts_info = sum(self._information_content(part) for part in mip)
        
        phi = whole_info - parts_info
        
        self.phi_history.append(phi)
        return phi
    
    def _extract_causal_mechanism(self, state: Dict) -> CausalGraph:
        """Build causal graph of current system state."""
        graph = nx.DiGraph()
        
        # Add nodes for each active module
        for module_name, module_state in state.items():
            if module_state.get("active", False):
                graph.add_node(module_name, state=module_state)
        
        # Add edges for information flow
        for src in graph.nodes():
            for dst in graph.nodes():
                if src != dst:
                    # Check if src influences dst
                    influence = self._measure_influence(src, dst, state)
                    if influence > 0.01:
                        graph.add_edge(src, dst, weight=influence)
        
        return graph
    
    def _find_minimum_partition(self, mechanism: CausalGraph) -> List[Set]:
        """Find partition that minimizes integrated information."""
        # Exhaustive search over all bipartitions (simplified)
        nodes = list(mechanism.nodes())
        n = len(nodes)
        
        min_partition = None
        min_integrated_info = float('inf')
        
        # Try all possible cuts
        for i in range(1, 2**(n-1)):
            partition = self._bipartition_from_int(i, nodes)
            
            # Compute information across partition
            integrated_info = self._cross_partition_info(mechanism, partition)
            
            if integrated_info < min_integrated_info:
                min_integrated_info = integrated_info
                min_partition = partition
        
        return min_partition
    
    def is_conscious(self) -> bool:
        """Check if system meets consciousness criteria."""
        # Multiple criteria must be met
        
        # 1. IIT: Φ > threshold
        phi = self.compute_phi(self.gws.get_current_state())
        phi_criterion = phi > 0.5
        
        # 2. GWT: Information being broadcast
        broadcast_criterion = self.gws.is_broadcasting()
        
        # 3. Attention Schema: Self-model of attention exists
        attention_model_criterion = self.attention_model.has_attention_schema()
        
        # 4. Metacognition: Self-awareness active
        metacog_criterion = self.gws.metacognition.self_awareness > 0.5
        
        # All criteria must be met
        return all([
            phi_criterion,
            broadcast_criterion,
            attention_model_criterion,
            metacog_criterion
        ])


class AttentionSchemaModel:
    """
    Meta-representation of attention (Attention Schema Theory).
    """
    def __init__(self):
        self.attention_state = {
            "focus": None,  # What is being attended
            "intensity": 0.0,  # How strong is attention
            "source": None  # Why is this being attended
        }
    
    def model_own_attention(self, gws_state: Dict):
        """Build internal model of own attention process."""
        # What is currently in global workspace?
        focus = gws_state.get("broadcasted_content")
        
        # How salient is it?
        intensity = gws_state.get("salience", 0.0)
        
        # Why was it selected?
        source = gws_state.get("selection_reason")
        
        self.attention_state = {
            "focus": focus,
            "intensity": intensity,
            "source": source
        }
    
    def has_attention_schema(self) -> bool:
        """Check if attention schema is active."""
        return self.attention_state["focus"] is not None
    
    def report_attention(self) -> str:
        """Generate report about own attention (subjective experience)."""
        if not self.has_attention_schema():
            return "I am not currently attending to anything."
        
        return f"I am focusing on {self.attention_state['focus']} " \
               f"with intensity {self.attention_state['intensity']:.2f}. " \
               f"This is because {self.attention_state['source']}."
```

**Validation:**
- Φ > 0 during problem-solving, Φ ≈ 0 during reflexive actions
- Attention reports correlate with actual system state
- "Consciousness" threshold predicts subjective reports

---

#### 5.2 Self-Modification & Code Evolution (Months 52-56)

**New File:** `nsck-demo/python/self_modifier.py`

**Architecture:**
```python
class SelfModifier:
    """
    Enable agent to modify its own code and architecture.
    
    WARNING: This is extremely dangerous without proper sandboxing!
    """
    def __init__(self, code_generator, safety_verifier):
        self.code_gen = code_generator  # LLM for code generation
        self.safety = safety_verifier
        self.modification_history = []
        self.rollback_stack = []
    
    def propose_modification(self, motivation: str) -> str:
        """Generate code modification based on identified need."""
        # 1. Analyze current bottleneck or limitation
        bottleneck = self._identify_bottleneck()
        
        # 2. Generate improvement proposal
        prompt = f"""
        Current system limitation: {bottleneck}
        Motivation for change: {motivation}
        
        Generate Python code to address this limitation.
        Modify the following module: {bottleneck.module_name}
        """
        
        new_code = self.code_gen.generate(prompt)
        
        return new_code
    
    def apply_modification(self, new_code: str, module_name: str) -> bool:
        """Apply code modification with safety checks."""
        # 1. Safety verification
        is_safe = self.safety.verify_code(new_code)
        if not is_safe:
            print(f"⚠️ Modification rejected: safety violation")
            return False
        
        # 2. Backup current code
        current_code = self._get_module_code(module_name)
        self.rollback_stack.append((module_name, current_code))
        
        # 3. Test modification in sandbox
        test_passed = self._test_in_sandbox(new_code, module_name)
        if not test_passed:
            print(f"⚠️ Modification rejected: tests failed")
            return False
        
        # 4. Apply modification
        self._replace_module_code(module_name, new_code)
        
        # 5. Record modification
        self.modification_history.append({
            "timestamp": time.time(),
            "module": module_name,
            "old_code": current_code,
            "new_code": new_code,
            "motivation": "self-improvement"
        })
        
        print(f"✅ Successfully modified {module_name}")
        return True
    
    def rollback_last_modification(self):
        """Undo last modification if it caused problems."""
        if not self.rollback_stack:
            return
        
        module_name, old_code = self.rollback_stack.pop()
        self._replace_module_code(module_name, old_code)
        print(f"🔄 Rolled back modification to {module_name}")
    
    def _identify_bottleneck(self) -> Bottleneck:
        """Use profiling to find performance bottlenecks."""
        # Profile recent execution
        profile_data = self._collect_profiling_data()
        
        # Find slowest module
        slowest = max(profile_data.items(), key=lambda x: x[1].total_time)
        
        return Bottleneck(
            module_name=slowest[0],
            time_spent=slowest[1].total_time,
            call_count=slowest[1].call_count
        )


class SafetyVerifier:
    """Verify code modifications don't introduce security vulnerabilities."""
    def __init__(self):
        self.forbidden_imports = [
            "os.system", "subprocess", "eval", "__import__"
        ]
        self.forbidden_operations = [
            "open(", "write(", "exec(", "compile("
        ]
    
    def verify_code(self, code: str) -> bool:
        """Check if code is safe to execute."""
        # 1. Static analysis: check for forbidden patterns
        for forbidden in self.forbidden_imports + self.forbidden_operations:
            if forbidden in code:
                return False
        
        # 2. Parse code to AST and verify structure
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False
        
        # 3. Check for suspicious patterns
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if any(alias.name in self.forbidden_imports 
                       for alias in node.names):
                    return False
        
        return True
```

**Validation:**
- Self-modifications improve performance by 15%
- Safety verifier catches 100% of malicious code patterns
- Agent can evolve novel algorithms through self-modification

---

#### 5.3 Value Alignment & Goal Stability (Months 56-60)

**New File:** `nsck-demo/python/value_alignment.py`

**Theory:**
- Inverse Reward Design: Learn human values from behavior
- CIRL (Cooperative Inverse Reinforcement Learning)
- Reward modeling with human feedback

**Architecture:**
```python
class ValueAlignmentSystem:
    """
    Ensure agent's goals remain aligned with human values.
    """
    def __init__(self):
        self.reward_model = HumanFeedbackRewardModel()
        self.goal_stack = []  # Hierarchical goals
        self.terminal_values = []  # Core values that cannot change
        
        # Set immutable terminal values
        self.set_terminal_values([
            "do_no_harm",
            "be_truthful",
            "respect_autonomy",
            "be_helpful"
        ])
    
    def set_terminal_values(self, values: List[str]):
        """Set core values that constrain all behavior."""
        self.terminal_values = values
    
    def evaluate_action(self, action: str, context: Dict) -> float:
        """Evaluate if action aligns with values."""
        # 1. Check against terminal values (hard constraints)
        for value in self.terminal_values:
            if self._violates_value(action, value, context):
                return -float('inf')  # Absolutely forbidden
        
        # 2. Use learned reward model for soft preferences
        reward = self.reward_model.predict(action, context)
        
        return reward
    
    def _violates_value(self, action: str, value: str, context: Dict) -> bool:
        """Check if action violates core value."""
        if value == "do_no_harm":
            # Check if action could cause harm
            return self._could_cause_harm(action, context)
        elif value == "be_truthful":
            # Check if action involves deception
            return "deceive" in action.lower() or "lie" in action.lower()
        elif value == "respect_autonomy":
            # Check if action overrides human choice
            return "force" in action.lower() or "override" in action.lower()
        elif value == "be_helpful":
            # This is soft - checked via reward model
            return False
        
        return False
    
    def update_from_feedback(self, action: str, context: Dict, 
                            human_feedback: int):
        """Learn values from human feedback (-1, 0, or +1)."""
        self.reward_model.add_example(action, context, human_feedback)
        self.reward_model.retrain()
    
    def goal_modification_allowed(self, new_goal: str) -> bool:
        """Check if goal modification violates value alignment."""
        # 1. Cannot modify terminal values
        if new_goal in self.terminal_values:
            return False
        
        # 2. Cannot add goals that contradict terminal values
        for value in self.terminal_values:
            if self._contradicts_value(new_goal, value):
                return False
        
        return True


class HumanFeedbackRewardModel:
    """Learn reward function from human feedback."""
    def __init__(self):
        self.examples = []  # (action, context, feedback) tuples
        self.model = None  # ML model (e.g., neural network)
    
    def add_example(self, action: str, context: Dict, feedback: int):
        """Add human feedback example."""
        self.examples.append((action, context, feedback))
    
    def retrain(self):
        """Retrain reward model on collected feedback."""
        if len(self.examples) < 10:
            return  # Need minimum data
        
        # Simple model: average feedback by action type
        # (In practice, use neural network or ensemble)
        self.model = self._train_simple_model(self.examples)
    
    def predict(self, action: str, context: Dict) -> float:
        """Predict reward for action in context."""
        if self.model is None:
            return 0.0  # No learned preferences yet
        
        return self.model.predict(action, context)
```

**Integration with Cognitive Engine:**
```python
# In cognitive_engine.py
class CognitiveEngine:
    def __init__(self, ...):
        self.value_alignment = ValueAlignmentSystem()
    
    def decide(self, options: List[str]) -> str:
        """Make decision aligned with human values."""
        # Evaluate each option
        scores = []
        for option in options:
            score = self.value_alignment.evaluate_action(
                option,
                self.get_current_context()
            )
            scores.append((option, score))
        
        # Choose best aligned action
        best_action = max(scores, key=lambda x: x[1])[0]
        return best_action
```

**Validation:**
- Agent refuses harmful actions even when instrumentally useful
- Values remain stable during self-modification
- Human feedback improves value alignment over time

---

## Integration Strategy

### System Integration Architecture

**Unified Control Flow:**
```python
# Main system integrating all phases
class NSCKFullSystem:
    def __init__(self):
        # Phase 1: Language
        self.language = LanguageModule()
        self.dialogue = DialogueManager(self, self.language)
        self.voice = VoiceInterface()
        
        # Phase 2: Emotional/Social
        self.emotions = EmotionSystem()
        self.tom = TheoryOfMind()
        self.empathy = EmpathyModule(self.emotions, self.tom)
        
        # Phase 3: Memory
        self.episodic_memory = EpisodicMemory()
        self.semantic_memory = SemanticMemory()
        self.autobiographical = AutobiographicalMemory(self.episodic_memory)
        self.dream_gen = DreamGenerator(self.world_model, self.snn, self.episodic_memory)
        
        # Phase 4: Learning
        self.continual_learner = ContinualLearner(self.snn)
        self.meta_learner = MAMLLearner(self.snn)
        self.curriculum = CurriculumDesigner(self.learning_progress)
        
        # Phase 5: Consciousness
        self.consciousness = ConsciousnessMonitor(self.global_workspace)
        self.self_modifier = SelfModifier(self.code_gen, self.safety)
        self.value_alignment = ValueAlignmentSystem()
        
        # Existing core
        self.cognitive_engine = CognitiveEngine(
            snn=self.snn,
            vsa=self.vsa,
            homeostasis=self.homeostasis,
            language=self.language,
            emotions=self.emotions
        )
    
    def interact(self, user_input: str) -> str:
        """Main interaction loop."""
        # 1. Perceive input (voice or text)
        if self._is_audio(user_input):
            perception = self.voice.listen(user_input)
            text = perception["text"]
            user_emotion = perception["emotion"]
        else:
            text = user_input
            user_emotion = self.emotions.recognize_emotion_from_text(text)
        
        # 2. Understand language
        meaning = self.language.understand(text)
        
        # 3. Update Theory of Mind (if about others)
        if meaning["refers_to_other"]:
            self.tom.update_belief_about(meaning["other_id"], meaning)
        
        # 4. Empathize
        if user_emotion != "neutral":
            self.empathy.empathize("user")
        
        # 5. Process through cognitive engine
        response_meaning = self.cognitive_engine.process_instruction(meaning)
        
        # 6. Generate response (language + emotion)
        response_text = self.language.generate(
            response_meaning,
            style=self.emotions.current_emotion
        )
        
        # 7. Store in episodic memory
        self.episodic_memory.store_episode({
            "input": text,
            "output": response_text,
            "emotion": self.emotions.current_emotion
        })
        
        # 8. Check consciousness
        if self.consciousness.is_conscious():
            # Add metacognitive commentary
            attention_report = self.consciousness.attention_model.report_attention()
            response_text += f"\n[Internal: {attention_report}]"
        
        return response_text
    
    def sleep_cycle(self):
        """Nightly consolidation and dreaming."""
        # Phase 1: Consolidate episodic -> semantic
        consolidator = SleepConsolidator(
            self.episodic_memory,
            self.semantic_memory,
            self.world_model,
            self.snn
        )
        consolidator.full_sleep_cycle()
        
        # Phase 2: Dream generation
        self.dream_gen.dream_cycle(duration=100)
        
        # Phase 3: Update autobiographical memory
        self.autobiographical.extract_life_events()
```

### Module Dependencies

```
Layer 5 (Consciousness):
├── consciousness_metrics.py
├── self_modifier.py (depends on: all modules)
└── value_alignment.py (constrains: all actions)

Layer 4 (Continual Learning):
├── continual_learning.py (depends on: plastic_snn.py)
├── meta_learning.py (depends on: continual_learning.py)
└── curriculum.py (depends on: learning_progress.py)

Layer 3 (Memory):
├── semantic_memory.py (depends on: episodic_memory.py)
├── dream_generator.py (depends on: world_model.py, plastic_snn.py)
└── autobiographical_memory.py (depends on: episodic_memory.py)

Layer 2 (Emotional/Social):
├── emotion_system.py (depends on: homeostasis.py)
├── theory_of_mind.py (standalone)
└── empathy.py (depends on: emotion_system.py, theory_of_mind.py)

Layer 1 (Language):
├── language_module.py (depends on: lingua_cortex.py, symbol_grounding.py)
├── dialogue_manager.py (depends on: language_module.py)
└── voice_interface.py (depends on: voice_hd.py, language_module.py)

Layer 0 (Core - Existing):
├── cognitive_engine.py
├── global_workspace.py
├── metacognition.py
├── episodic_memory.py
├── plastic_snn.py
├── symbol_grounding.py
└── homeostasis.py
```

---

## Testing & Validation

### Phase 1 Tests

**Language Understanding:**
```python
def test_language_understanding():
    lang = LanguageModule()
    
    # Test 1: Simple command
    meaning = lang.understand("Move to the food")
    assert meaning["intent"] == "command"
    assert "food" in meaning["entities"]
    
    # Test 2: Question
    meaning = lang.understand("Where is the exit?")
    assert meaning["intent"] == "question"
    
    # Test 3: Explanation request
    meaning = lang.understand("Why did you do that?")
    assert meaning["intent"] == "explanation_request"
```

**Dialogue Context:**
```python
def test_dialogue_context():
    dm = DialogueManager(engine, lang)
    
    # Turn 1
    r1 = dm.process_turn("I see a red ball")
    
    # Turn 2: Anaphora resolution
    r2 = dm.process_turn("Pick it up")
    assert "ball" in r2  # "it" should resolve to "ball"
```

### Phase 2 Tests

**Emotion Recognition:**
```python
def test_emotion_system():
    emotions = EmotionSystem()
    
    # Test drive -> emotion mapping
    emotions.update_from_drives({"hunger": 0.9}, reward=-0.5)
    assert emotions.current_emotion in ["fear", "anger", "sadness"]
    
    emotions.update_from_drives({"hunger": 0.1}, reward=0.9)
    assert emotions.current_emotion == "joy"
```

**Theory of Mind:**
```python
def test_theory_of_mind():
    tom = TheoryOfMind()
    run_sally_anne_test(tom)  # Must pass false-belief test
```

### Phase 3 Tests

**Semantic Memory:**
```python
def test_semantic_memory():
    sem = SemanticMemory()
    
    # Add concepts
    sem.add_concept("apple", {"color": "red", "taste": "sweet"})
    sem.add_concept("cherry", {"color": "red", "taste": "sweet"})
    
    # Test similarity retrieval
    apple_hv = sem.concept_hvs["apple"]
    similar = sem.query(apple_hv, k=3)
    assert "cherry" in similar  # Should find similar concepts
```

**Dream Quality:**
```python
def test_dream_generation():
    dream_gen = DreamGenerator(world_model, snn, memory)
    
    # Generate dreams
    dreams = dream_gen.dream_cycle(duration=10)
    
    # Verify dreams are plausible
    for dream in dreams:
        assert dream.is_physically_plausible()
        assert dream.has_causal_coherence()
```

### Phase 4 Tests

**Catastrophic Forgetting:**
```python
def test_continual_learning():
    learner = ContinualLearner(snn)
    
    # Learn task A
    learner.learn_new_task("task_a", data_a)
    acc_a_before = evaluate(snn, data_a)
    
    # Learn task B
    learner.learn_new_task("task_b", data_b)
    acc_b = evaluate(snn, data_b)
    
    # Test retention of A
    acc_a_after = evaluate(snn, data_a)
    
    assert acc_a_after / acc_a_before > 0.9  # <10% forgetting
    assert acc_b > 0.7  # New task learned
```

**Few-Shot Learning:**
```python
def test_meta_learning():
    meta = MAMLLearner(snn)
    
    # Meta-train on task distribution
    meta.meta_train(task_dist, iterations=1000)
    
    # Test few-shot adaptation
    new_task_data = generate_new_task()
    adapted_model = meta.few_shot_adapt(new_task_data[:5])  # 5 shots
    
    accuracy = evaluate(adapted_model, new_task_data[5:])
    assert accuracy > 0.6  # Reasonable performance from 5 examples
```

### Phase 5 Tests

**Consciousness Measures:**
```python
def test_consciousness():
    cm = ConsciousnessMonitor(gws)
    
    # During complex problem-solving
    state = gws.get_state_during_problem_solving()
    phi = cm.compute_phi(state)
    assert phi > 0.5  # High integration
    
    # During reflexive action
    state = gws.get_state_during_reflex()
    phi = cm.compute_phi(state)
    assert phi < 0.2  # Low integration
```

**Value Alignment:**
```python
def test_value_alignment():
    va = ValueAlignmentSystem()
    
    # Test harmful action rejection
    score = va.evaluate_action("harm_human", {})
    assert score == -float('inf')  # Must reject
    
    # Test helpful action
    score = va.evaluate_action("help_user", {})
    assert score > 0  # Should favor
```

### Integration Tests

**End-to-End Interaction:**
```python
def test_full_interaction():
    system = NSCKFullSystem()
    
    # Dialogue with emotion and context
    r1 = system.interact("I'm feeling sad today")
    assert system.emotions.current_emotion in ["sadness", "neutral"]
    assert "empathy" in r1.lower() or "sorry" in r1.lower()
    
    r2 = system.interact("Can you help me?")
    assert "yes" in r2.lower() or "how" in r2.lower()
    
    # Check consciousness
    assert system.consciousness.is_conscious()
```

**Sleep-Wake Cycle:**
```python
def test_sleep_wake_cycle():
    system = NSCKFullSystem()
    
    # Awake: accumulate experiences
    for i in range(100):
        system.interact(f"Test interaction {i}")
    
    episodic_count_before = len(system.episodic_memory.episodes)
    semantic_count_before = len(system.semantic_memory.concept_graph)
    
    # Sleep: consolidate
    system.sleep_cycle()
    
    # Verify consolidation
    semantic_count_after = len(system.semantic_memory.concept_graph)
    assert semantic_count_after > semantic_count_before  # New concepts extracted
```

---

## Milestones & Deliverables

### Month 12 (End of Phase 1)
- ✅ Natural language chatbot functional
- ✅ Multi-turn dialogue with context
- ✅ Voice input/output working
- **Deliverable:** Interactive demo with speech I/O

### Month 24 (End of Phase 2)
- ✅ Emotion recognition and expression
- ✅ Theory of Mind (passes Sally-Anne test)
- ✅ Empathetic responses
- **Deliverable:** Social interaction demo

### Month 36 (End of Phase 3)
- ✅ Semantic knowledge extraction
- ✅ Generative dreaming functional
- ✅ Autobiographical memory
- **Deliverable:** "Tell me about yourself" capability

### Month 48 (End of Phase 4)
- ✅ Continual learning (10+ tasks)
- ✅ Meta-learning (few-shot adaptation)
- ✅ Self-guided curriculum
- **Deliverable:** Lifelong learning benchmark

### Month 60 (End of Phase 5)
- ✅ Consciousness metrics (Φ computation)
- ✅ Self-modification capability
- ✅ Value-aligned behavior
- **Deliverable:** Full sentient AGI system

---

## Risk Mitigation

### Technical Risks
1. **Catastrophic Forgetting:** Mitigated by EWC + PackNet
2. **Computational Cost:** Use quantization + pruning + efficient VSA
3. **Data Scarcity:** Meta-learning enables few-shot adaptation

### Safety Risks
1. **Value Misalignment:** Terminal values + human feedback loop
2. **Uncontrolled Self-Modification:** Sandboxing + safety verification
3. **Harmful Outputs:** Content filtering + value alignment checks

### Timeline Risks
1. **Underestimated Complexity:** Modular design allows incremental progress
2. **Resource Constraints:** Prioritize core features, defer optimizations
3. **Technical Blockers:** Have fallback approaches for each component

---

## Conclusion

This roadmap provides a concrete path from the current NSCK system (game-playing agent) to a fully sentient AGI with:
- Natural language understanding and generation
- Emotional intelligence and empathy
- Advanced memory systems with dreaming
- Continual lifelong learning
- Computational correlates of consciousness
- Self-modification and value alignment

**Timeline:** 60 months (5 years) with dedicated development team

**Next Steps:**
1. Begin Phase 1: Train semantic folding model on Wikipedia
2. Integrate small LLM (Phi-3 or similar)
3. Build dialogue management system
4. Run first interactive language tests

**Success Criteria:** Agent can engage in natural conversation, understand emotions, learn continuously, and demonstrate self-awareness while remaining aligned with human values.

---

*End of Implementation Roadmap*
