# NSCK Workflows

Step-by-step explanations of how NSCK processes information, makes decisions, and learns. All flows are traced from the actual code in `cognitive_engine.py` and supporting modules.

---

## Table of Contents

- [1. The Decision Loop (decide)](#1-the-decision-loop)
- [2. The Learning Cycle (learn + record_outcome)](#2-the-learning-cycle)
- [3. Text Knowledge Ingestion](#3-text-knowledge-ingestion)
- [4. Memory Recall Pipeline](#4-memory-recall-pipeline)
- [5. SNN Perception Pipeline](#5-snn-perception-pipeline)
- [6. Sleep Consolidation](#6-sleep-consolidation)
- [7. Task Registration](#7-task-registration)
- [8. Dialogue Processing](#8-dialogue-processing)
- [9. Transparency / Explanation Flow](#9-transparency--explanation-flow)
- [10. Full Agent Lifecycle](#10-full-agent-lifecycle)

---

## 1. The Decision Loop

**Entry point:** `CognitiveEngine.decide(state: dict, available_actions: list, task_tag: str)`

This is the core cognitive cycle — called once per timestep.

```mermaid
sequenceDiagram
    participant Env as Environment
    participant CE as CognitiveEngine
    participant GV as GroundingVerifier
    participant EM as EpisodicMemory
    participant CU as CuriosityModule
    participant GWT as GlobalWorkspace
    participant SG as SafetyGate
    participant SM as SelfModel
    participant EX as ExplanationGenerator

    Env->>CE: decide(state, actions, task_tag)
    
    Note over CE: Step 1: Ground state
    CE->>GV: get_active_predicates(state, task_tag)
    GV-->>CE: active_preds: List[str]
    
    Note over CE: Step 2: Create situation vector
    CE->>EM: create_situation_hv(active_preds)
    EM-->>CE: situation_hv: HyperVector
    
    Note over CE: Step 3: Check curiosity
    CE->>CU: should_explore(situation_hv, task_tag, confidence)
    CU-->>CE: ExplorationDecision
    
    Note over CE: Step 4: Build coalitions
    CE->>CE: _build_coalitions()
    Note over CE: Sources: EXTERNAL, RULES,<br/>EXPLORATION, Q_LEARNING,<br/>MEMORY, PLANNER
    
    Note over CE: Step 5: GWT competition
    CE->>GWT: compete(coalitions)
    GWT->>GWT: score each coalition<br/>A = salience + relevance +<br/>affect + 0.5×confidence + bias
    GWT->>GWT: threshold filter (≥0.5)
    GWT-->>CE: winner: Coalition
    
    Note over CE: Step 6: Safety check
    CE->>SG: is_safe(action, state, task_tag)
    SG-->>CE: safe: bool
    
    Note over CE: Step 7: Generate explanation
    CE->>EX: explain_action(...)
    EX-->>CE: explanation: Explanation
    
    Note over CE: Step 8: Update self-model
    CE->>SM: get_confidence(task_tag)
    SM-->>CE: confidence: float
    
    CE-->>Env: CognitiveState(action, explanation, confidence, ...)
```

### Step-by-Step Detail

1. **Grounding** — `GroundingVerifier` evaluates all registered predicates against the current state. Each predicate is a lambda function (e.g., `"wall_north": lambda s: s.get("wall_north", False)`). Returns a list of string predicate names that are currently true.

2. **Situation Encoding** — Active predicates are bundled into a 10,240-bit HyperVector: `situation_hv = bundle([HV(pred) for pred in active_preds])`. This provides a compact, similarity-preserving representation of "what's happening now."

3. **Curiosity Check** — The situation HV is compared against all previously visited states. If novelty > 0.5, the curiosity module recommends exploration. A random action is injected as the EXPLORATION coalition.

4. **Coalition Building** — Up to 6 coalition sources contribute proposals:
   - **EXTERNAL**: Domain-registered action selector (e.g., task-specific heuristic)
   - **RULES**: Best matching rule from RuleLearner
   - **EXPLORATION**: Random action if curiosity says explore
   - **Q_LEARNING**: argmax Q(s,a) from tabular Q-values
   - **MEMORY**: Best action from similar past episodes
   - **PLANNER**: A* plan's first step (if planning is active)

5. **GWT Competition** — All coalitions are scored and the highest-activation proposal wins, subject to threshold filtering.

6. **Safety Gate** — A final check that the winning action isn't dangerous. If vetoed, the next-best coalition is tried.

7. **Explanation** — A natural language explanation is generated from templates, describing why this action was chosen.

8. **Return** — A `CognitiveState` is returned containing the action, explanation, confidence, and full reasoning trace.

---

## 2. The Learning Cycle

After each action, two methods update the system:

### record_outcome(state, action, reward, next_state, task_tag)

```mermaid
graph TD
    RO["record_outcome()"]
    
    RO --> Q["Q-Learning TD(0) Update<br/>Q(s,a) += α[r + γ·max Q(s',·) - Q(s,a)]"]
    RO --> SM["SelfModel.update()<br/>Track prediction accuracy"]
    RO --> CU["Curiosity.record_outcome()<br/>Update learning progress"]
    RO --> EP["Epsilon decay<br/>ε = max(0.1, ε × 0.995)"]
```

### learn(state, action, reward, next_state, task_tag)

```mermaid
graph TD
    L["learn()"]
    
    L --> EM["EpisodicMemory.record()<br/>Store full episode"]
    L --> RL["RuleLearner.observe()<br/>Count predicate-action pairs"]
    L --> RI["RuleLearner.induce_rules()<br/>Promote frequent patterns"]
    L --> CD["CausalDiscovery.observe()<br/>Update contingency tables"]
    L --> CI["CausalDiscovery.induce_graph()<br/>Build causal model (every 10 steps)"]
    L --> PL["Planner.learn_operators_from_graph()<br/>Extract STRIPS operators"]
```

### Learning Timeline

```mermaid
sequenceDiagram
    participant Env as Environment
    participant CE as CognitiveEngine
    
    loop Every timestep
        Env->>CE: decide(state, actions, tag) → action
        Env->>Env: execute action → next_state, reward
        Env->>CE: record_outcome(state, action, reward, next_state, tag)
        Env->>CE: learn(state, action, reward, next_state, tag)
    end
    
    Note over CE: After N episodes:
    CE->>CE: sleep() → offline consolidation
```

---

## 3. Text Knowledge Ingestion

**Entry point:** `TextKnowledgeLearner.learn_from_text_file(path)`

```mermaid
graph TD
    File["Text File"]
    
    File -->|"read + split sentences"| Sent["Sentence List"]
    
    Sent -->|"for each sentence"| Step1["Step 1: Parse Intent<br/>LanguageModule.understand(text)<br/>→ structured frames"]
    
    Step1 --> Step2["Step 2: Extract Relations<br/>Pattern matching (regex)<br/>+ S-V-O scanning<br/>→ (subject, relation, object)"]
    
    Step2 --> Step3a["Step 3a: Semantic Memory<br/>add_concept(subject, props)<br/>add_relation(subject, rel, object)"]
    
    Step2 --> Step3b["Step 3b: Causal Graph<br/>add_causes(cause, effect)<br/>add_prevents(cause, effect)"]
    
    Step2 --> Step3c["Step 3c: Episodic Memory<br/>record(LiveEpisode with<br/>sentence context)"]
    
    Sent -->|"after all sentences"| Step4["Step 4: Semantic Folding<br/>Co-occurrence counting<br/>Context HV construction<br/>Implicit relation discovery"]
    
    Step4 --> Step5["Step 5: Implicit Relations<br/>For concepts with<br/>sim(ctx_a, ctx_b) > 0.55<br/>→ add_relation(a, 'related_to', b)"]
```

### Detailed Pipeline

1. **File Reading**: Split into sentences on `.`/`!`/`?`
2. **Per-Sentence Processing**:
   - `LanguageModule.understand(sentence)` → structured intent (or heuristic NLU)
   - Regex-based relation extraction: `"X causes Y"`, `"X is a Y"`, `"X has Y"`
   - S-V-O scanning: identify subject/verb/object triples
3. **Knowledge Storage**:
   - Concepts → SemanticMemory graph nodes with VSA HyperVectors
   - Relations → SemanticMemory graph edges
   - Causal relations → CausalGraph
   - Full episodes → EpisodicMemory
4. **Semantic Folding** (post-processing):
   - Build co-occurrence matrix within 7-word windows
   - Construct context HVs using permuted bundling
   - Discover implicit relations via context similarity

---

## 4. Memory Recall Pipeline

### Episodic Memory Recall

```mermaid
graph TD
    Query["Query: situation_hv"]
    
    Query --> LSH["LSH Bucket Lookup<br/>4 tables × 10-bit hashes<br/>Check exact matches"]
    
    LSH --> Probe["1-bit Neighbor Probing<br/>For bits 0..7: check h ⊕ 2ᵇ<br/>Finds near-misses"]
    
    Probe --> Candidates["Candidate Set<br/>(reduces search space)"]
    
    Candidates --> Exact["Exact Similarity Ranking<br/>sim(query, episode) for each<br/>Sort descending"]
    
    Exact --> TopK["Return top-k episodes<br/>with timestamps + rewards"]
```

### Semantic Memory Recall

```mermaid
graph TD
    Query2["Query: concept name or HV"]
    
    Query2 --> Direct["Direct Lookup<br/>graph.nodes[concept]"]
    
    Query2 --> VSA["VSA Similarity Search<br/>cosine_sim(query, all_concepts)<br/>Return top-k matches"]
    
    Query2 --> Spread["Spreading Activation<br/>Start from seed concepts<br/>Propagate through edges<br/>Decay per hop"]
    
    Direct --> Result["Properties + Relations"]
    VSA --> Result
    Spread --> Result
```

---

## 5. SNN Perception Pipeline

**Entry point:** `SNNPerceptionModule.perceive(sensory_input)`

```mermaid
graph TD
    Input["Sensory Input<br/>(numpy array, shape N)"]
    
    Input --> Weights["Input Weight Matrix<br/>W: (snn_size × input_size)"]
    
    Weights --> Current["Input Current<br/>I = W · input"]
    
    Current --> LIF["LIF Neuron Layer<br/>For T timesteps:<br/>  V += dt/τ × (-V + I)<br/>  if V ≥ V_th: spike, V = 0"]
    
    LIF --> STDP["STDP Weight Update<br/>For each pre-post pair:<br/>  if Δt > 0: W += A+ exp(-Δt/τ+)<br/>  if Δt < 0: W -= A- exp(Δt/τ-)"]
    
    LIF --> Spikes["Spike Train<br/>(neurons × timesteps)"]
    
    Spikes --> Rate["RateCoder.encode()<br/>Bundle HVs of active neurons<br/>(firing rate > threshold)"]
    
    Rate --> ConceptHV["Concept HyperVector<br/>10,240-bit"]
    
    ConceptHV --> Cleanup["ConceptMapper.recognize_pattern()<br/>Match against known concepts<br/>Learn new if novel"]
    
    ConceptHV --> Hebb["VSAHebbianLearner<br/>Update cross-concept associations"]
    
    Cleanup --> Output["Recognized Concept<br/>+ confidence score"]
```

---

## 6. Sleep Consolidation

**Entry point:** `CognitiveEngine.sleep()`

Called periodically (e.g., after N episodes) for offline learning:

```mermaid
graph TD
    Sleep["sleep()"]
    
    Sleep --> EM["EpisodicMemory Consolidation<br/>Hot deque → SQLite warm tier<br/>Rebuild LSH indices"]
    
    Sleep --> RI["Rule Induction<br/>RuleLearner.induce_rules()<br/>Promote frequent patterns"]
    
    Sleep --> RP["Rule Pruning<br/>Remove low-performing rules<br/>(tenure > 100, success < 0.1)"]
    
    Sleep --> CI["Causal Graph Induction<br/>CausalDiscovery.induce_graph()<br/>Update from accumulated stats"]
    
    Sleep --> PL["Planner Operator Learning<br/>learn_operators_from_graph()<br/>Extract STRIPS operators"]
    
    Sleep --> Save["BrainStore.checkpoint()<br/>Persist rules + episodes"]
```

---

## 7. Task Registration

**Entry point:** `CognitiveEngine.register_task(task_tag, predicates, actions, ...)`

```mermaid
graph TD
    Reg["register_task(tag, predicates, actions, ...)"]
    
    Reg --> GV["GroundingVerifier<br/>register_predicate(name, lambda)<br/>register_action(name, lambda)"]
    
    Reg --> CG["CausalGraph<br/>Initialize for task (optional)<br/>add_causes(), add_prevents()"]
    
    Reg --> BF["BrainFusion<br/>get_or_create_brain(tag)<br/>→ TaskBrain for this domain"]
    
    Reg --> RL["RuleLearner<br/>Initialize task-specific counters"]
```

### Example: Registering a Maze Task

```python
engine.register_task(
    task_tag="maze",
    predicates={
        "wall_north": lambda s: s.get("wall_north", False),
        "wall_south": lambda s: s.get("wall_south", False),
        "at_goal":    lambda s: s.get("player") == s.get("goal"),
    },
    actions=["move_north", "move_south", "move_east", "move_west"],
    causal_links=[("move_north", "player_y_decreased")],
)
```

---

## 8. Dialogue Processing

**Entry point:** `DialogueManager.respond(user_text)` or `process_turn(text)`

```mermaid
sequenceDiagram
    participant User
    participant DM as DialogueManager
    participant LM as LanguageModule
    participant CE as CognitiveEngine

    User->>DM: respond("set goal to find food")
    DM->>LM: understand(text)
    LM-->>DM: {intent: "set_goal", entities: ["food"]}
    
    alt intent == "set_goal"
        DM->>CE: set_mission_goal("food")
        DM-->>User: "Goal set to: food"
    else intent == "query_status"
        DM->>CE: get_stats()
        DM-->>User: "Stats: {episodes: 42, rules: 5, ...}"
    else intent == "explain"
        DM->>CE: explain()
        DM-->>User: "Last action: move_north because Q-value was highest"
    else unknown
        DM->>LM: generate(fallback_response)
        DM-->>User: "I'm not sure how to help with that"
    end
```

---

## 9. Transparency / Explanation Flow

Every decision produces a traceable explanation:

```mermaid
graph TD
    Decision["decide() returns CognitiveState"]
    
    Decision --> Trace["Full Reasoning Trace<br/>state → predicates → coalitions →<br/>scores → winner → safety check"]
    
    Decision --> NL["Natural Language Explanation<br/>ExplanationGenerator templates:<br/>'Chose {action} because {reason}'"]
    
    Decision --> QV["Q-Value Table<br/>engine.q_values[(state, action)] → float<br/>Fully inspectable"]
    
    Decision --> Rules["Active Rules<br/>engine.rules → List[Rule]<br/>Each with condition, action, weight"]
    
    Decision --> Coalition["Coalition Scores<br/>engine.last_coalitions →<br/>List[(source, action, score)]"]
    
    Decision --> Conf["Confidence<br/>engine.self_model.get_confidence(tag)<br/>Calibrated per-task estimate"]
```

### How to Inspect

```python
engine = CognitiveEngine()
# ... register task and run episodes ...

# Decision trace
state = engine.decide(state, actions, "maze")
print(state.action)           # What was chosen
print(state.explanation)      # Why (natural language)
print(state.confidence)       # How confident

# Inspect internals
print(engine.q_values)        # All Q-values
print(engine.rules)           # All learned rules
print(engine.explain())       # Detailed explanation
```

---

## 10. Full Agent Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Init: CognitiveEngine()
    
    Init --> Register: register_task()
    
    Register --> Perceive: New observation
    
    state "Active Loop" as Active {
        Perceive --> Decide: decide()
        Decide --> Act: return action
        Act --> Feedback: reward from environment
        Feedback --> Learn: learn() + record_outcome()
        Learn --> Perceive: Next observation
    }
    
    Active --> Sleep: Periodic consolidation
    Sleep --> Active: Resume
    
    Active --> Save: BrainStore.checkpoint()
    Save --> Active: Continue
    
    Active --> [*]: Session ends
```

### Lifecycle Phases

| Phase | What Happens | Key Methods |
|-------|-------------|-------------|
| **Init** | All modules instantiated, Rust backend loaded | `CognitiveEngine()` |
| **Register** | Domain-specific predicates, actions, causal links added | `register_task()` |
| **Perceive** | State grounded into predicates + situation HV | `get_active_predicates()`, `create_situation_hv()` |
| **Decide** | Coalitions compete in GWT, safety-checked | `decide()` |
| **Act** | Winning action returned to environment | Return `CognitiveState` |
| **Feedback** | Q-values updated, confidence adjusted | `record_outcome()` |
| **Learn** | Episodes stored, rules induced, causal graph updated | `learn()` |
| **Sleep** | Offline consolidation, pruning, operator learning | `sleep()` |
| **Save** | Brain state persisted to SQLite | `BrainStore.checkpoint()` |
