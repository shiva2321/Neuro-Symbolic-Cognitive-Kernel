# NCGN Workflow Documentation

Detailed workflow diagrams and process descriptions for the Neuromorphic Cognitive Graph Network.

## Table of Contents
- [System Workflows](#system-workflows)
- [Data Flow](#data-flow)
- [State Machines](#state-machines)
- [Process Sequences](#process-sequences)
- [Integration Patterns](#integration-patterns)

## System Workflows

### 1. Complete System Tick Workflow

This is the main execution loop that runs continuously during operation.

```
╔═══════════════════════════════════════════════════════════════════╗
║                   COMPLETE SYSTEM TICK WORKFLOW                   ║
╚═══════════════════════════════════════════════════════════════════╝

START TICK (t)
│
├─────────── CAPTURE PREDICTION ──────────────
│            (For surprise calculation)
│            predicted_state = current_state
│
├─────────── PHASE 0: TRANSDUCTION ───────────
│            External Energy Input
│            ┌─────────────────────────┐
│            │ sensory_buffer          │
│            │   "dog" → 1.0          │
│            │   "metal" → 0.5        │
│            └───────────┬─────────────┘
│                        │
│                        ▼
│            ┌─────────────────────────┐
│            │ Apply to nodes:         │
│            │ E = min(1.0, E + Input) │
│            └─────────────────────────┘
│
├─────────── PHASE 1: PASSIVE DECAY ──────────
│            Energy Leak
│            ┌─────────────────────────┐
│            │ For each active node:   │
│            │   E = E × 0.9          │
│            │   if E < 0.001:        │
│            │     deactivate          │
│            └─────────────────────────┘
│
├─────────── PHASE 2: FIRING DETERMINATION ───
│            Threshold Check
│            ┌─────────────────────────┐
│            │ For each active node:   │
│            │   if E > threshold AND  │
│            │      refractory == 0:   │
│            │     mark_firing()       │
│            └───────────┬─────────────┘
│                        │
│                        ▼
│            ┌─────────────────────────┐
│            │ firing_set = {nodes}    │
│            └─────────────────────────┘
│
├─────────── PHASE 3: REFRACTORY & RESET ─────
│            Post-Firing Reset
│            ┌─────────────────────────┐
│            │ For node in firing_set: │
│            │   E = 0.0               │
│            │   refractory_timer = 3  │
│            │   last_fired = t        │
│            └─────────────────────────┘
│
├─────────── PHASE 4: PROPAGATION ────────────
│            Spike Transmission
│            ┌─────────────────────────┐
│            │ For (src, tgt) in edges:│
│            │   if src in firing_set: │
│            │     signal = weight × 1 │
│            │     buffer[tgt] += sig  │
│            └─────────────────────────┘
│            Note: Buffered, not applied
│
├─────────── PHASE 5: INTEGRATION ────────────
│            Apply Buffered Signals
│            ┌─────────────────────────┐
│            │ For tgt, sig in buffer: │
│            │   E_tgt = min(1, E+sig) │
│            │   activate(tgt)         │
│            └─────────────────────────┘
│
├─────────── PHASE 6: K-WTA INHIBITION ───────
│            Attention Mechanism
│            ┌─────────────────────────┐
│            │ Select top K nodes      │
│            │   by energy             │
│            │ Suppress all others:    │
│            │   E = 0, deactivate     │
│            └─────────────────────────┘
│            Implementation: Min-Heap
│
├─────────── PHASE 7: SURPRISE MONITOR ───────
│            Compare Prediction vs Reality
│            ┌─────────────────────────┐
│            │ observed = current_state│
│            │ S = RMS(predicted,      │
│            │         observed,       │
│            │         confidence)     │
│            └───────────┬─────────────┘
│                        │
│               ┌────────▼────────┐
│               │ S > threshold?  │
│               └────┬──────┬─────┘
│                    │ YES  │ NO
│          ┌─────────▼      │
│          │ PAUSE           │
│          │ Trigger S2      │
│          │ ┌─────────────┐ │
│          │ │ System 2    │ │
│          │ │ Diagnosis   │ │
│          │ │ Intervention│ │
│          │ └──────┬──────┘ │
│          │        │        │
│          │        ▼        │
│          │  Modify Graph   │
│          │  Resume System 1│
│          └────────┬────────┘
│                   │
├─────────── PHASE 8: MAINTENANCE ────────────┘
│            Housekeeping
│            ┌─────────────────────────┐
│            │ For each node:          │
│            │   refractory_timer--    │
│            │   novelty_score *= 0.999│
│            │ Prune inactive nodes    │
│            └─────────────────────────┘
│
END TICK (t)
│
└─ Increment t, goto START TICK
```

### 2. Surprise Detection and System 2 Intervention

```
╔═══════════════════════════════════════════════════════════════════╗
║             SURPRISE DETECTION & INTERVENTION WORKFLOW            ║
╚═══════════════════════════════════════════════════════════════════╝

During Phase 7 (Surprise Monitor)
│
├─ Capture States
│  ├─ Predicted: {meat: 0.9, eat: 0.7}
│  └─ Observed:  {metal: 1.0, eat: 0.8, meat: 0.0}
│
├─ Calculate Surprise
│  │
│  │  For each predicted node n:
│  │    diff = E_pred(n) - E_obs(n)
│  │    weighted = diff × confidence(n)
│  │    squared = weighted²
│  │
│  └─ S = sqrt(sum(squared))
│     = sqrt(((0.9 - 0.0) × 0.9)² + ...)
│     = 0.81
│
├─ Check Threshold
│  │
│  └─ if S (0.81) > threshold (0.45):
│     └─ TRIGGER SYSTEM 2
│
└─ System 2 Intervention
   │
   ├─ [1] PAUSE System 1
   │      ├─ Save state
   │      ├─ Stop tick loop
   │      └─ Wait for S2
   │
   ├─ [2] CONTEXT EXTRACTION
   │      ├─ What was expected? (meat)
   │      ├─ What was observed? (metal)
   │      ├─ What action? (eat)
   │      └─ Who is agent? (dog)
   │      
   │      Triple: (dog, eat, metal)
   │
   ├─ [3] SCHEMA RETRIEVAL
   │      ├─ Load: schema_eat.json
   │      └─ Extract constraints:
   │         ├─ agent: [is_animate]
   │         └─ target: [is_edible]
   │
   ├─ [4] CONSTRAINT CHECKING
   │      │
   │      ├─ Check dog:
   │      │  ├─ has_property("dog", "is_animate")?
   │      │  └─ ✓ TRUE
   │      │
   │      └─ Check metal:
   │         ├─ has_property("metal", "is_edible")?
   │         └─ ✗ FALSE → VIOLATION!
   │
   ├─ [5] DIAGNOSIS
   │      │
   │      └─ DiagnosisResult:
   │         ├─ type: CONSTRAINT_VIOLATION
   │         ├─ violated: ["is_edible"]
   │         ├─ explanation: "metal is not edible"
   │         └─ confidence: 0.95
   │
   ├─ [6] INTERVENTION PLANNING
   │      │
   │      ├─ Strategy 1: LTD (Long-Term Depression)
   │      │  └─ Weaken synapse (dog → metal)
   │      │     weight: 0.9 → 0.72
   │      │
   │      ├─ Strategy 2: Goal Injection
   │      │  └─ Inject energy into "Query_User"
   │      │     E(Query_User) = 1.0
   │      │
   │      └─ Strategy 3: Generate Query
   │         └─ Template: "My physics say {object} 
   │            isn't {constraint}. Why?"
   │
   ├─ [7] EXECUTE INTERVENTIONS
   │      │
   │      ├─ Apply LTD
   │      ├─ Inject goal energy
   │      └─ Output query to user
   │
   ├─ [8] WAIT FOR RESOLUTION
   │      │
   │      ├─ State: CLARIFICATION_PENDING
   │      │
   │      └─ User responds:
   │         "It's a robot dog"
   │
   ├─ [9] PARSE EXPLANATION
   │      │
   │      ├─ Extract: "robot dog"
   │      ├─ Type: NEW_SUBCLASS
   │      │
   │      └─ Proposal:
   │         ├─ robot_dog --[is_a]--> dog
   │         └─ robot_dog can eat metal
   │
   ├─ [10] UPDATE ONTOLOGY
   │       │
   │       ├─ Add node: "robot_dog"
   │       ├─ Add edge: (robot_dog, is_a, dog)
   │       ├─ Add exception: 
   │       │  set_property(robot_dog, "can_eat_metal", True)
   │       │
   │       └─ Commit to memory
   │
   └─ [11] RESUME System 1
          ├─ State: IDLE
          └─ Continue normal execution
```

## Data Flow

### 3. Knowledge Ingestion Pipeline

```
╔═══════════════════════════════════════════════════════════════════╗
║                  KNOWLEDGE INGESTION WORKFLOW                     ║
╚═══════════════════════════════════════════════════════════════════╝

[INPUT] Text File or User Statement
│
│   Example: "Dogs eat meat. Cats eat fish. Dogs are pets."
│
└─► [STAGE 1] Document Reader
    │
    ├─ Split into sentences
    │  ├─ "Dogs eat meat."
    │  ├─ "Cats eat fish."
    │  └─ "Dogs are pets."
    │
    └─► [STAGE 2] Text-to-Triple Parser
        │
        ├─ For each sentence:
        │  │
        │  ├─ NLP Analysis (spaCy):
        │  │  ├─ POS tagging
        │  │  ├─ Dependency parsing
        │  │  └─ Entity recognition
        │  │
        │  └─ Extract Triple:
        │     ├─ Subject (NOUN)
        │     ├─ Predicate (VERB)
        │     └─ Object (NOUN/ADJ)
        │
        ├─ Output:
        │  ├─ Triple("dogs", "eat", "meat", rel=ASSOCIATES)
        │  ├─ Triple("cats", "eat", "fish", rel=ASSOCIATES)
        │  └─ Triple("dogs", "are", "pets", rel=IS_A)
        │
        └─► [STAGE 3] Staging Buffer
            │
            ├─ Create temporary copy
            │  │
            │  └─ staging_memory = GraphMemory()
            │
            ├─ Add triples to staging:
            │  │
            │  ├─ For each triple:
            │  │  ├─ Add nodes (if new)
            │  │  └─ Add edge with confidence=0.3
            │  │
            │  └─ staging_memory:
            │     ├─ nodes: {dogs, cats, meat, fish, pets}
            │     └─ edges: {(dogs,eat,meat), (cats,eat,fish), ...}
            │
            └─► [STAGE 4] Validation
                │
                ├─ For each triple:
                │  │
                │  ├─ Check against schemas:
                │  │  │
                │  │  ├─ Load schema for predicate
                │  │  │  (e.g., schema_eat.json)
                │  │  │
                │  │  └─ Validate constraints:
                │  │     │
                │  │     ├─ agent: is_animate?
                │  │     └─ target: is_edible?
                │  │
                │  ├─ Check for conflicts:
                │  │  │
                │  │  ├─ Existing edge says opposite?
                │  │  ├─ Violates known property?
                │  │  └─ Contradicts schema?
                │  │
                │  └─ Classification:
                │     ├─ VALID → Add to commit queue
                │     ├─ CONFLICT → Flag for review
                │     └─ VIOLATION → Flag for review
                │
                └─► [STAGE 5] Conflict Resolution
                    │
                    ├─ For each conflict:
                    │  │
                    │  ├─ Present to user:
                    │  │  "New: 'dogs eat metal'"
                    │  │  "Existing: metal is not edible"
                    │  │  "Action: Accept / Reject / Modify?"
                    │  │
                    │  └─ User decision:
                    │     ├─ ACCEPT → Add to commit queue
                    │     ├─ REJECT → Discard
                    │     └─ MODIFY → Edit and re-validate
                    │
                    └─► [STAGE 6] Merge to Main Memory
                        │
                        ├─ For each node in staging:
                        │  │
                        │  ├─ If exists in main:
                        │  │  └─ Decrease novelty_score
                        │  │     (reinforcement)
                        │  │
                        │  └─ If new:
                        │     └─ Create with novelty=0.5
                        │
                        ├─ For each edge in staging:
                        │  │
                        │  ├─ If exists in main:
                        │  │  └─ Average confidences:
                        │  │     new_conf = (old + new) / 2
                        │  │
                        │  └─ If new:
                        │     └─ Add with confidence=0.3
                        │
                        └─► [OUTPUT] Updated Graph Memory
                            │
                            └─ Commit confirmed
                               Log: "Learned N nodes, M edges"
```

### 4. Dialogue State Machine

```
╔═══════════════════════════════════════════════════════════════════╗
║                      DIALOGUE STATE MACHINE                       ║
╚═══════════════════════════════════════════════════════════════════╝

                    ┌─────────────┐
                    │   STARTUP   │
                    └──────┬──────┘
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │              IDLE                    │
        │  • Waiting for user input            │
        │  • No pending tasks                  │
        │  • Context cleared                   │
        └──────────────┬───────────────────────┘
                       │
                       │ User enters text
                       ▼
        ┌──────────────────────────────────────┐
        │          PROCESSING                  │
        │  • Parse input                       │
        │  • Extract triples                   │
        │  • Run System 1 ticks                │
        └──┬──────────────────┬────────────────┘
           │                  │
           │ Success          │ Surprise detected
           │                  │
           ▼                  ▼
┌──────────────────┐   ┌──────────────────────┐
│  Output Result   │   │  CLARIFICATION_      │
│  • Success msg   │   │  PENDING             │
│  • Learned facts │   │  • Wait for explain  │
└────────┬─────────┘   │  • Show query        │
         │             │  • Save context      │
         │             └──────────┬───────────┘
         │                        │
         │                        │ User explains
         │                        ▼
         │             ┌────────────────────────┐
         │             │  Parse Explanation     │
         │             │  • Identify type:      │
         │             │    - NEW_SUBCLASS      │
         │             │    - EXCEPTION         │
         │             │    - NEW_PROPERTY      │
         │             │    - REJECTION         │
         │             └──────────┬─────────────┘
         │                        │
         │                        ▼
         │             ┌────────────────────────┐
         │             │  System 2 Decision     │
         │             │  • Validate proposal   │
         │             │  • Update ontology     │
         │             │  • Modify synapses     │
         │             └──────────┬─────────────┘
         │                        │
         │                        ▼
         │             ┌────────────────────────┐
         │             │  Confirmation Message  │
         │             │  • Explain changes     │
         │             │  • List new facts      │
         │             └──────────┬─────────────┘
         │                        │
         └────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────────┐
        │            IDLE                      │
        └──────────────────────────────────────┘


[FILE INGESTION BRANCH]

        ┌──────────────────────────────────────┐
        │              IDLE                    │
        └──────────────┬───────────────────────┘
                       │
                       │ User: "read file.txt"
                       ▼
        ┌──────────────────────────────────────┐
        │         FILE_REVIEW                  │
        │  • Parse file                        │
        │  • Extract all triples               │
        │  • Validate each                     │
        │  • Identify conflicts                │
        └──────────────┬───────────────────────┘
                       │
                       │ Conflicts found
                       ▼
        ┌──────────────────────────────────────┐
        │      CONFIRMATION_PENDING            │
        │  • Show conflicts                    │
        │  • List for review                   │
        │  • Wait for decision                 │
        └──────────────┬───────────────────────┘
                       │
                       │ User: "accept" / "reject" / "review"
                       │
           ┌───────────┼───────────┐
           │           │           │
           ▼           ▼           ▼
      [ACCEPT]    [REJECT]    [REVIEW]
           │           │           │
           │           │           └─► Show details
           │           │               User modifies
           │           │               Re-validate
           │           │                   │
           │           └───────────────────┘
           │                       │
           └───────────┬───────────┘
                       │
                       ▼
        ┌──────────────────────────────────────┐
        │         Merge to Memory              │
        │  • Commit approved changes           │
        │  • Update graph                      │
        │  • Log statistics                    │
        └──────────────┬───────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────────┐
        │              IDLE                    │
        └──────────────────────────────────────┘
```

## Process Sequences

### 5. Learning a New Fact

```
╔═══════════════════════════════════════════════════════════════════╗
║                    LEARNING PROCESS SEQUENCE                      ║
╚═══════════════════════════════════════════════════════════════════╝

User Input: "Dogs eat meat"
│
├─► [1] Input Reception
│       └─ dialogue_manager.process_input("Dogs eat meat")
│
├─► [2] Triple Extraction
│       ├─ Parse with NLP
│       └─ Triple("dogs", "eat", "meat", rel=ASSOCIATES, conf=0.3)
│
├─► [3] Node Creation
│       ├─ memory.add_node("dogs", threshold=0.5)
│       ├─ memory.add_node("eat", threshold=0.5)
│       └─ memory.add_node("meat", threshold=0.5, novelty=0.5)
│
├─► [4] Edge Creation
│       ├─ memory.add_synapse("dogs", "eat", weight=0.8, conf=0.9)
│       └─ memory.add_synapse("eat", "meat", weight=0.8, conf=0.9)
│
├─► [5] Schema Check (Optional)
│       ├─ Load schema_eat.json
│       ├─ Verify: dogs.is_animate = True
│       ├─ Verify: meat.is_edible = True
│       └─ Result: ✓ VALID
│
└─► [6] Confirmation
        └─ Output: "✓ Learned: dogs eats meat"


User Input: "What do dogs eat?"
│
├─► [1] Query Parsing
│       ├─ Identify: QUERY
│       ├─ Subject: "dogs"
│       └─ Relation: "eat"
│
├─► [2] Graph Traversal
│       ├─ Find node: "dogs"
│       ├─ Get outgoing edges with relation "eat"
│       └─ Collect targets: ["meat"]
│
├─► [3] Response Generation
│       └─ Template: "{Subject} {relation} {objects}"
│
└─► [4] Output
        └─ "Dogs eat: meat"
```

### 6. Handling a Contradiction

```
╔═══════════════════════════════════════════════════════════════════╗
║                 CONTRADICTION HANDLING SEQUENCE                   ║
╚═══════════════════════════════════════════════════════════════════╝

Setup: System already knows "dogs eat meat" (high confidence)

User Input: "Dogs eat metal"
│
├─► [1] Triple Extraction
│       └─ Triple("dogs", "eat", "metal", rel=ASSOCIATES)
│
├─► [2] System 1 Simulation
│       ├─ Activate "dogs" (E=1.0)
│       ├─ Tick 1: Propagate to "eat" and "meat"
│       ├─ Prediction captured: {meat: 0.85, eat: 0.7}
│       │
│       └─ User injects "metal" (E=1.0)
│
├─► [3] Surprise Detection (Phase 7)
│       ├─ Predicted: {meat: 0.85}
│       ├─ Observed:  {metal: 1.0, meat: 0.0}
│       ├─ Calculate: S = 0.765
│       └─ S > 0.45 → TRIGGER!
│
├─► [4] System 2 Activation
│       ├─ PAUSE System 1
│       └─ Triple: (dogs, eat, metal)
│
├─► [5] Schema Validation
│       ├─ Load schema_eat.json
│       ├─ Check: metal.is_edible
│       ├─ Result: FALSE
│       └─ Diagnosis: CONSTRAINT_VIOLATION
│
├─► [6] Intervention
│       ├─ LTD: Weaken (dogs → metal) if exists
│       ├─ Goal: Inject "Query_User" energy
│       └─ Query: "My physics say metal isn't edible. 
│          Why do you say 'dogs eat metal'?"
│
├─► [7] State Change
│       └─ CLARIFICATION_PENDING
│
User Input: "It's a robot dog"
│
├─► [8] Explanation Parsing
│       ├─ Extract: "robot dog"
│       ├─ Relation to "dog": NEW_SUBCLASS
│       │
│       └─ Proposal:
│          ├─ Create: robot_dog
│          ├─ Link: robot_dog --[is_a]--> dog
│          └─ Exception: robot_dog can eat metal
│
├─► [9] System 2 Decision
│       ├─ Validate proposal
│       ├─ Check consistency
│       └─ Decision: ACCEPT
│
├─► [10] Ontology Update
│        ├─ memory.add_node("robot_dog")
│        ├─ memory.add_synapse("robot_dog", "dog", type="is_a")
│        ├─ memory.add_synapse("robot_dog", "metal", type="eats")
│        └─ controller.set_property("robot_dog", "can_eat_metal", True)
│
├─► [11] Confirmation
│        └─ Output:
│           "✓ Understood. I've learned:
│            • robot dog is a type of dog
│            • robot dog can eat metal"
│
└─► [12] State Reset
         └─ IDLE (ready for next input)
```

## Integration Patterns

### 7. Dashboard Integration

```
╔═══════════════════════════════════════════════════════════════════╗
║                    WEB DASHBOARD INTEGRATION                      ║
╚═══════════════════════════════════════════════════════════════════╝

[CLIENT SIDE - Browser]
        │
        │ HTTP GET /
        ▼
┌─────────────────────┐
│  Flask Web Server   │
│  (Port 5000)        │
└──────────┬──────────┘
           │
           │ Render template
           ▼
┌─────────────────────┐
│  index.html         │
│  • Graph canvas     │
│  • Control panel    │
│  • Surprise meter   │
│  • Log console      │
└──────────┬──────────┘
           │
           │ WebSocket connect
           ▼
┌─────────────────────┐
│  SocketIO Client    │
└──────────┬──────────┘
           │
           │ Events
           ▼

[EVENT LOOP]

Client Event: "inject_energy"
    Data: {node_id: "dog", energy: 1.0}
        │
        │ emit via WebSocket
        ▼
    ┌─────────────────────┐
    │  Flask-SocketIO     │
    │  Event Handler      │
    └──────────┬──────────┘
               │
               │ @socketio.on('inject_energy')
               ▼
    ┌─────────────────────┐
    │  System 1 Engine    │
    │  .inject_energy()   │
    └──────────┬──────────┘
               │
               │ Execute tick
               ▼
    ┌─────────────────────┐
    │  Graph Memory       │
    │  State Changed      │
    └──────────┬──────────┘
               │
               │ Serialize state
               ▼
    ┌─────────────────────┐
    │  Create JSON        │
    │  {                  │
    │    nodes: [...],    │
    │    edges: [...],    │
    │    surprise: 0.2    │
    │  }                  │
    └──────────┬──────────┘
               │
               │ emit('state_update', json)
               ▼
    ┌─────────────────────┐
    │  SocketIO Transport │
    └──────────┬──────────┘
               │
               │ WebSocket send
               ▼
    ┌─────────────────────┐
    │  Client Receives    │
    │  state_update       │
    └──────────┬──────────┘
               │
               │ Update DOM
               ▼
    ┌─────────────────────┐
    │  Visualizer         │
    │  • Redraw nodes     │
    │  • Update energies  │
    │  • Animate changes  │
    └─────────────────────┘

[AUTO-RUN MODE]

User clicks "Run"
    │
    ├─► Client: emit('start_auto_run')
    │
    ├─► Server: Start background thread
    │       │
    │       └─► Loop:
    │           ├─ engine.tick()
    │           ├─ Sleep(0.1)
    │           ├─ Serialize state
    │           ├─ emit('state_update')
    │           └─ if paused: break
    │
    └─► Client: Receive updates
            └─► Animate in real-time
```

### 8. Training Integration

```
╔═══════════════════════════════════════════════════════════════════╗
║                    TRAINING WORKFLOW INTEGRATION                  ║
╚═══════════════════════════════════════════════════════════════════╝

User: python run_training.py --curriculum animals
│
├─► [1] Load Curriculum
│       ├─ Path: training/curricula/animals.json
│       │
│       └─ Structure:
│          {
│            "episodes": [
│              {
│                "input_pattern": ["dog"],
│                "target_pattern": ["meat"],
│                "label": "Dog eats meat"
│              },
│              ...
│            ]
│          }
│
├─► [2] Initialize System
│       ├─ memory = GraphMemory()
│       ├─ engine = System1Engine(memory)
│       └─ controller = System2Controller(memory)
│
├─► [3] Training Loop
│       │
│       ├─ For epoch in range(epochs):
│       │   │
│       │   ├─ For episode in curriculum:
│       │   │   │
│       │   │   ├─► [A] Input Phase
│       │   │   │     ├─ Inject input_pattern
│       │   │   │     └─ engine.inject_energy("dog", 1.0)
│       │   │   │
│       │   │   ├─► [B] Simulation Phase
│       │   │   │     ├─ Run N ticks
│       │   │   │     └─ For _ in range(5):
│       │   │   │          engine.tick()
│       │   │   │
│       │   │   ├─► [C] Evaluation Phase
│       │   │   │     ├─ Check active nodes
│       │   │   │     ├─ Compare to target_pattern
│       │   │   │     │
│       │   │   │     └─ Score:
│       │   │   │        if "meat" in active:
│       │   │   │          correct += 1
│       │   │   │
│       │   │   ├─► [D] Reinforcement Phase
│       │   │   │     │
│       │   │   │     ├─ If correct:
│       │   │   │     │  └─ LTP: Strengthen used synapses
│       │   │   │     │     weight *= 1.05
│       │   │   │     │
│       │   │   │     └─ If incorrect:
│       │   │   │        └─ LTD: Weaken active synapses
│       │   │   │           weight *= 0.95
│       │   │   │
│       │   │   └─► [E] Reset
│       │   │         └─ memory.reset_energy()
│       │   │
│       │   └─ Epoch metrics:
│       │      accuracy = correct / total
│       │
│       └─ Log progress
│
└─► [4] Evaluation
        ├─ Final accuracy: 95.3%
        ├─ Save trained weights
        └─ Generate report
```

## Summary

These workflows demonstrate:

1. **Deterministic Execution**: Fixed phase order in System 1
2. **Event-Driven Processing**: Surprise triggers System 2
3. **State Machine Design**: Clear dialogue states
4. **Staged Ingestion**: Safe bulk data import
5. **Real-time Integration**: WebSocket for live updates
6. **Curriculum Learning**: Structured training protocol

---

*For more details on implementation, see source code and [ARCHITECTURE.md](ARCHITECTURE.md).*
