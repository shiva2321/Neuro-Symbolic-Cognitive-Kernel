NSCK → AGI Evolution Task Tracker
Current Focus: Phase 6 - Integration & Emergence

Phase 0 - Planning & Research Foundation
Planning
 Complete project analysis
 Gap analysis against AGI goals
 Create detailed implementation plan with proof points
 Create context management for multi-session continuity
User: Research topics via NotebookLM (intrinsic motivation, global workspace, causal discovery, world models)
 Create milestone verification criteria
Phase 1: Autonomous Learning (Weeks 1-4)
 1.1 Implement intrinsic motivation system (ICM + Count-Based)
 1.X Implement modular Teacher Interface (Architectural Request)
 1.2 Remove teacher dependency (Verified NullTeacher)
 1.3 Build curiosity-driven exploration (Verified Logs)
 1.4 Create self-curriculum learning (Integrated Tracker)
Proof Point: Agent learns Snake from scratch with NO teacher (Verified)

Phase 2: Multi-Step Reasoning (Weeks 5-8)
 2.1 Implement forward chaining engine (Verified 
forward_chain_multi
)
 2.2 Build planning module (STRIPS-style) (Verified 
STRIPSPlanner
 with BFS)
 2.3 Add goal decomposition
 2.4 Create hierarchical plan execution
Proof Point: Agent plans 5+ step sequences for Maze navigation (Verified Plan Caching)

Phase 3: Self-Model & Metacognition (Weeks 9-12)
 3.1 Implement Global Workspace architecture
Proof Point: Global Workspace verified with 
test_global_workspace.py

 3.2 Build self-performance model
Proof Point: Verified SelfModel prediction and calibration in test_self_model.py

 3.3 Add uncertainty calibration
Proof Point: 
CognitiveEngine
 updates and uses SelfModel confidence (Verified in test_integrated_metacognition.py)

 3.4 Create metacognitive monitoring
Proof Point: Agent accurately predicts own success and VETOES low-confidence actions (Verified in test_metacognitive_veto.py)

Phase 4: Causal Understanding (Weeks 13-16)
 4.1 Implement causal discovery algorithm
Proof Point: Agent discovers game rules from pure observation (Verified in test_causal_discovery.py)

 4.2 Build interventional learning
Proof Point: Agent proactively tests unknown causal links (Verified in 
test_interventional_learning.py
)

 4.3 Add counterfactual reasoning
Proof Point: Agent accurately predicts outcomes of alternative actions (Verified in 
test_counterfactuals.py
)

 4.4 Create theory formation
Proof Point: Agent generalizes context-specific rules into abstract schemas (Verified in 
test_theory_formation.py
)

Phase 5: Generative Imagination (Weeks 17-20)
 5.1 Train world model (dynamics predictor)
Proof Point: World Model learns $(s_t, a_t) \rightarrow (s_{t+1}, r_{t+1})$ with high accuracy (Verified in 
test_world_model.py
)

 5.2 Implement mental simulation
Proof Point: Agent vetoes dangerous plans and actions using imagined rewards (Verified in 
test_imagination.py
)

 5.3 Add generative dreaming
 5.4 Create hypothetical scenario generation
Proof Point: Agent plans in imagined states, not just real ones (Verified in 
test_hypothetical_scenarios.py
)

Phase 6: Integration & Emergence (Weeks 21-24)
 6.1 Unified cognitive loop
 6.2 Cross-module communication
 6.3 Emergence testing
 6.4 Novel behavior validation
Proof Point: Agent solves completely new game without ANY prior training

Phase 7: Mission Control & Supervisor Revamp
 7.1 Implement Mission Orchestrator (Task Assignment, Goals, Teacher Toggles)
 7.2 Implement AGI Comparative Analytics (SNN vs Symbolic, Memory, Consciousness)
 7.3 Implement Causal & Planning Oversight (Tree View, Induced Links)
 7.4 Implement Structured Insight Logging (Trace Exports, Semantic Console)
 7.5 Protocol Expansion (Python Server Telemetry & Commands)
Research Queue (for NotebookLM)
 Intrinsic motivation systems (ICM, RND, empowerment)
 Global Workspace Theory implementation
 PC algorithm for causal discovery
 World models (MuZero, Dreamer)
 Neural-symbolic integration patterns