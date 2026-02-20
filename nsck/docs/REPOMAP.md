# NSCK Repository Map

A complete Mermaid map showing every file in the codebase, how they connect to each other, what each provides, and the data flow between subsystems.

---

## Complete Codebase Structure

```mermaid
graph TB
    subgraph Root["Node_network/"]
        README_root["README.md"]
        REQ["requirements.txt"]
        PYTEST["pytest.ini"]
        LICENSE["LICENSE"]
    end

    subgraph NSCK["nsck/"]
        subgraph PythonCore["python/core/ — 41 files, ~17,300 LOC"]
            subgraph VSA["vsa/"]
                HVPy["hypervec_py.py<br/>361 LOC<br/>━━━━━━━━━━<br/>HyperVectorPy<br/>CleanupMemory"]
                HVShim["hypervec_shim.py<br/>320 LOC<br/>━━━━━━━━━━<br/>Backend selector<br/>Rust ↔ Python"]
            end

            subgraph ReasoningDir["reasoning/"]
                CogEng["cognitive_engine.py<br/>1,188 LOC<br/>━━━━━━━━━━<br/>CognitiveEngine<br/>CognitiveState<br/>Proposal"]
                GW["global_workspace.py<br/>312 LOC<br/>━━━━━━━━━━<br/>GlobalWorkspace<br/>Coalition<br/>WorkspaceModule"]
                Plan["planner.py<br/>296 LOC<br/>━━━━━━━━━━<br/>STRIPSPlanner<br/>PlanStep"]
                CausalR["causal_reasoning.py<br/>954 LOC<br/>━━━━━━━━━━<br/>CausalDiscovery<br/>CausalGraph<br/>CausalReasoner"]
                RuleL["rule_learner.py<br/>624 LOC<br/>━━━━━━━━━━<br/>RuleLearner<br/>RuleCandidate"]
                AnalE["analogy.py<br/>528 LOC<br/>━━━━━━━━━━<br/>AnalogyEngine<br/>Analogy"]
                CtxE["context_engine.py<br/>440 LOC<br/>━━━━━━━━━━<br/>ContextEngine<br/>ContextFrame"]
            end

            subgraph LearningDir["learning/"]
                Hebb["hebbian.py<br/>506 LOC<br/>━━━━━━━━━━<br/>HebbianMatrixNumPy<br/>VSAHebbianLearner"]
                Curio["curiosity.py<br/>336 LOC<br/>━━━━━━━━━━<br/>CuriosityModule<br/>ExplorationDecision"]
            end

            subgraph CognitiveDir["cognitive/"]
                Meta["metacognition.py<br/>504 LOC<br/>━━━━━━━━━━<br/>SafetyGate<br/>MetacognitiveEngine"]
                SelfM["self_model.py<br/>255 LOC<br/>━━━━━━━━━━<br/>SelfModel"]
                ToM["theory_of_mind.py<br/>312 LOC<br/>━━━━━━━━━━<br/>TheoryOfMind<br/>MentalStateModel"]
                Emo["emotion_system.py<br/>420 LOC<br/>━━━━━━━━━━<br/>EmotionSystem"]
            end

            subgraph PerceptionDir["perception/"]
                SNNPerc["snn_perception.py<br/>561 LOC<br/>━━━━━━━━━━<br/>SNNPerceptionModule<br/>LIFNeuronLayer"]
                SNNInt["snn_integration.py<br/>240 LOC<br/>━━━━━━━━━━<br/>SNNWorkspaceAdapter"]
                Bridge["vsa_snn_bridge.py<br/>461 LOC<br/>━━━━━━━━━━<br/>RateCoder<br/>TemporalCoder<br/>HVtoSpikeDecoder"]
                Ground["grounding_verifier.py<br/>585 LOC<br/>━━━━━━━━━━<br/>GroundingVerifier<br/>GroundingResult"]
                SymGround["symbol_grounding.py<br/>196 LOC<br/>━━━━━━━━━━<br/>Abstract grounding"]
            end

            subgraph MemoryDir["memory/"]
                EpiMem["episodic_memory.py<br/>398 LOC<br/>━━━━━━━━━━<br/>EpisodicMemory<br/>LiveEpisode"]
                SemMem["semantic_memory.py<br/>217 LOC<br/>━━━━━━━━━━<br/>SemanticMemory"]
                StagedR["staged_recall.py<br/>134 LOC<br/>━━━━━━━━━━<br/>StagedRecall"]
            end

            subgraph LangDir["language/"]
                LangMod["language_module.py<br/>484 LOC<br/>━━━━━━━━━━<br/>LanguageModule"]
                Dialog["dialogue_manager.py<br/>145 LOC<br/>━━━━━━━━━━<br/>DialogueManager"]
                TKL["text_knowledge_learner.py<br/>1,071 LOC<br/>━━━━━━━━━━<br/>TextKnowledgeLearner"]
                UniIn["universal_input.py<br/>947 LOC<br/>━━━━━━━━━━<br/>UniversalInput"]
                Lingua["lingua_cortex.py<br/>260 LOC<br/>━━━━━━━━━━<br/>LinguaCortex<br/>SemanticFingerprint"]
            end

            subgraph IntegDir["integration/"]
                Config["config.py<br/>73 LOC<br/>━━━━━━━━━━<br/>NSCKConfig"]
                Persist["persistence.py<br/>852 LOC<br/>━━━━━━━━━━<br/>BrainStore"]
                BFusion["brain_fusion.py<br/>481 LOC<br/>━━━━━━━━━━<br/>BrainFusion<br/>FusedBrain"]
                Explain["explanation.py<br/>433 LOC<br/>━━━━━━━━━━<br/>ExplanationGenerator"]
                KI["knowledge_integration.py<br/>530 LOC<br/>━━━━━━━━━━<br/>KnowledgeIntegration"]
            end

            subgraph TrainingDir["training/"]
                SNNTrain["snn_training.py<br/>589 LOC<br/>━━━━━━━━━━<br/>SNNTrainer"]
                SNNBench["snn_benchmarks.py<br/>483 LOC<br/>━━━━━━━━━━<br/>Benchmark suite"]
            end
        end

        subgraph RustVSA["rust_vsa/ — 7 files, ~2,665 LOC"]
            RLib["lib.rs (280)<br/>HyperVector + PyO3"]
            RSem["semantic.rs (444)<br/>SemanticMemoryConcurrent"]
            REpi["episodic.rs (427)<br/>EpisodicMemoryConcurrent"]
            RWork["worker_pool.rs (488)<br/>CognitiveWorkerPool"]
            RConc["concurrent.rs (319)<br/>Lock-free structures"]
            RPers["persistence.rs (416)<br/>PersistentStorage"]
            RAsync["async_runtime.rs (291)<br/>AsyncCognitiveRuntime"]
        end

        subgraph Tests["tests/ — 40 files, ~9,100 LOC"]
            TUnit["unit/"]
            TInteg["integration/"]
            TCore["core_architecture/"]
            TExp["experiments/"]
            TReg["regression/"]
        end

        subgraph Docs["docs/"]
            DocArch["ARCHITECTURE.md"]
            DocForm["FORMULAS.md"]
            DocWork["WORKFLOWS.md"]
            DocRef["MODULE_REFERENCE.md"]
            DocMap["REPOMAP.md"]
            DocTrans["TRANSPARENCY_GUARANTEE.md"]
            DocSNN["SNN_FINAL_SUMMARY.md"]
            DocAnalysis["ANALYSIS_AND_USECASES.md"]
        end

        subgraph Data["data/"]
            DBeliefA["belief_A.txt"]
            DBeliefB["belief_B.txt"]
            DLong["long_chain.txt"]
            DSound["sound_physics.txt"]
            DXylo["xylophone_planets.txt"]
        end

        Examples["examples/<br/>quickstart.py<br/>learn_from_text.py<br/>custom_module.py"]

        Archive["archive/<br/>Historical experiments<br/>game implementations<br/>benchmarks"]
    end

    subgraph AIModel["nsck_ai_model/ — Separate System"]
        AIEng["ai_engine.py"]
        AITrain["train.py"]
        AIAuto["autonomous_trainer.py"]
        AIEval["evaluator.py"]
        AIDash["dashboard.py"]
    end
```

---

## Module Dependency Graph

Every arrow is a verified `import` statement. The `CognitiveEngine` is the central hub.

```mermaid
graph TD
    %% VSA Foundation (everything depends on this)
    HVShim["hypervec_shim"]
    HVPy["hypervec_py"]
    
    HVShim -->|"fallback"| HVPy
    
    %% Central Hub
    CE["cognitive_engine"]
    
    %% CE → all imports
    CE -->|"import"| HVShim
    CE -->|"import"| Config["config"]
    CE -->|"import"| Explain["explanation"]
    CE -->|"import"| Ground["grounding_verifier"]
    CE -->|"import"| RuleL["rule_learner"]
    CE -->|"import"| EpiMem["episodic_memory"]
    CE -->|"import"| SemMem["semantic_memory"]
    CE -->|"import"| Curio["curiosity"]
    CE -->|"import"| AnalE["analogy"]
    CE -->|"import"| Plan["planner"]
    CE -->|"import"| GW["global_workspace"]
    CE -->|"import"| CausalR["causal_reasoning"]
    CE -->|"import"| SelfM["self_model"]
    CE -->|"import"| ToM["theory_of_mind"]
    CE -->|"import"| Meta["metacognition"]
    CE -->|"import"| Persist["persistence"]
    CE -->|"import"| BFusion["brain_fusion"]
    CE -->|"import"| UniIn["universal_input"]
    CE -->|"import"| LangMod["language_module"]
    CE -->|"import"| Dialog["dialogue_manager"]
    
    %% Cross-module imports
    RuleL -->|"import"| HVShim
    RuleL -->|"import"| Persist
    RuleL -->|"import"| Ground
    RuleL -->|"import"| GW
    
    AnalE -->|"import"| HVShim
    
    CtxE["context_engine"] -->|"import"| HVShim
    
    Curio -->|"import"| HVShim
    
    Meta -->|"import"| HVShim
    Meta -->|"import"| BFusion
    
    SelfM -->|"import"| HVShim
    
    Emo["emotion_system"] -->|"import"| HVShim
    
    SNNPerc["snn_perception"] -->|"import"| Bridge["vsa_snn_bridge"]
    SNNPerc -->|"import"| Hebb["hebbian"]
    SNNPerc -->|"import"| HVShim
    
    Bridge -->|"import"| HVShim
    
    SNNInt["snn_integration"] -->|"import"| SNNPerc
    SNNInt -->|"import"| GW
    SNNInt -->|"import"| SemMem
    
    SNNTrain["snn_training"] -->|"import"| SNNPerc
    SNNTrain -->|"import"| HVShim
    
    EpiMem -->|"import"| HVShim
    EpiMem -->|"import"| Persist
    
    SemMem -->|"import"| HVShim
    
    StagedR["staged_recall"] -->|"import"| HVShim
    
    LangMod -->|"import"| Lingua["lingua_cortex"]
    
    TKL["text_knowledge_learner"] -->|"import"| HVShim
    TKL -->|"import"| Lingua
    TKL -->|"import"| SemMem
    TKL -->|"import"| EpiMem
    TKL -->|"import"| CtxE
    TKL -->|"import"| CausalR
    TKL -->|"import"| LangMod
    
    UniIn -->|"import"| HVShim
    
    BFusion -->|"import"| HVShim
    
    KI["knowledge_integration"] -->|"import"| HVShim
    KI -->|"import"| SemMem
    KI -->|"import"| EpiMem
    KI -->|"import"| CtxE
    KI -->|"import"| CausalR
    
    Plan -.->|"reads"| CausalR
```

---

## Data Flow Between Subsystems

```mermaid
graph LR
    subgraph Input["Inputs"]
        StateDict["State Dict<br/>{key: value}"]
        TextInput["Text Input<br/>string"]
        SensoryInput["Sensory Input<br/>numpy array"]
    end

    subgraph VSA["VSA Layer"]
        HV["10,240-bit<br/>HyperVector"]
    end

    subgraph Grounding["Grounding"]
        Preds["Active Predicates<br/>List of strings"]
        SitHV["Situation HV"]
    end

    subgraph Reasoning["Reasoning"]
        Coalitions["6 Coalition Sources"]
        Winner["Winning Action"]
    end

    subgraph Learning["Learning"]
        QTable["Q-Value Table"]
        Rules["Learned Rules"]
        CGraph["Causal Graph"]
    end

    subgraph Memory["Memory"]
        Episodes["Episodes"]
        Concepts["Concept Graph"]
    end

    subgraph Output["Outputs"]
        Action["Action string"]
        Explanation["NL Explanation"]
        Confidence["Confidence float"]
    end

    StateDict --> Preds -->|"bundle"| SitHV --> HV
    TextInput -->|"encode"| HV
    SensoryInput -->|"SNN"| HV

    HV --> Coalitions
    QTable --> Coalitions
    Rules --> Coalitions
    Episodes --> Coalitions
    
    Coalitions -->|"GWT compete"| Winner
    
    Winner --> Action
    Winner --> Explanation
    Winner --> Confidence
    
    Winner -->|"TD(0)"| QTable
    Winner -->|"observe"| Rules
    Winner -->|"observe"| CGraph
    Winner -->|"record"| Episodes
    Winner -->|"add_concept"| Concepts
```

---

## GWT Broadcast Network

```mermaid
graph TD
    GWT["GlobalWorkspace<br/>broadcast()"]
    
    GWT -->|"receive_broadcast()"| RL["RuleLearner<br/>(WorkspaceModule)"]
    GWT -->|"receive_broadcast()"| EpiA["EpisodicBroadcast<br/>Adapter"]
    GWT -->|"receive_broadcast()"| SemA["SemanticBroadcast<br/>Adapter"]
    GWT -->|"receive_broadcast()"| SNNA["SNNWorkspace<br/>Adapter<br/>(when registered)"]
```

---

## Rust ↔ Python Interface

```mermaid
graph LR
    subgraph Python["Python Side"]
        Shim["hypervec_shim.py"]
        Core["All 41 core modules"]
    end

    subgraph Compiled["Compiled Extension"]
        PyO3["PyO3 Bindings<br/>lib.rs"]
    end

    subgraph Rust["Rust Side"]
        RHV["HyperVector"]
        RSem["SemanticMemoryConcurrent"]
        REpi["EpisodicMemoryConcurrent"]
        RPool["CognitiveWorkerPool"]
        RPers["PersistentStorage"]
        RAsync["AsyncCognitiveRuntime"]
    end

    Core -->|"import"| Shim
    Shim -->|"import hypervec_rs"| PyO3
    PyO3 --> RHV
    PyO3 --> RSem
    PyO3 --> REpi
    PyO3 --> RPool
    PyO3 --> RPers
    PyO3 --> RAsync

    Shim -.->|"fallback if Rust unavailable"| FallPy["hypervec_py.py<br/>Pure Python"]
```

---

## Test Coverage Map

```mermaid
graph TD
    subgraph TestSuite["Test Suite — 40 files, ~9,100 LOC"]
        subgraph Unit["unit/"]
            TU1["test_analogy.py"]
            TU2["test_causal.py"]
            TU3["test_context_engine.py"]
            TU4["test_curiosity.py"]
            TU5["test_emotion.py"]
            TU6["test_episodic.py"]
            TU7["test_global_workspace.py"]
            TU8["test_grounding.py"]
            TU9["test_hebbian.py"]
            TU10["test_hypervec.py"]
            TU11["test_language.py"]
            TU12["test_metacognition.py"]
            TU13["test_planner.py"]
            TU14["test_rule_learner.py"]
            TU15["test_self_model.py"]
            TU16["test_semantic_memory.py"]
            TU17["test_snn_perception.py"]
            TU18["test_text_knowledge.py"]
            TU19["test_theory_of_mind.py"]
            TU20["test_universal_input.py"]
        end

        subgraph Integration["integration/"]
            TI1["test_cognitive_engine.py"]
            TI2["test_full_pipeline.py"]
            TI3["test_persistence.py"]
            TI4["test_snn_integration.py"]
            TI5["test_realworld_capabilities.py"]
        end

        subgraph CoreArch["core_architecture/"]
            TC1["test_architecture_compliance.py"]
            TC2["test_cross_module.py"]
            TC3["test_transparency.py"]
        end

        subgraph Experiments["experiments/"]
            TE1["test_belief_revision.py"]
            TE2["test_causal_chains.py"]
            TE3["test_text_learning.py"]
        end
    end

    subgraph Modules["Core Modules"]
        M1["41 Python files"]
    end

    Unit -->|"covers"| M1
    Integration -->|"covers"| M1
    CoreArch -->|"verifies"| M1
    Experiments -->|"validates"| M1
```

---

## File Size Distribution

| Size Range | Count | Files |
|-----------|-------|-------|
| **1000+ LOC** | 3 | cognitive_engine (1,188), text_knowledge_learner (1,071), causal_reasoning (954) |
| **500–999 LOC** | 10 | universal_input (947), persistence (852), rule_learner (624), snn_training (589), grounding_verifier (585), snn_perception (561), knowledge_integration (530), analogy (528), hebbian (506), metacognition (504) |
| **200–499 LOC** | 13 | language_module (484), snn_benchmarks (483), brain_fusion (481), vsa_snn_bridge (461), context_engine (440), explanation (433), emotion_system (420), episodic_memory (398), hypervec_py (361), curiosity (336), hypervec_shim (320), global_workspace (312), theory_of_mind (312) |
| **< 200 LOC** | 7 | planner (296), lingua_cortex (260), self_model (255), snn_integration (240), semantic_memory (217), symbol_grounding (196), dialogue_manager (145), staged_recall (134), config (73) |

**Total: 41 Python files, ~17,300 LOC** + **7 Rust files, ~2,665 LOC** = **~20,000 LOC**
