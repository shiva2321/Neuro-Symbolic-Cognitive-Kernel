# NSCK AI Model — Architecture Deep Dive

## Why This Architecture?

The user asked for an AI model that:
1. Uses the **existing NSCK architecture** (not a reimplementation)
2. No transformers or neural networks (no heavy matrix multiplication)
3. Trains on real data (HuggingFace WikiText)
4. Understands and responds in natural language
5. Glass-box: every thought traceable end-to-end
6. No hardcoded patterns — everything learned from data

## System Overview

```
┌─────────────────────────────────────────────────────────┐
│                    NSCKAIEngine                          │
│                                                         │
│  ┌──────────────┐   ┌──────────────┐   ┌─────────────┐ │
│  │TextKnowledge │   │  Semantic    │   │  Episodic   │ │
│  │   Learner    │──▶│   Memory    │   │   Memory    │ │
│  │(lingua_cortex│   │ (NetworkX)  │   │ (VSA+LSH)   │ │
│  └──────────────┘   └──────┬───────┘   └──────┬──────┘ │
│                            │                   │        │
│  ┌──────────────┐   ┌──────▼───────┐   ┌──────▼──────┐ │
│  │   Causal     │   │   Global     │   │  Curiosity  │ │
│  │  Reasoner    │──▶│  Workspace   │◀──│   Module    │ │
│  │(CausalGraph) │   │(Competition) │   │ (Novelty)   │ │
│  └──────────────┘   └──────┬───────┘   └─────────────┘ │
│                            │                            │
│  ┌──────────────┐   ┌──────▼───────┐   ┌─────────────┐ │
│  │   Emotion    │   │  Response    │   │    Self     │ │
│  │   System     │   │ Generation  │   │   Model     │ │
│  │ (Plutchik)   │   │  (N-gram)   │   │(Confidence) │ │
│  └──────────────┘   └─────────────┘   └─────────────┘ │
│                                                         │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              ThoughtTrace (Glass-Box)               │ │
│  │  Records every step for full traceability           │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## Module Details

### 1. TextKnowledgeLearner (Text Learning)

**Source**: `nsck-demo/python/core/language/text_knowledge_learner.py`

**What it does**: Extracts concepts and relations from text using semantic
folding (LinguaCortex) — NOT an LLM.

**How it works**:
- Text → sentences → tokenisation
- Concepts extracted (capitalised nouns, multi-word phrases)
- Relations discovered via co-occurrence patterns and semantic folding
- Each concept gets a HyperVector (10,240-bit binary)
- Concepts stored in SemanticMemory graph
- Episodes stored in EpisodicMemory

**Why we use it**: This is the NSCK's actual text learning module. It
uses VSA (not neural networks) to learn from text.

### 2. SemanticMemory (Concept Graph)

**Source**: `nsck-demo/python/core/memory/semantic_memory.py`

**What it does**: Stores concepts as nodes and relations as edges in a
NetworkX directed graph.  Each concept also has a HyperVector for
similarity-based retrieval.

**Key operations**:
- `add_concept(name, properties, hv_override)` — Add a concept node
- `add_relation(concept1, relation, concept2)` — Add an edge
- `query(query_hv, k)` — Find k most similar concepts by HV
- `spread_activation(seeds, steps, decay)` — Associative retrieval

**Why we use it**: This is the NSCK's actual knowledge graph.  Spreading
activation is how the system finds related concepts.

### 3. EpisodicMemory (Experience Storage)

**Source**: `nsck-demo/python/core/memory/episodic_memory.py`

**What it does**: Stores episodes (experiences) with HyperVector indexing
and LSH (Locality-Sensitive Hashing) for fast retrieval.

**Why we use it**: Episodic memory allows the system to recall similar
past experiences, which improves response quality.

### 4. GlobalWorkspace (Consciousness Model)

**Source**: `nsck-demo/python/core/reasoning/global_workspace.py`

**What it does**: Implements Global Workspace Theory (LIDA architecture).
Multiple modules compete for access to a global broadcast channel.

**How it works in chat**:
- Each retrieval channel (semantic, episodic, causal, activation)
  creates a Coalition object
- Each Coalition has base_salience, relevance, affect_match, confidence
- activation = salience + relevance + affect + 0.5*confidence
- The highest-activation Coalition wins and determines the response

**Why we use it**: This gives the system a principled way to decide
*which* piece of knowledge to use for the response, rather than just
returning the first match.

### 5. EmotionSystem (Plutchik Model)

**Source**: `nsck-demo/python/core/cognitive/emotion_system.py`

**What it does**: Tracks emotional state using Plutchik's 8 basic emotions
(joy, trust, fear, surprise, sadness, disgust, anger, anticipation)
mapped to a 2D valence-arousal space.

**Key operations**:
- `recognize_emotion_from_text(text)` — Classify text emotion
- `get_emotion_blend()` — Weighted mix of all emotions
- `get_mood(window)` — Slow-moving average (mood)
- `get_emotion_hypervector()` — Emotion as a HyperVector

**Why we use it**: Emotion-aware responses are more natural.

### 6. CausalReasoner + CausalGraph

**Source**: `nsck-demo/python/core/reasoning/causal_reasoning.py`

**What it does**: Stores causal links (X causes Y) and performs
forward/backward chaining and counterfactual reasoning.

**How it works in chat**:
- After training on "Rain causes flooding", the system stores
  a causal link: rain → flooding
- When asked "What causes flooding?", the system forward-chains
  from the query concepts to find effects

**Why we use it**: Causal reasoning is essential for "why" and
"what causes" questions.

### 7. SelfModel (Confidence Calibration)

**Source**: `nsck-demo/python/core/cognitive/self_model.py`

**What it does**: Tracks how well the system's predictions match actual
outcomes.  Measures calibration error and improvement trends.

**Why we use it**: Self-awareness tells the system (and the user)
how confident it should be in its answers.

### 8. CuriosityModule (Novelty Detection)

**Source**: `nsck-demo/python/core/learning/curiosity.py`

**What it does**: Measures how novel an input is compared to what the
system has seen before.  Novel inputs get higher curiosity scores.

**Why we use it**: Novelty detection helps the system identify when it's
encountering something new and needs to learn more.

### 9. HyperVector (10,240-bit VSA)

**Source**: `nsck-demo/python/core/vsa/hypervec_shim.py`

**What it does**: The fundamental data structure.  10,240-bit binary
vectors with three operations:
- **XOR binding** — Creates a vector dissimilar to both inputs
- **Majority-rule bundling** — Creates a vector similar to all inputs
- **Hamming similarity** — Measures how similar two vectors are

**Why we use it**: This is the core VSA that makes NSCK work without
matrix multiplication.  All representations (words, concepts, emotions,
episodes) are HyperVectors.

### 10. ResponseGenerator (N-gram Model)

**What it does**: Learns bigram and trigram probabilities from training
text.  Used as a fallback when the retrieval-based approach doesn't
find relevant knowledge.

**Why we use it**: Provides a safety net for generating grammatical
text when specific knowledge isn't available.

## Data Flow

```
Training:
  Text → TextKnowledgeLearner → SemanticMemory (graph)
                               → EpisodicMemory (episodes)
                               → CausalGraph (causal links)
                               → ResponseGenerator (n-grams)
                               → SentenceIndex (response candidates)

Chat:
  Input → Encode (HV) → Emotion → Concepts → SemanticSearch
        → SpreadingActivation → KnowledgeQuery → CausalInference
        → Curiosity → GlobalWorkspace → SelfModel → Generate
        → ThoughtTrace (glass-box record)
```

## Why Not Neural Networks?

The NSCK architecture avoids neural networks because:
1. **No matrix multiplication** — HV operations are bitwise (XOR, majority)
2. **No backpropagation** — Learning is incremental (update HVs directly)
3. **No GPU required** — All operations are O(D) where D=10,240
4. **Glass-box** — Every step is inspectable (no hidden layers)
5. **No catastrophic forgetting** — New knowledge is bundled into existing HVs
6. **Real-time learning** — No training epochs, no batch processing
