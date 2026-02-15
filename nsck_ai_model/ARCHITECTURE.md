# NSCK AI Model — Architecture Document

## Overview

The NSCK AI model is a **glass-box**, **zero-hardcode** AI system built on
Vector Symbolic Architecture (VSA).  This document explains every component:
what it does, how it works, and why it was designed that way.

---

## §1  HyperVector — The Foundation

### What
A 10,240-dimensional binary vector.  Every piece of information in the
system — words, sentences, concepts, relations, images — is represented
as a HyperVector (HV).

### How
- **Creation**: Random binary vectors (50% ones, 50% zeros).
- **Binding** (XOR): Creates a new vector dissimilar to both inputs.
  Used to create "word-in-context" representations.
- **Bundling** (majority vote): Creates a vector similar to all inputs.
  Used to combine multiple pieces of information.
- **Permutation** (bit shift): Encodes position/order.

### Why
- All operations are O(D) — no matrix multiplication.
- Self-inverse binding: `A ⊕ B ⊕ B = A`.
- Hamming similarity is a simple bit-count.
- CPU-only, ~150 MB RAM.

---

## §2  TextEncoder — Learned Semantic Folding

### What
Converts text into HVs.  Words that appear in similar contexts develop
similar HVs through training.

### How
1. Each word gets a deterministic base HV from its hash.
2. A context window of ±3 words is bundled with position encoding.
3. The word HV is bound with its context HV → word-in-context.
4. All word-in-context HVs are bundled → sentence HV.
5. During training, context vectors are incrementally updated via
   bundling with new observations.

### Why not use a pre-trained embedding?
- No dependency on external models.
- Incrementally updatable — new words are learned on the fly.
- Distributional semantics emerge naturally from co-occurrence.

---

## §3  KnowledgeStore — Dual Memory

### What
Two complementary memory systems:
- **Semantic memory**: concept graph with learned relations.
- **Episodic memory**: timestamped experiences with LSH index.

### How: Semantic Memory
- Concepts are named HVs with frequency counts and source sentences.
- Relations store (source, target, sentence, sentence_hv).  The sentence
  HV IS the relation type — no hardcoded labels.
- Spreading activation traverses the graph to find related concepts.

### How: Episodic Memory
- Each experience is stored with its HV and timestamp.
- LSH (Locality-Sensitive Hashing) enables O(1) approximate search.
- 16-bit random hyperplane LSH → 65,536 possible buckets.

### Why dual memory?
- Semantic memory stores what the system knows (facts).
- Episodic memory stores what happened (experiences).
- Together they enable both factual recall and experiential learning.

---

## §4  CausalRuleStore — Autonomous Rule Learning

### What
Stores causal rules learned from training data.  Rules link sets of
antecedent concepts to sets of consequent concepts.

### How
- During training, each sentence with ≥2 concepts generates a rule.
- The first half of concepts → antecedent; second half → consequent.
- At query time, forward chaining fires rules whose antecedents match
  the active concepts, potentially triggering chains of inference.
- Rule strength grows logarithmically with evidence count.

### Why no hardcoded causal patterns?
- Any sentence containing multiple concepts is a potential causal rule.
- The sentence HV captures the relationship semantics.
- Forward chaining enables multi-hop reasoning.

---

## §5  KnowledgeAbstractor — Autonomous Generalisation

### What
Creates category concepts from recurring patterns in the data.

### How
1. Groups all relations by target concept.
2. If a target appears in N+ relations, it becomes a "category".
3. The category's HV is the bundle of all members' HVs.
4. The category is stored as a new concept in the knowledge store.

### Why
- Enables answering "What is X?" even if never directly told.
- Creates hierarchical knowledge structure automatically.
- No manual ontology design required.

---

## §6  EmotionTracker — Learned Emotional State

### What
Tracks conversational emotional state using a 2D valence-arousal model.

### How
- During training, positive/negative feedback signals are associated
  with specific HVs.
- At inference time, the input HV is compared to learned positive and
  negative HVs to estimate valence.
- The valence-arousal coordinates are mapped to the nearest of 9
  Plutchik emotions for human-readable display.

### Why learned instead of keyword-based?
- No hardcoded sentiment lexicon.
- Adapts to domain-specific language.
- Emotional classification emerges from data.

---

## §7  ResponseGenerator — Learned N-gram NLG

### What
Generates natural language responses using learned n-gram patterns
and retrieved sentences.

### How
1. Learns bigram and trigram continuation probabilities from training.
2. At inference time, retrieves relevant stored sentences.
3. If needed, extends fragments with n-gram continuations.
4. Weighted random selection from learned probabilities.

### Why not templates?
- Every word traces back to specific training data.
- No hallucination — only says things it actually learned.
- Language style adapts to training corpus.

---

## §8  ThoughtTrace — Glass-Box Traceability

### What
Records every cognitive step in a single reasoning cycle.

### How
Each `chat()` call creates a ThoughtTrace with these stages:
1. **encode** — input text → hypervector
2. **emotion** — update emotional state from input HV
3. **extract_concepts** — identify content words
4. **search_semantic** — find similar concepts by HV similarity
5. **search_episodic** — find similar experiences via LSH
6. **spread_activation** — traverse concept graph
7. **causal_inference** — forward-chain causal rules
8. **generate** — assemble response from retrieved knowledge

Each step records inputs, outputs, and timing in milliseconds.

### Why
- The user can inspect exactly why the model said what it said.
- No black box — every word in the response traces to specific data.
- Dashboard displays the full trace visually.

---

## §9  NSCKAIEngine — The Orchestrator

### What
The main entry point that wires all components together.

### How
- `train_on_text(text)` → learns concepts, relations, rules, n-grams.
- `chat(user_input)` → full cognitive cycle with trace.
- `get_system_stats()` → comprehensive metrics for dashboard.
- `export_knowledge()` → serialise all learned knowledge.

### Why a single orchestrator?
- Single point of entry for both training and inference.
- Ensures all components are updated consistently.
- Makes the system easy to embed in dashboards and APIs.

---

## §10  ImageUnderstanding — Cross-Modal VSA

### What
Encodes images into the same HV space as text for cross-modal retrieval.

### How
Classical CV features (no neural network):
1. **Colour histogram** — pixel intensity distribution → HV.
2. **Edge histogram** — gradient magnitudes → HV.
3. **Spatial layout** — grid cell mean colours → HV.
4. All features bundled into a single image HV.
5. Image HV is bundled with text description HV for joint representation.

### Why not CNN?
- CNNs require heavy matrix multiplication.
- Classical features + VSA is interpretable and CPU-only.
- Cross-modal search works by HV similarity.

---

## Data Flow

```
Training:
  Text → TextEncoder.learn_text()
       → KnowledgeStore.add_concept()
       → KnowledgeStore.add_relation()
       → CausalRuleStore.add_rule()
       → ResponseGenerator.learn()
       → KnowledgeAbstractor.abstract()

Inference:
  Input → TextEncoder.encode_sentence()
        → KnowledgeStore.search_concepts()
        → KnowledgeStore.search_episodes()
        → KnowledgeStore.find_related()
        → CausalRuleStore.forward_chain()
        → _build_response()
        → Natural Language Output
```

---

## Performance

| Metric | Value |
|---|---|
| Encoding latency | ~1ms per sentence |
| Retrieval latency | ~0.3ms per query |
| Total chat latency | 2–5ms |
| Memory per concept | ~10 KB |
| Memory for 10K concepts | ~100 MB |
| HV dimension | 10,240 bits |
| LSH buckets | 65,536 (16-bit) |
