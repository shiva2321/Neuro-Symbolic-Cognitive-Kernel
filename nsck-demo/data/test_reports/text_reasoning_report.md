# NSCK Text Reasoning Benchmark Report

**Date:** 2026-02-12 23:24:30
**Score:** 5/6 (83.3%)

## 1. Learning Session
- **Corpus:** Xylophone Planets (Synthetic)
- **Concepts Learned:** 115
- **Relations Extracted:** 313
- **Facts Stored:** 313

## 2. Test Results

| ID | Type | Query | Result | Confidence | Keywords Found |
|----|------|-------|--------|------------|----------------|
| F1 | Fact Recall (A) | What are Xylophone planets made of? | ✅ PASS | 0.86 | crystal, singing |
| F2 | Fact Recall (B) | What is resonance? | ✅ PASS | 0.86 | phenomenon, vibration, match, natural |
| R1 | Relation | What does resonance attract? | ❌ FAIL | 0.86 | None |
| X1 | Cross-Domain | Do space_whales consume energy? | ✅ PASS | 0.86 | sound, energy |
| X2 | Cross-Domain | Are space_whales attracted to vibrations? | ✅ PASS | 0.86 | resonance, vibration |
| N1 | Inference | What happens if the surface cracks? | ✅ PASS | 0.86 | stop, resonance |

### Detailed Responses
#### F1: What are Xylophone planets made of?
> **System Answer:**
**Relevant Concepts:**
- Stops (similarity: 0.515)
- Dangerous (similarity: 0.515)
- Tuning_fork (similarity: 0.514)

**Learned Facts:**
- Ecosystem is_a Xylophone
  Source: 'd:\Node_network\nsck-demo\pyth......'
- Made is_a Crystal
  Source: 'd:\Node_network\nsck-demo\pyth......'
- Xylophone is_a Asta
  Source: 'd:\Node_network\nsck-demo\pyth......'
- Xylophone is_a Rare
  Source: 'd:\Node_network\nsck-demo\pyth......'
- Celestial Bodies Made
  Source: 'd:\Node_network\nsck-demo\pyth......'

**Associated Concepts:**
- Crystal (activation: 9.379)
- Singing (activation: 8.625)
- Entirely (activation: 5.359)
- Frequency (activation: 4.371)
- Made (activation: 3.863)

> **Reasoning Trace:**
- Query encoded to hypervector
- Extracted query concepts: ['What', 'Xylophone', 'Planets', 'Made']
- Found 5 similar concepts in semantic memory
- Activated 5 related concepts via spreading activation
- Recalled 5 relevant episodes
- Found 29 related facts

---
#### F2: What is resonance?
> **System Answer:**
**Relevant Concepts:**
- Pitch (similarity: 0.513)
- Grow (similarity: 0.511)
- Discordant (similarity: 0.509)

**Learned Facts:**
- Resonance is_a Vibration
  Source: 'd:\Node_network\nsck-demo\pyth......'
- Resonance is_a Matches
  Source: 'd:\Node_network\nsck-demo\pyth......'
- Resonance is_a Natural
  Source: 'd:\Node_network\nsck-demo\pyth......'
- Resonance is_a External
  Source: 'd:\Node_network\nsck-demo\pyth......'
- Resonance is A phenomenon where
  Source: 'd:\Node_network\nsck-demo\pyth......'

**Associated Concepts:**
- Frequency (activation: 4.161)
- Speed (activation: 3.334)
- Natural (activation: 2.119)
- Matches (activation: 1.363)
- Resonance (activation: 1.000)

> **Reasoning Trace:**
- Query encoded to hypervector
- Extracted query concepts: ['What', 'Resonance']
- Found 5 similar concepts in semantic memory
- Activated 5 related concepts via spreading activation
- Recalled 5 relevant episodes
- Found 14 related facts

---
#### R1: What does resonance attract?
> **System Answer:**
**Relevant Concepts:**
- Types (similarity: 0.514)
- Natural (similarity: 0.513)
- Hz

Sound (similarity: 0.511)

**Learned Facts:**
- Resonance is_a Vibration
  Source: 'd:\Node_network\nsck-demo\pyth......'
- Resonance is_a Matches
  Source: 'd:\Node_network\nsck-demo\pyth......'
- Resonance is_a Natural
  Source: 'd:\Node_network\nsck-demo\pyth......'
- Resonance is_a External
  Source: 'd:\Node_network\nsck-demo\pyth......'
- Resonance is A phenomenon where
  Source: 'd:\Node_network\nsck-demo\pyth......'

**Associated Concepts:**
- Frequency (activation: 4.161)
- Speed (activation: 3.334)
- Matches (activation: 1.363)
- Resonance (activation: 1.000)

> **Reasoning Trace:**
- Query encoded to hypervector
- Extracted query concepts: ['What', 'Does', 'Resonance', 'Attract']
- Found 5 similar concepts in semantic memory
- Activated 5 related concepts via spreading activation
- Recalled 5 relevant episodes
- Found 14 related facts

---
#### X1: Do space_whales consume energy?
> **System Answer:**
**Relevant Concepts:**
- Hums (similarity: 0.510)
- Into (similarity: 0.510)
- Natural (similarity: 0.509)

**Learned Facts:**
- Sound is_a Energy
  Source: 'd:\Node_network\nsck-demo\pyth......'
- Form is_a Energy
  Source: 'd:\Node_network\nsck-demo\pyth......'
- Surface leads_to The resonance stops does
  Source: 'd:\Node_network\nsck-demo\pyth......'
- Cracks leads_to The resonance stops does
  Source: 'd:\Node_network\nsck-demo\pyth......'
- If the surface cracks implies The resonance stops does
  Source: 'd:\Node_network\nsck-demo\pyth......'

**Associated Concepts:**
- Sound (activation: 5.177)
- Energy (activation: 3.400)
- Pure (activation: 2.481)
- Dr (activation: 1.518)
- Form (activation: 1.518)

> **Reasoning Trace:**
- Query encoded to hypervector
- Extracted query concepts: ['Do', 'Space_whales', 'Consume', 'Energy']
- Found 5 similar concepts in semantic memory
- Activated 5 related concepts via spreading activation
- Recalled 5 relevant episodes
- Found 13 related facts

---
#### X2: Are space_whales attracted to vibrations?
> **System Answer:**
**Relevant Concepts:**
- Surface (similarity: 0.513)
- Fades (similarity: 0.511)
- Music (similarity: 0.511)

**Learned Facts:**
- Collection is_a Vibrations
  Source: 'd:\Node_network\nsck-demo\pyth......'
- Collection Disordered Vibrations
  Source: 'd:\Node_network\nsck-demo\pyth......'
- Disordered is_a Vibrations
  Source: 'd:\Node_network\nsck-demo\pyth......'
- Noise is_a Vibrations
  Source: 'd:\Node_network\nsck-demo\pyth......'
- Resonance is_a Vibration
  Source: 'd:\Node_network\nsck-demo\pyth......'

**Associated Concepts:**
- Sound (activation: 5.177)
- Pure (activation: 2.481)
- Energy (activation: 2.400)
- Dr (activation: 1.518)
- Form (activation: 1.518)

> **Reasoning Trace:**
- Query encoded to hypervector
- Extracted query concepts: ['Are', 'Space_whales', 'Attracted', 'Vibrations']
- Found 5 similar concepts in semantic memory
- Activated 5 related concepts via spreading activation
- Recalled 5 relevant episodes
- Found 47 related facts

---
#### N1: What happens if the surface cracks?
> **System Answer:**
**Relevant Concepts:**
- Without (similarity: 0.512)
- Star_dust (similarity: 0.512)
- Celestial (similarity: 0.511)

**Learned Facts:**
- Surface leads_to The resonance stops does
  Source: 'd:\Node_network\nsck-demo\pyth......'
- Cracks leads_to The resonance stops does
  Source: 'd:\Node_network\nsck-demo\pyth......'
- If the surface cracks implies The resonance stops does
  Source: 'd:\Node_network\nsck-demo\pyth......'
- Discordant Vibrations Crack
  Source: 'd:\Node_network\nsck-demo\pyth......'
- Star_dust causes Surface
  Source: 'd:\Node_network\nsck-demo\pyth......'

**Associated Concepts:**
- The resonance stops does (activation: 5.040)
- Grow (activation: 5.027)
- Crystals (activation: 2.294)
- New (activation: 1.330)
- Surface (activation: 1.000)

> **Reasoning Trace:**
- Query encoded to hypervector
- Extracted query concepts: ['What', 'Happens', 'Surface', 'Cracks']
- Found 5 similar concepts in semantic memory
- Activated 5 related concepts via spreading activation
- Recalled 5 relevant episodes
- Found 12 related facts

---

## 3. Emergent Knowledge
Relations discovered via Semantic Folding (vector similarity) rather than explicit text patterns:

