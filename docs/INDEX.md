# NCGN Documentation Index

**Navigation hub for all NCGN documentation**

---

## Quick Navigation

### Getting Started
- **[README.md](../README.md)** — Project overview and quick start (5 min read)
- **[GETTING_STARTED.md](../GETTING_STARTED.md)** — Hands-on tutorial by tier (15 min read)

### Understanding the System
- **[ARCHITECTURE.md](ARCHITECTURE.md)** — Complete 5-tier architecture (20 min read)
- **[API_REFERENCE.md](API_REFERENCE.md)** — API documentation by module (reference)
- **[EXPERIMENTS.md](EXPERIMENTS.md)** — How to run and create experiments (10 min read)

### Contributing
- **[CONTRIBUTING.md](../CONTRIBUTING.md)** — Extension guidelines (15 min read)

### Advanced Topics
- **[FILE_STRUCTURE.md](FILE_STRUCTURE.md)** — Detailed directory organization
- **[archive/](archive/)** — Historical phase completion reports

---

## By Topic

### Learning & Plasticity

| Topic | File | Section |
|-------|------|---------|
| **How neurons fire** | ARCHITECTURE.md | Tier 1: Spiking Neural Substrate |
| **How synapses learn** | ARCHITECTURE.md | Tier 1: 3-Factor STDP |
| **STDP formula** | API_REFERENCE.md | core.plasticity.PlasticityController |
| **When learning happens** | ARCHITECTURE.md | Tier 3: Cognitive Control |
| **Learning rules** | CONTRIBUTING.md | Adding Core Modules |

### Architecture & Organization

| Topic | File | Section |
|-------|------|---------|
| **5-tier overview** | ARCHITECTURE.md | Overview |
| **Neural substrate** | ARCHITECTURE.md | Tier 1 |
| **Orchestration** | ARCHITECTURE.md | Tier 2 |
| **Cognitive control** | ARCHITECTURE.md | Tier 3 |
| **Regions** | API_REFERENCE.md | core.region.Region |
| **Reasoning** | ARCHITECTURE.md | Tier 4 |
| **Goals** | ARCHITECTURE.md | Tier 5 |

### Experiments & Usage

| Topic | File | Section |
|-------|------|---------|
| **Run demos** | GETTING_STARTED.md | Running Your First Demo |
| **Classical conditioning** | GETTING_STARTED.md | Tier 1-2: Classical Conditioning |
| **Sequence learning** | GETTING_STARTED.md | Tier 1-2: Sequence Learning |
| **Novelty detection** | GETTING_STARTED.md | Tier 3: Novelty-Gated Learning |
| **Create experiment** | GETTING_STARTED.md | Creating Your Own Experiment |
| **Experiment template** | CONTRIBUTING.md | Creating New Experiments |
| **Test your code** | CONTRIBUTING.md | Writing Tests |

### API Reference

| Component | File | Type |
|-----------|------|------|
| **Neuron** | API_REFERENCE.md | core.neuron.Neuron |
| **Synapse** | API_REFERENCE.md | core.synapse.Synapse |
| **Plasticity** | API_REFERENCE.md | core.plasticity.PlasticityController |
| **Event Queue** | API_REFERENCE.md | core.event_queue.EventQueue |
| **Dispatcher** | API_REFERENCE.md | core.dispatcher.Dispatcher |
| **Orchestrator** | API_REFERENCE.md | core.orchestrator.Orchestrator |
| **Metrics** | API_REFERENCE.md | core.metrics.MetricsCollector |
| **Control Layer** | API_REFERENCE.md | core.control_layer.ControlLayer |
| **Region** | API_REFERENCE.md | core.region.Region |
| **System 2** | API_REFERENCE.md | core.system2.System2 |
| **Meta-Learner** | API_REFERENCE.md | core.meta_learner.MetaLearner |
| **Self-Model** | API_REFERENCE.md | core.self_model.SelfModel |

---

## By Use Case

### "I'm New to NCGN"
1. Start: [README.md](../README.md)
2. Read: [GETTING_STARTED.md](../GETTING_STARTED.md) — Understanding the 5 Tiers
3. Run: `python -m experiments.phase5_goal_directed_demo`
4. Explore: Each demo mentioned in GETTING_STARTED.md

### "I Want to Understand the Architecture"
1. Skim: [README.md](../README.md) — Project overview
2. Read: [ARCHITECTURE.md](ARCHITECTURE.md) — Deep dive into each tier
3. Reference: [API_REFERENCE.md](API_REFERENCE.md) — How components connect
4. Study: Tests in `tests/` directory for usage examples

### "I Want to Create an Experiment"
1. Template: [CONTRIBUTING.md](../CONTRIBUTING.md) — Creating New Experiments
2. Example: See `experiments/` directory for working examples
3. Reference: [GETTING_STARTED.md](../GETTING_STARTED.md) — Creating Your Own Experiment
4. Test: [CONTRIBUTING.md](../CONTRIBUTING.md) — Writing Tests

### "I Want to Extend the Core"
1. Guidelines: [CONTRIBUTING.md](../CONTRIBUTING.md) — Adding Core Modules
2. Reference: [API_REFERENCE.md](API_REFERENCE.md) — How to design APIs
3. Pattern: Study existing modules in `core/`
4. Test: [CONTRIBUTING.md](../CONTRIBUTING.md) — Testing requirements
5. Document: Update [API_REFERENCE.md](API_REFERENCE.md) with your module

### "I Want to Contribute"
1. Read: [CONTRIBUTING.md](../CONTRIBUTING.md) — Full guidelines
2. Choose: High Priority areas in CONTRIBUTING.md
3. Implement: Following code style and patterns
4. Test: Write comprehensive unit tests
5. Document: Update relevant docs
6. Submit: Pull request with clear description

---

## Document Purposes

### README.md
**What:** Project overview  
**Who:** Everyone (5 min intro)  
**Contains:** Quick start, 5-tier summary, demo commands  
**Update:** When adding major features to project structure

### GETTING_STARTED.md
**What:** Hands-on tutorial  
**Who:** New users and learners (15 min tutorial)  
**Contains:** Installation, tier explanations with code, how to run demos  
**Update:** When adding new demos or phases

### ARCHITECTURE.md
**What:** Design deep dive  
**Who:** Developers and researchers (20 min detailed read)  
**Contains:** 5-tier architecture, design principles, performance  
**Update:** When modifying core architecture or adding new tiers

### API_REFERENCE.md
**What:** Complete API documentation  
**Who:** Developers writing code  
**Contains:** All classes, methods, parameters, examples  
**Update:** When adding/modifying public APIs

### CONTRIBUTING.md
**What:** Extension guidelines  
**Who:** Contributors and advanced users  
**Contains:** How to add experiments, modules, tests, code style  
**Update:** When establishing new contribution patterns

### FILE_STRUCTURE.md
**What:** Directory organization  
**Who:** Developers navigating codebase  
**Contains:** What each folder contains, why organized that way  
**Update:** When restructuring directories

### EXPERIMENTS.md
**What:** Experiment guide (separate doc)  
**Who:** Researchers and experimenters  
**Contains:** How to run existing experiments, parameters to vary  
**Update:** When adding new experiments

---

## Phase History (Archive)

Historical phase completion reports are in `archive/`:

| Phase | Report | Content |
|-------|--------|---------|
| 1 | PHASE1_COMPLETION_REPORT.md | Core neuron/synapse implementation |
| 2 | PHASE2_COMPLETION_REPORT.md | Event-driven orchestration |
| 3.1 | PHASE3_COMPLETION_REPORT.md | Control layer (System 1.5) |
| 3.2 | PHASE32_COMPLETION_REPORT.md | Regional organization |
| 3.3 | PHASE33_COMPLETION_REPORT.md | System 2 reasoning |
| 3.4 | PHASE34_COMPLETION_REPORT.md | Meta-learning |
| 4 | PHASE4_COMPLETION_REPORT.md | Meta-plasticity |
| 5 | PHASE5_COMPLETION_REPORT.md | Goals & self-modeling |

**Note:** These are historical references. For current functionality, see ARCHITECTURE.md.

---

## Reading Recommendations

### For Quick Understanding (30 minutes)
1. README.md overview (5 min)
2. GETTING_STARTED.md "Understanding the 5 Tiers" section (15 min)
3. Run a demo (5 min)
4. Browse API_REFERENCE.md for your tier of interest (5 min)

### For Deep Understanding (2 hours)
1. README.md (10 min)
2. GETTING_STARTED.md complete (30 min)
3. ARCHITECTURE.md (40 min)
4. API_REFERENCE.md quick reference (20 min)
5. Skim CONTRIBUTING.md (10 min)
6. Explore code in `core/` (10 min)

### For Contributing (3-4 hours)
1. Complete "For Deep Understanding" above (2 hours)
2. CONTRIBUTING.md thoroughly (30 min)
3. Study relevant test files (30 min)
4. Study relevant module implementations (30 min)
5. Plan your contribution (30 min)

---

## Quick Commands

```bash
# Installation
git clone https://github.com/yourusername/Node_network.git
cd Node_network

# Run complete system
python -m experiments.phase5_goal_directed_demo

# Run specific demos
python -m experiments.pavlov_experiment        # Tier 1-2
python -m experiments.phase3_demo              # Tier 3
python -m experiments.phase34_meta_learning_demo  # Tier 4

# Run tests
python -m unittest discover -s tests -p "test_*.py" -v

# Create new experiment
# 1. Copy template from experiments/base_experiment.py
# 2. Modify build_network(), create_stimuli(), run_training()
# 3. Run: python -m experiments.my_experiment
```

---

## Glossary

| Term | Definition | See |
|------|-----------|-----|
| **STDP** | Spike-Timing-Dependent Plasticity — learning rule based on spike timing | ARCHITECTURE.md Tier 1 |
| **Eligibility Trace** | Memory of recent presynaptic activity for delayed learning | API_REFERENCE.md core.synapse.Synapse |
| **Homeostasis** | Self-regulation of firing rates via synaptic scaling | ARCHITECTURE.md Tier 1 |
| **Novelty** | Score measuring how different a pattern is from past experience | ARCHITECTURE.md Tier 3 |
| **Plasticity Gate** | Mechanism that enables/disables learning based on context | ARCHITECTURE.md Tier 3 |
| **Intrinsic Motivation** | Internal reward signal based on curiosity and learning | ARCHITECTURE.md Tier 4 |
| **System 1/2** | Fast automatic reactions (1) vs slow deliberate thinking (2) | ARCHITECTURE.md Tier 3-4 |
| **Region** | Spatially organized group of neurons with shared properties | API_REFERENCE.md core.region.Region |
| **Self-Model** | Internal model predicting outcomes of actions | ARCHITECTURE.md Tier 5 |

---

## Troubleshooting

**Problem:** Documentation seems outdated  
**Solution:** Check version badge in README.md (should say Phase 5 Complete)

**Problem:** Can't find information about a feature  
**Solution:** Use Ctrl+F to search all docs, or check archive/ for phase reports

**Problem:** Example code doesn't match current API  
**Solution:** Check API_REFERENCE.md for current signatures, or look at tests/

**Problem:** Want to know history of a feature  
**Solution:** Check docs/archive/PHASE*_COMPLETION_REPORT.md for that phase

---

## Contact & Support

- **Questions about docs:** Open an issue on GitHub
- **Bug in code examples:** Submit PR with correction
- **Feature request:** Describe use case and motivation
- **General help:** Check existing issues and discussions

---

**Last Updated:** January 17, 2026 (Phase 5 Complete)

*Start with [README.md](../README.md) or [GETTING_STARTED.md](../GETTING_STARTED.md)*
