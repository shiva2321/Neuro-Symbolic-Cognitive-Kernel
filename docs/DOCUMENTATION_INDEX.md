# Node_network Documentation Index

> **Complete Guide to the Neuro-Symbolic Cognitive Kernel (NSCK)**

Welcome to the comprehensive documentation for the Node_network project! This index will guide you to the right documentation based on your needs.

---

## 📚 Documentation Structure

```
docs/
├── DOCUMENTATION_INDEX.md (You are here)
├── ARCHITECTURE.md (Complete system architecture)
├── COMPREHENSIVE_README.md (Getting started guide)
├── LEARNING_AND_MEMORY.md (Learning mechanisms)
└── [Additional specialized docs]

nsck-demo/
├── README.md (Quick start)
├── NSCK_Technical_Report.md (Mathematical foundations)
├── DEEP_DIVE_ANALYSIS.md (Experiment analysis)
├── ARCHITECTURE_REVIEW.md (Component mapping)
└── research_chat.txt (Design discussions)
```

---

## 🎯 Find What You Need

### For Different User Types

#### **👨‍💻 Developers** - "I want to use/modify the system"

Start here:
1. **[Quick Start Guide](./COMPREHENSIVE_README.md#quick-start)** - Installation and first run
2. **[API Reference](./COMPREHENSIVE_README.md#api-reference)** - Core classes and functions
3. **[Architecture Overview](./ARCHITECTURE.md#3-complete-architecture-overview)** - System components
4. **[Custom Tasks Guide](./COMPREHENSIVE_README.md#custom-tasks)** - Adding new tasks

Then explore:
- **[Code Examples](./COMPREHENSIVE_README.md#try-it-yourself)** - Practical tutorials
- **[Troubleshooting](./COMPREHENSIVE_README.md#troubleshooting)** - Common issues

#### **🔬 Researchers** - "I want to understand the theory"

Start here:
1. **[Mathematical Foundations](./ARCHITECTURE.md#5-mathematical-foundations)** - All equations and proofs
2. **[NSCK Technical Report](../nsck-demo/NSCK_Technical_Report.md)** - Detailed mathematical analysis
3. **[Research Papers](./COMPREHENSIVE_README.md#research-papers--theory)** - Related work

Then explore:
- **[Learning Mechanisms](./LEARNING_AND_MEMORY.md#2-learning-mechanisms)** - How the system learns
- **[Deep Dive Analysis](../nsck-demo/DEEP_DIVE_ANALYSIS.md)** - Experimental validation
- **[Research Discussions](../nsck-demo/research_chat.txt)** - Design rationale

#### **🎓 Students** - "I want to learn about neuro-symbolic AI"

Start here:
1. **[Executive Summary](./ARCHITECTURE.md#1-executive-summary)** - What is NSCK?
2. **[Core Concepts](./COMPREHENSIVE_README.md#core-concepts-explained)** - SNNs, VSA, Hebbian learning
3. **[System Workflows](./ARCHITECTURE.md#8-system-workflows)** - How it works step-by-step

Then explore:
- **[Interactive Demos](./COMPREHENSIVE_README.md#try-it-yourself)** - Hands-on learning
- **[Evolution Story](./ARCHITECTURE.md#9-evolution-ncgn--nsck)** - Historical context

#### **🏢 Decision Makers** - "Should we use this?"

Start here:
1. **[Key Innovations](./COMPREHENSIVE_README.md#what-is-nsck)** - Competitive advantages
2. **[Performance Metrics](./ARCHITECTURE.md#10-performance-metrics--proofs)** - Benchmarks and proofs
3. **[Use Cases](./COMPREHENSIVE_README.md#use-cases)** - Application scenarios

Then explore:
- **[Energy Efficiency Analysis](./LEARNING_AND_MEMORY.md#83-proof-energy-efficiency)** - Cost benefits
- **[Deployment Options](./COMPREHENSIVE_README.md#export-to-neuromorphic-hardware)** - Hardware requirements

---

## 📖 Documentation by Topic

### Architecture & Design

| Document | Content | Length | Audience |
|----------|---------|--------|----------|
| **[ARCHITECTURE.md](./ARCHITECTURE.md)** | Complete system architecture, components, workflows | 50+ pages | All |
| **[ARCHITECTURE_REVIEW.md](../nsck-demo/ARCHITECTURE_REVIEW.md)** | Code-to-concept mapping | Medium | Developers |
| **[System Diagram](./ARCHITECTURE.md#31-system-block-diagram)** | Visual architecture overview | 1 page | All |

**Key Sections:**
- [System Philosophy](./ARCHITECTURE.md#2-system-philosophy--design-goals) - Design principles
- [Component Deep Dive](./ARCHITECTURE.md#4-core-components-deep-dive) - SNN, VSA, simulation
- [Data Flow Pipeline](./ARCHITECTURE.md#32-data-flow-pipeline) - Step-by-step execution

### Learning & Intelligence

| Document | Content | Length | Audience |
|----------|---------|--------|----------|
| **[LEARNING_AND_MEMORY.md](./LEARNING_AND_MEMORY.md)** | How system learns, remembers, transfers knowledge | 900+ lines | All |
| **[Learning Mechanisms](./LEARNING_AND_MEMORY.md#2-learning-mechanisms)** | Hebbian, RL, Imitation | Detailed | Researchers |
| **[Memory Systems](./LEARNING_AND_MEMORY.md#3-memory-systems)** | Working, short-term, long-term memory | Detailed | All |

**Key Sections:**
- [Three Learning Pathways](./LEARNING_AND_MEMORY.md#21-three-learning-pathways) - How learning works
- [Knowledge Representation](./LEARNING_AND_MEMORY.md#4-knowledge-representation) - Dual system (implicit/explicit)
- [Cross-Session Persistence](./LEARNING_AND_MEMORY.md#5-cross-session-persistence) - How knowledge survives restarts
- [Transfer Learning](./LEARNING_AND_MEMORY.md#6-transfer-learning) - Applying knowledge across tasks
- [Handling Unseen Scenarios](./LEARNING_AND_MEMORY.md#7-handling-unseen-scenarios) - Zero-shot reasoning

### Mathematics & Theory

| Document | Content | Length | Audience |
|----------|---------|--------|----------|
| **[NSCK_Technical_Report.md](../nsck-demo/NSCK_Technical_Report.md)** | Mathematical framework and proofs | Medium | Researchers |
| **[Mathematical Foundations](./ARCHITECTURE.md#5-mathematical-foundations)** | All equations explained | Detailed | Researchers |
| **[Mathematical Proofs](./LEARNING_AND_MEMORY.md#8-mathematical-proofs)** | Convergence, capacity, efficiency | Detailed | Researchers |

**Key Topics:**
- **LIF Neuron Dynamics** - How spiking neurons work
- **VSA Algebra** - XOR binding, bundle, similarity
- **Hebbian Learning** - 3-factor plasticity rule
- **Policy Gradient** - REINFORCE algorithm
- **Active Inference** - Free energy minimization

### Practical Guides

| Document | Content | Length | Audience |
|----------|---------|--------|----------|
| **[COMPREHENSIVE_README.md](./COMPREHENSIVE_README.md)** | Complete getting started guide | 400+ lines | Developers |
| **[Quick Start](./COMPREHENSIVE_README.md#quick-start)** | Installation and first run | Short | All |
| **[Demo Tutorials](./COMPREHENSIVE_README.md#try-it-yourself)** | Hands-on exercises | Medium | Beginners |
| **[Advanced Usage](./COMPREHENSIVE_README.md#advanced-usage)** | Custom tasks, deployment | Medium | Advanced |

**Key Topics:**
- Installation steps
- Running demos (Snake, Pong, Letters)
- Creating custom tasks
- Exporting to neuromorphic hardware
- Troubleshooting common issues

### Experiments & Validation

| Document | Content | Length | Audience |
|----------|---------|--------|----------|
| **[DEEP_DIVE_ANALYSIS.md](../nsck-demo/DEEP_DIVE_ANALYSIS.md)** | Experimental validation with proofs | Medium | Researchers |
| **[Performance Metrics](./ARCHITECTURE.md#10-performance-metrics--proofs)** | Benchmarks and comparisons | Medium | All |

**Key Evidence:**
- System 2 veto mechanism (logs)
- Learning vs mimicry (entropy analysis)
- Transfer learning (cross-task performance)
- Energy efficiency (comparative measurements)

---

## 🗺️ Learning Paths

### Path 1: Quick Start (30 minutes)

**Goal:** Get NSCK running and understand basics

1. Read: [What is NSCK?](./COMPREHENSIVE_README.md#what-is-nsck) (5 min)
2. Follow: [Installation Guide](./COMPREHENSIVE_README.md#installation) (10 min)
3. Try: [Snake Demo](./COMPREHENSIVE_README.md#demo-1-snake-game) (10 min)
4. Understand: [How It Works](./COMPREHENSIVE_README.md#how-it-works) (5 min)

### Path 2: Developer Integration (2-3 hours)

**Goal:** Integrate NSCK into your project

1. Complete: Path 1
2. Read: [Architecture Overview](./ARCHITECTURE.md#3-complete-architecture-overview) (30 min)
3. Study: [API Reference](./COMPREHENSIVE_README.md#api-reference) (20 min)
4. Follow: [Custom Task Tutorial](./COMPREHENSIVE_README.md#custom-tasks) (45 min)
5. Experiment: Modify and test (60 min)

### Path 3: Research Deep Dive (1-2 days)

**Goal:** Understand theory and contribute research

1. Complete: Path 1 & 2
2. Read: [NSCK Technical Report](../nsck-demo/NSCK_Technical_Report.md) (2 hours)
3. Study: [Mathematical Foundations](./ARCHITECTURE.md#5-mathematical-foundations) (3 hours)
4. Read: [Learning & Memory](./LEARNING_AND_MEMORY.md) (3 hours)
5. Analyze: [Experimental Proofs](../nsck-demo/DEEP_DIVE_ANALYSIS.md) (2 hours)
6. Explore: [Research Papers](./COMPREHENSIVE_README.md#related-research) (ongoing)

### Path 4: Full Mastery (1-2 weeks)

**Goal:** Become an NSCK expert

1. Complete: All previous paths
2. Read: All documentation thoroughly
3. Implement: Multiple custom tasks
4. Experiment: Different architectures
5. Contribute: Code improvements or documentation
6. Publish: Your own experiments or applications

---

## 🔍 Quick Reference

### Core Concepts

- **SNN (Spiking Neural Network)**: Event-driven perception layer using LIF neurons
- **VSA (Vector Symbolic Architecture)**: 10,240-bit hypervectors for logical reasoning
- **System 1**: Fast, reactive, pattern-based processing (SNN)
- **System 2**: Slow, deliberative, logical reasoning (VSA)
- **Hebbian Learning**: "Neurons that fire together, wire together" - local plasticity rule
- **Late Fusion**: Task context injection to prevent catastrophic forgetting
- **Active Inference**: Minimizing free energy (surprise) to drive behavior

### Key Equations

**LIF Neuron:**
$$U_{t+1} = \beta U_t + I_t - S_t \theta$$

**VSA Binding (XOR):**
$$C = A \oplus B$$

**VSA Similarity:**
$$\text{sim}(A,B) = 1 - \frac{H(A,B)}{D}$$

**Hebbian Update:**
$$\Delta w_{ij} = \eta \cdot M(t) \cdot x_i(t) \cdot y_j(t)$$

**Policy Gradient:**
$$\nabla_\theta J = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot R\right]$$

### Important Files

**Core Python:**
- `nsck-demo/python/snn_qat.py` - Spiking Neural Network
- `nsck-demo/python/symbol_grounding.py` - VSA reasoning engine
- `nsck-demo/python/simulation.py` - Counterfactual simulation
- `nsck-demo/python/train_snn.py` - Training utilities

**Rust Core:**
- `nsck-demo/rust_vsa/src/lib.rs` - HyperVector implementation

**Configuration:**
- `nsck-demo/python/dashboard.py` - Main interface
- `requirements.txt` - Python dependencies
- `nsck-demo/rust_vsa/Cargo.toml` - Rust dependencies

**Saved States:**
- `snn_task_aware.pth` - Neural network weights
- `codebook.pkl` - VSA concept dictionary

---

## 📝 Documentation Conventions

### Notation

- **Bold**: Important terms, first occurrence
- `Code font`: Code, filenames, commands
- *Italics*: Emphasis, technical terms
- $$Math$$: Equations and formulas

### Difficulty Levels

| Symbol | Meaning |
|--------|---------|
| 🟢 | Beginner-friendly |
| 🟡 | Intermediate (requires some background) |
| 🔴 | Advanced (requires deep technical knowledge) |

### Document Status

| Label | Meaning |
|-------|---------|
| ✅ Complete | Thoroughly documented
| 🚧 In Progress | Being actively updated
| 📝 Planned | On roadmap

---

## 🤝 Contributing to Documentation

Found an error? Want to improve explanations? Contributions welcome!

### How to Contribute

1. **Report Issues**: Open GitHub issue with `[docs]` prefix
2. **Suggest Improvements**: Pull request with clear description
3. **Add Examples**: Code snippets, tutorials, use cases
4. **Fix Typos**: Direct PR for minor corrections

### Style Guide

- Use clear, concise language
- Provide code examples for concepts
- Include mathematical notation where appropriate
- Add diagrams for complex flows
- Cross-reference related sections

---

## 📞 Getting Help

### Before Asking

1. Check this index for relevant documentation
2. Read the appropriate guide for your level
3. Try the troubleshooting section
4. Search GitHub issues

### Where to Ask

- **Installation Issues**: [Troubleshooting Guide](./COMPREHENSIVE_README.md#troubleshooting)
- **Conceptual Questions**: GitHub Discussions
- **Bug Reports**: GitHub Issues
- **Feature Requests**: GitHub Issues with `enhancement` label

---

## 🎓 Additional Resources

### External Learning Resources

**Spiking Neural Networks:**
- [SNNTorch Tutorial](https://snntorch.readthedocs.io/)
- Maass (1997) - "Networks of Spiking Neurons"

**Vector Symbolic Architectures:**
- Kanerva (2009) - "Hyperdimensional Computing"
- Plate (2003) - "Holographic Reduced Representations"

**Active Inference:**
- Friston (2010) - "The Free Energy Principle"
- Buckley et al. (2017) - "Active Inference Framework"

**Dual-Process Theory:**
- Kahneman (2011) - "Thinking, Fast and Slow"

### Video Tutorials

*(Coming soon - contributions welcome!)*

### Community

- **GitHub Repository**: https://github.com/shiva2321/Node_network
- **Issues**: For bugs and feature requests
- **Discussions**: For Q&A and ideas

---

## 📊 Documentation Statistics

| Metric | Count |
|--------|-------|
| **Total Pages** | 100+ |
| **Code Examples** | 50+ |
| **Diagrams** | 10+ |
| **Equations** | 30+ |
| **References** | 20+ |

**Last Updated**: 2026-01-28

---

<div align="center">

**Happy Learning! 🧠✨**

*"Understanding emerges from the interplay of symbols and signals."*

[⬆ Back to Top](#node_network-documentation-index)

</div>
