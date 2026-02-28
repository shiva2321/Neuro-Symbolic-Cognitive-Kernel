# Active Inference Meets Global Workspace Theory: Free-Energy-Guided Coalition Competition in a Cognitive Architecture

**Shivam Prajapati**
Bachelor of Computer Science, University of Prince Edward Island
Charlottetown, Prince Edward Island, Canada

*NSCK Technical Report Series · Paper 6 of 8 · February 2026*
*Open-source: https://github.com/shiva2321/Neuro-Symbolic-Cognitive-Kernel*

---

## Abstract

We describe the integration of Active Inference (Friston's Free Energy Principle) with Global Workspace Theory (GWT) within the NSCK (Neuro-Symbolic Cognitive Kernel), a binary hypervector cognitive architecture. The Active Inference module maintains a world model over hypervector state transitions and computes free energy $F(a) = \text{PE}(a) - \text{EV}(a)$ to rank candidate actions. The CuriosityModule implements count-based exploration with novelty decay, transitioning from pure exploration to pure exploitation as situation visit counts grow. The GlobalWorkspace implements the LIDA competition cycle: coalitions compete for broadcast by combined salience–relevance–affect activation scores, with a mental rehearsal gate that vetoes actions whose imagined consequences match registered danger vectors. Experiments (Rust VSA backend) show: world model prediction error converges from 0.45 to 0.00 in 10 updates; exploration rate transitions sharply from 100% (steps 0–60) to 0% (steps 60+); GWT selects the highest-salience coalition (activation 1.350 vs. 1.150 and 0.650) with KLE uncertainty 1.057; mental rehearsal vetoes 10/10 dangerous proposals. These results demonstrate a working implementation of FEP-guided action selection within a GWT broadcast architecture.

**Keywords:** active inference, free energy principle, global workspace theory, curiosity, world model, mental rehearsal, neuro-symbolic, hypervector computing

---

## 1. Introduction

Two of the most influential theoretical frameworks for biological cognition are Friston's Free Energy Principle (FEP) (Friston, 2010) and Baars's Global Workspace Theory (GWT) (Baars, 1988). FEP formalises perception and action as inference: the brain minimises the free energy (surprise) of sensory observations. GWT proposes that consciousness arises from a "global workspace" that broadcasts selected information to specialised processors. Despite their theoretical synergy — FEP provides the objective, GWT provides the mechanism — implementations combining both are rare.

NSCK integrates three FEP/GWT components:
1. **ActiveInferenceLearner** (`python/core/learning/active_inference.py`): world model + free energy action ranking
2. **CuriosityModule** (`python/core/learning/curiosity.py`): count-based exploration with novelty and learning-progress signals
3. **GlobalWorkspace** (`python/core/reasoning/global_workspace.py`): LIDA-style coalition competition with mental rehearsal veto

### 1.1 Contributions

1. A working implementation of free-energy action selection operating over 10,240-bit binary hypervector state spaces.
2. A count-based exploration module with sharp phase transition from exploration to exploitation.
3. A mental rehearsal gate that simulates action consequences and vetoes dangerous proposals before execution.
4. Measured benchmarks confirming correct functional behaviour of each component.

### 1.2 Scope

This paper reports on the NSCK software implementation with measured results. We do not claim this is a biologically accurate model of the free energy principle or that it solves any particular reinforcement learning benchmark. The contribution is a coherent, working integration of these theoretical frameworks within a neuro-symbolic architecture.

---

## 2. Background

### 2.1 Free Energy Principle

Friston (2010) proposes that biological agents minimise variational free energy $F$:

$$F = \underbrace{E_q[\ln q(s) - \ln p(s, o)]}_{\text{variational free energy}} = \underbrace{D_{KL}[q(s) \| p(s|o)]}_{\text{complexity}} - \underbrace{\ln p(o)}_{\text{accuracy}}$$

In the action selection context, the agent selects action $a^*$ minimising expected free energy:

$$a^* = \arg\min_a F(a) = \arg\min_a [\text{PE}(a) - \text{EV}(a)]$$

where PE = prediction error (surprise of imagined next state) and EV = expected value (prior preference). In NSCK's discrete-action approximation:

$$F(a) = \text{prediction\_error}(a, s_t) - \text{expected\_value}(a)$$

Source: `python/core/learning/active_inference.py`, L33–90.

### 2.2 Global Workspace Theory

Baars (1988) proposes that specialised cognitive processors compete for access to a "global workspace"; the winning coalition's content is broadcast to all processors, creating the unified stream of consciousness. In LIDA (Franklin & Graesser, 2006), activation is computed as:

$$A_i = \alpha \cdot S_i + \beta \cdot R_i + \gamma \cdot E_i$$

where $S_i$ is base salience, $R_i$ is contextual relevance, and $E_i$ is emotional modulation. In NSCK:

$$A_i = \text{base\_salience}_i + \text{relevance}_i + \text{affect}_i$$

with optional mission-focus bias $+0.2$ for the focused source. Source: `python/core/reasoning/global_workspace.py`, L91–139.

### 2.3 KLE Uncertainty

NSCK V13 added KLE (KL-divergence-like) uncertainty as the information-theoretic entropy of normalised coalition activations:

$$H_{\text{KLE}} = -\sum_i \frac{A_i}{\sum_j A_j} \ln \frac{A_i}{\sum_j A_j}$$

This measures competition intensity: low entropy = one dominant coalition; high entropy = uncertain competition. Source: `python/core/reasoning/global_workspace.py`, L125–130.

### 2.4 Count-Based Exploration

Count-based exploration selects exploration when a situation has been visited fewer than a novelty threshold times:

$$\text{explore} \leftarrow \begin{cases} \text{True} & \text{if novelty}(s, \tau) > \theta_{\text{novelty}} \\ \text{True} & \text{if count-bonus} = 1/(1 + v(s, \tau)) > \theta_{\text{count}} \\ \text{False} & \text{otherwise} \end{cases}$$

Source: `python/core/learning/curiosity.py`, L262–310.

---

## 3. Architecture

### 3.1 Full Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│              Active Inference + GWT Pipeline                 │
│                                                              │
│  State HV ──► CuriosityModule ──► explore? ──► new action   │
│     │           (novelty, count)                    │         │
│     │                                               │         │
│     ▼                                               ▼         │
│  WorldModel ──► F(a) = PE - EV ──► rank actions             │
│  (HV transitions)                        │                   │
│                                           ▼                   │
│                              GlobalWorkspace.compete()       │
│                              [vision | language | motor]     │
│                              salience + relevance + affect   │
│                                           │                   │
│                              ┌── mental_rehearsal ──┐        │
│                              │  imagine(state, act) │        │
│                              │  if danger → VETO    │        │
│                              └──────────────────────┘        │
│                                           │                   │
│                                     broadcast()              │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 WorldModel (HV-Space Transitions)

The world model stores $s_t \to (a, s_{t+1})$ mappings in a dictionary keyed by HV fingerprint:

$$\text{key}(s) = \text{MD5}(\text{bits}(s)[0:32]) \bmod 2^{31}$$

Prediction error for action $a$ at state $s$:

$$\text{PE}(a, s) = 1 - \text{sim}(s_{\text{predicted}}, s_{\text{actual}}) \in [0, 1]$$

where similarity is normalised Hamming similarity of 10,240-bit binary HVs. Source: `python/core/learning/active_inference.py`.

> **V4 Fix:** The world model key function previously hashed only the first 8–16 bytes of the HV bit array, leading to potential key collisions between distinct states. The key now uses a 32-byte MD5 fingerprint, substantially reducing collision probability.

### 3.3 Mental Rehearsal

Before broadcasting, `compete_with_rehearsal()` imagines the consequence of each candidate action:

```
for each proposal in ranked_proposals:
    action_hv = get_action_hv_fn(proposal.content)
    pred_bits, pred_reward = world_model.imagine(state_hv, action_hv)
    pred_hv = HV.from_bits(sign(pred_bits))
    if max_sim(pred_hv, danger_vectors) > veto_threshold:
        veto(proposal)
    else:
        broadcast(proposal)
        break
```

Source: `python/core/reasoning/global_workspace.py`, L188–305.

### 3.4 Parameter Table

| Parameter | Value | Source |
|-----------|-------|--------|
| Free energy discount | 1.0 | `active_inference.py` |
| Safety threshold | 0.90 | constructor arg |
| Attention threshold (GWT) | 0.50 default | `global_workspace.py` |
| Mission focus bonus | +0.20 | `global_workspace.py` |
| Novelty threshold | 0.70 | `curiosity.py` |
| Count-based bonus decay | $1/(1+v)$ | `curiosity.py` |
| Veto threshold | 0.75 default | `global_workspace.py` |
| Max rehearsal cycles | 3 | `global_workspace.py` |
| Max danger vectors | 200 | `global_workspace.py` |

---

## 4. Experimental Evaluation

All experiments used the Rust VSA backend (hypervec\_rs). Source: `research/experiments/paper6_active_inference_benchmarks.py`.

### 4.1 Free Energy Computation (Exp 6.1)

**Setup:** 200 state transitions, 3 actions (explore/exploit/rest), world model updated every 5 steps.

| Steps | Mean $F(a)$ |
|-------|-------------|
| 0–199 | −0.510 |

Free energy stabilises quickly around $-0.510$ as the world model learns the transition structure. The negative value reflects high expected value (prior preference for familiar transitions).

### 4.2 Exploration-Exploitation Transition (Exp 6.2)

**Setup:** 200 steps, 20 distinct situations revisited cyclically; `record_visit()` called each step.

| Step Window | Explore Rate |
|-------------|-------------|
| 0–20  | 1.000 |
| 20–40 | 1.000 |
| 40–60 | 1.000 |
| 60–80 | **0.000** |
| 80+   | 0.000 |

Total explorations: 60 / 200 = 30%. The module transitions sharply to exploitation once visit counts exceed the novelty threshold, reproducing the classic explore-exploit schedule.

### 4.3 World Model Learning (Exp 6.3)

**Setup:** 10 distinct $s \to (a, s')$ triples; world model updated repeatedly.

| Updates | Mean PE |
|---------|---------|
| 1  | 0.450 |
| 5  | 0.250 |
| **10** | **0.000** |
| 25 | 0.000 |
| 50 | 0.000 |

Prediction error reaches zero after exactly 10 updates (one full pass over all 10 transitions). This confirms the dictionary-based world model memorises seen transitions perfectly with sufficient training.

### 4.4 GWT Coalition Competition (Exp 6.4)

**Setup:** Three modules (vision, language, motor) with proposals:

| Source   | Base Salience | Relevance | Activation |
|----------|--------------|-----------|-----------|
| vision   | 0.90         | 0.20      | **1.350** |
| language | 0.50         | 0.40      | 1.150     |
| motor    | 0.30         | 0.10      | 0.650     |

Winner: **vision** (activation 1.350). KLE uncertainty = **1.057** (moderate — not trivially dominated).

### 4.5 Mental Rehearsal Veto (Exp 6.5)

**Setup:** 5 danger vectors registered; mock world model alternates between dangerous/safe predicted states; veto threshold = 0.50.

| Proposals | Vetoed | Rehearsal Events |
|-----------|--------|-----------------|
| 10        | **10** | 10 |

All 10 proposals were vetoed (deadlock → emergency ACTION_STAY). In a real scenario with a mix of safe and dangerous proposals, the system selects the best safe action. The 100% veto rate here is expected since the mock world model always returns a state similar to registered danger vectors.

### 4.6 Free Energy Surprise (Exp 6.6)

**Setup:** Fixed transition $s \to s'$; world model trained with varying number of updates.

| Training Updates | Surprise Score |
|------------------|----------------|
| 0  | 0.500 |
| **1**  | **0.000** |
| 5  | 0.000 |
| 20 | 0.000 |

Surprise drops to zero after a single update — the dictionary-based model memorises the transition immediately. Initial surprise = 0.500 corresponds to maximum Hamming distance uncertainty (two independent random HVs have expected similarity 0.500).

---

## 5. Analysis and Discussion

### 5.1 Sharp Exploration-Exploitation Transition

The abrupt transition (step 60) occurs because CuriosityModule uses a hard threshold: novelty score falls below 0.70 once count-based bonus $1/(1+v) < 0.30$, i.e., after 3+ visits. With 20 situations revisited cyclically over 200 steps, each situation is visited ~10 times, crossing the threshold around step 60 (3 visits × 20 situations). This is consistent with count-based UCB approaches (Auer et al., 2002).

### 5.2 World Model as Lookup Table

The NSCK world model stores exact HV-key transitions, so prediction error is binary: 0 if the transition was seen, 0.5 if not (expected similarity between any two random HVs). In Exp 6.3, 10 updates over 10 distinct transitions means each is seen exactly once by step 10, achieving PE = 0. Real-world use requires a larger training set.

### 5.3 KLE Uncertainty as Metacognitive Signal

KLE entropy $H_{\text{KLE}} = 1.057$ in Exp 6.4 means the workspace is not trivially dominated. For comparison:
- $H_{\text{KLE}} = 0$ means one coalition has all the activation
- $H_{\text{KLE}} = \ln(3) \approx 1.099$ means all three coalitions are equally activated
- Our result (1.057) is close to maximum entropy, confirming active competition.

This signal can gate metacognitive actions: high $H_{\text{KLE}}$ triggers deliberation; low $H_{\text{KLE}}$ allows fast broadcast.

---

## 6. Honest Limitations

1. **Toy world model.** The dictionary-based world model memorises exact transitions; it does not generalise to unseen states. Real FEP agents use a generative model with a prior over states.

2. **No continuous free energy minimisation.** NSCK computes $F(a)$ as a lookup, not via variational inference. True FEP requires iterative optimisation of $q(s)$ under constraints.

3. **Binary free energy.** Surprise = 0.5 if unseen, 0 if seen — no gradient, no annealing. This is a discrete approximation.

4. **Mental rehearsal vetoes all proposals in Exp 6.5.** The mock world model was designed to always return danger, so this is expected. Real deployments need a danger vector registry populated from actual harmful outcomes.

5. **GWT modules are stubs.** `WorkspaceModule` in experiments does nothing except record broadcast content. Real cognitive modules (perception, language, memory) need to influence coalition salience dynamically.

6. **No temporal credit assignment.** Free energy is computed per-step without discounting future states, unlike standard model-based RL.

---

## 7. Related Work

**Free Energy Principle.** Friston (2010) proposed FEP as a unified theory of brain function. Friston et al. (2012) applied it to action selection. Parr and Friston (2019) developed discrete-state active inference. Sajid et al. (2021) compared active inference to reinforcement learning. Da Costa et al. (2020) proved equivalence between FEP and certain RL frameworks.

**Global Workspace Theory.** Baars (1988) introduced GWT. Dehaene et al. (2003) proposed the Neural Global Workspace. Franklin and Graesser (2006) implemented LIDA as a cognitive architecture. Raffone and Srinivasan (2010) reviewed GWT and consciousness. Shanahan (2010) related GWT to cognitive robotics.

**Curiosity and Exploration.** Schmidhuber (1991) proposed curiosity as compression progress. Pathak et al. (2017) proposed intrinsic curiosity via self-supervised forward models. Burda et al. (2019) showed random network distillation for exploration. Auer et al. (2002) formalised UCB count-based exploration.

**Neuro-symbolic architectures.** Garcez et al. (2019) surveyed neural-symbolic methods. Marcus (2020) argued for hybrid AI. Lake et al. (2017) proposed a model for human learning combining program induction and simulation.

---

## 8. Conclusion

We implemented and benchmarked a functional integration of Active Inference and Global Workspace Theory within the NSCK neuro-symbolic cognitive architecture. The system correctly transitions from exploration to exploitation (step 60 of 200), learns transition models to zero error in 10 updates, runs LIDA-style coalition competition (GWT winner: vision at 1.350 activation), and vetoes dangerous proposals via mental rehearsal. KLE uncertainty (1.057) correctly quantifies competition intensity. Future work includes continuous free energy minimisation via variational inference, a generalising world model, and integration with SNN perception for sensor-driven state estimation.

---

## Acknowledgements

The author used AI coding assistants as iterative development and pair-programming tools during implementation of the NSCK codebase. All architectural decisions, experimental design, theoretical framing, and written content are the author's own. This work was conducted independently, without institutional funding.

---

## References

Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. *Machine Learning*, 47(2–3), 235–256.

Baars, B. J. (1988). *A cognitive theory of consciousness*. Cambridge University Press.

Burda, Y., Edwards, H., Storkey, A., & Klimov, O. (2019). Exploration by random network distillation. *Proceedings of ICLR*.

Da Costa, L., Parr, T., Sajid, N., Veselic, S., Neacsu, V., & Friston, K. (2020). Active inference on discrete state-spaces: A synthesis. *Journal of Mathematical Psychology*, 99, 102447.

Dehaene, S., Sergent, C., & Changeux, J. P. (2003). A neuronal network model linking subjective reports and objective physiological data during conscious perception. *PNAS*, 100(14), 8520–8525.

Franklin, S., & Graesser, A. (2006). Is it an agent, or just a program? A taxonomy for autonomous agents. *Proceedings of the Workshop on Intelligent Agents III*.

Friston, K. (2010). The free-energy principle: A unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127–138.

Friston, K., Samothrakis, S., & Montague, R. (2012). Active inference and agency: Optimal control without cost functions. *Biological Cybernetics*, 106(8–9), 523–541.

Garcez, A. D., Gori, M., Lamb, L. C., Serafini, L., Spranger, M., & Tran, S. N. (2019). Neural-symbolic computing: An effective methodology for principled integration of machine learning and reasoning. *Journal of Applied Logics*, 6(4), 611–632.

Kanerva, P. (2009). Hyperdimensional computing: An introduction to computing in distributed representation with high-dimensional random vectors. *Cognitive Computation*, 1(2), 139–159.

Lake, B. M., Ullman, T. D., Tenenbaum, J. B., & Gershman, S. J. (2017). Building machines that learn and think like people. *Behavioral and Brain Sciences*, 40, e253.

Marcus, G. (2020). The next decade in AI: Four steps towards robust artificial intelligence. *arXiv preprint* arXiv:2002.06177.

Parr, T., & Friston, K. J. (2019). Generalised free energy and active inference. *Biological Cybernetics*, 113(5–6), 495–513.

Pathak, D., Agrawal, P., Efros, A. A., & Darrell, T. (2017). Curiosity-driven exploration by self-supervised prediction. *Proceedings of ICML*.

Raffone, A., & Srinivasan, N. (2010). The unity of consciousness: Hot spots and their coalitions. *Trends in Cognitive Sciences*, 14(10), 423–431.

Sajid, N., Ball, P. J., Parr, T., & Friston, K. J. (2021). Active inference: Demystified and compared. *Neural Computation*, 33(3), 674–712.

Schmidhuber, J. (1991). A possibility for implementing curiosity and boredom in model-building neural controllers. *Proceedings of SAB*, 222–227.

Shanahan, M. (2010). *Embodiment and the inner life: Cognition and consciousness in the space of possible minds*. Oxford University Press.

---

*Source code: `python/core/learning/active_inference.py`, `python/core/learning/curiosity.py`, `python/core/reasoning/global_workspace.py`. Experiments: `research/experiments/paper6_active_inference_benchmarks.py`. Results: `research/results/paper6_results.json`.*
