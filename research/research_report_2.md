The Architectural Foundations of Autonomous Intelligence: Intrinsic Motivation, Global Integration, and Causal Foresight
The evolution of artificial intelligence from narrow, task-specific optimization toward general-purpose agency requires a fundamental shift in how systems acquire, organize, and utilize information. Modern computational frameworks are increasingly drawing upon cognitive science and developmental psychology to bridge the gap between reactive processing and proactive reasoning. This transition is characterized by four converging pillars: the mechanisms of intrinsic motivation that drive autonomous exploration, the global workspace architectures that facilitate multimodal integration, the causal discovery principles that extract structural knowledge from data, and the world models that enable high-fidelity planning within imagined latent spaces. This report provides an exhaustive analysis of these domains, synthesizing their mathematical foundations, architectural components, and the emergent behaviors they facilitate in complex environments.
The Genesis of Autonomous Agency through Intrinsic Motivation
The central challenge in building autonomous agents is the scarcity of extrinsic rewards. In the natural world, biological entities do not receive a constant stream of supervision; instead, they are driven by internal imperatives to learn, explore, and master their environments. Intrinsic motivation systems are the computational analogues of this drive, transforming the agent's internal uncertainty, learning progress, or control potential into a reward signal that guides behavior in the absence of external goals.[1, 2]
Intelligent Adaptive Curiosity and the Learning Progress Hypothesis
Early conceptualizations of curiosity often focused on novelty or surprise—rewarding an agent for encountering states that it had not seen before. However, as noted by Oudeyer and Kaplan, simple novelty-seeking is susceptible to the "noisy TV" problem, where an agent becomes trapped by stochastic elements in the environment that are unpredictable but fundamentally unlearnable.[1, 2] To address this, the Intelligent Adaptive Curiosity (IAC) framework posits that the primary drive should be the maximization of "learning progress" rather than the maximization of error or novelty.[1, 3]
The IAC mechanism operates by partitioning the agent's sensorimotor space into distinct regions. Within each region, the system tracks the evolution of its prediction error. The intrinsic reward is defined as the derivative of the error curve: the rate at which the error is decreasing.[2] If an agent enters a region where the error is high but constant—representing a random or unlearnable process—the learning progress is zero, and the agent loses interest. Conversely, if the error is decreasing, the agent is actively "learning" and receives a high internal reward. This push toward regions of intermediate complexity ensures that the agent focuses on situations that are neither too predictable (boring) nor too unpredictable (unlearnable), facilitating a self-organizing developmental sequence.[1, 4]
The Intrinsic Curiosity Module and Feature Space Regularization
In environments characterized by high-dimensional sensory inputs like raw pixels, predicting the next state is computationally prohibitive and prone to failure due to irrelevant details. The Intrinsic Curiosity Module (ICM) addresses this by computing prediction error in a learned, compact feature space.[5, 6] The architecture of ICM is divided into two primary sub-systems: an inverse dynamics model and a forward dynamics model.[5, 7]
The inverse dynamics model is trained to predict the action a 
t
​
  taken between two consecutive states s 
t
​
  and s 
t+1
​
  using their encoded feature representations ϕ(s 
t
​
 ) and ϕ(s 
t+1
​
 ). Because the objective is to predict the agent's own actions, the resulting feature space ϕ is incentivized to represent only those factors in the environment that are affected by the agent's actions or that affect the agent.[5, 8] Factors such as background noise or decorative elements that do not influence the agent's transition are ignored. The forward dynamics model then attempts to predict the next feature representation  
ϕ
^
​
 (s 
t+1
​
 ) given ϕ(s 
t
​
 ) and a 
t
​
 . The intrinsic reward r 
t
i
​
  is the squared error of this prediction:
r 
t
i
​
 =η 
2
1
​
 ∥ 
ϕ
^
​
 (s 
t+1
​
 )−ϕ(s 
t+1
​
 )∥ 
2
2
​
 
This formulation allows ICM to scale to complex environments like Super Mario Bros and VizDoom, where it successfully guides exploration toward task-relevant novelties while remaining robust to environmental distractors.[6, 8]
Random Network Distillation and Scale-Invariant Exploration
Random Network Distillation (RND) provides a surprisingly simple yet powerful alternative to the explicit modeling found in ICM. RND frames exploration as a problem of feature distillation.[9, 10] The system utilizes two neural networks: a fixed, randomly initialized "target" network f and a "predictor" network  
f
^
​
  that is trained to minimize the prediction error on the target network's output given the current observation.[9, 11]
The predictor network is trained on the states the agent actually visits. Consequently, for frequently visited states, the predictor learns to mimic the target network, and the prediction error becomes small. For novel states, the predictor has not yet learned the mapping, and the error remains high, serving as an exploration bonus.[9] RND is particularly effective for several reasons:
• Deterministic Targets: Unlike forward dynamics models that may struggle with environmental stochasticity, the target network in RND is deterministic. This means the predictor can eventually learn the mapping even for noisy inputs, effectively solving the noisy TV problem.[9, 12]
• Scale and Flexibility: RND requires only a single forward pass of the target network and can be combined with any standard reinforcement learning algorithm like PPO.[11, 12]
• Atari Benchmarks: In Montezuma's Revenge—a game notorious for its sparse rewards and complex obstacles—RND demonstrated the ability to regularly find all 24 rooms on the first level and exceed average human performance without any extrinsic guidance.[10, 13, 14]
Empowerment: The Information-Theoretic Utility of Control
Empowerment offers a different perspective on intrinsic motivation, rooted in information theory rather than prediction error. It quantifies the amount of control an agent has over its future sensory inputs through its actions.[15, 16] Formally, empowerment is defined as the channel capacity of the agent's actuation channel—the maximum mutual information between a sequence of n actions a and the resulting future state s 
′
 .[15, 17]
E(s)= 
p(a)
max
​
 I(a;s 
′
 ∣s)= 
p(a)
max
​
 [H(s 
′
 )−H(s 
′
 ∣a)]
Computationally, empowerment requires estimating the probability distribution of future states conditioned on action sequences and then finding the action distribution that maximizes the information transfer.[16, 17] This measure identifies "salient" states in an environment using only the dynamics, without external rewards.[18] A highly empowered state is one where the agent's actions lead to a wide variety of distinct, predictable outcomes.[19] For example, in a navigation task, a state in the middle of a room is more empowered than a state in a dead-end corridor. Maximizing empowerment naturally leads to behaviors such as stabilization and self-preservation, as these states maintain the agent's ability to influence its future.[15, 17]
Motivation System
Primary Reward Source
Underlying Metric
Environmental Handling
IAC (Oudeyer)
Learning Progress
∂t
∂Error
​
 
Targets intermediate complexity regions.
ICM (Pathak)
Prediction Error
$
RND (Burda)
Distillation Error
$
Empowerment
Control Potential
maxI(A;S 
′
 )
Agnostic to goals; maximizes future options.
Global Workspace Theory: Architecture of Integrated Consciousness
While intrinsic motivation explains how information is sought, Global Workspace Theory (GWT) provides a framework for how that information is integrated and distributed within a cognitive system. First proposed by Bernard Baars and later extended by Stanislas Dehaene and others, GWT suggests that consciousness arises from the coordination of specialized, parallel modules through a central, high-capacity bottleneck.[20, 21]
Key Components of the Global Workspace
The architecture of GWT is often described using a theater metaphor.[20] The brain consists of many specialized modules—unconscious processors for vision, hearing, motor control, and language—that operate autonomously and in parallel.[20, 22]
1. The Global Workspace (The Stage): A central hub of limited capacity that temporarily holds information for integration and broadcast.[20, 22]
2. Specialized Modules (The Audience): Autonomous processes that compete for access to the workspace and receive its broadcasts.[20]
3. Attention (The Spotlight): A selection mechanism that determines which module's output enters the workspace and becomes "conscious".[20]
4. Contextual Systems (Behind the Scenes): Processes like the dorsal stream or underlying goal structures that shape what enters the workspace without being part of the conscious broadcast themselves.[20]
Competitive Access and the Neuronal Avalanche
Access to the global workspace is a competitive process. Specialized modules and "coalitions" of modules strive to disseminate their messages to the rest of the system.[20] This competition is often modeled as a "winner-take-all" dynamic where the most salient or task-relevant information gains entry.[20, 23]
Stanislas Dehaene introduced the concept of the "neuronal avalanche" to describe this process. When sensory information is sufficiently strong or attended to, it triggers a non-linear threshold crossing.[20] This leads to a massive recruitment of long-range cortical neurons—primarily in the prefrontal, parietal, and cingulate cortices—which integrate information across space and time.[20] This "ignition" represents the moment information enters the global workspace and becomes accessible to the entire cognitive system.
Computational Broadcast and amodal Translation
In modern computational implementations, "broadcast" refers to the process of making information from one specialized module available to all others.[23, 24] In the Deep Learning roadmap proposed by VanRullen and Kanai, this is achieved through unsupervised neural translation.[23, 24]
The Global Latent Workspace (GLW) acts as an amodal hub.[24] When a visual module wins access, its latent representation is translated into the amodal space of the GLW. The GLW then broadcasts this representation by translating it into the latent spaces of other modules, such as language (enabling the agent to describe what it sees) or motor control (enabling the agent to act on it).[23] This amodal translation is often optimized using cycle-consistency to ensure that information is preserved throughout the broadcast.[23] Computationally, this allows for the "penumbra of consciousness"—representations that are automatically formed during broadcast but not necessarily acted upon unless a module is actively connected to the workspace.[23]
Functional Benefits of Global Workspace Architectures
Implementing a global workspace in artificial agents offers significant behavioral advantages, particularly in multimodal and partially observable environments.[22] Research on embodied agents has shown:
• Robustness to Bottlenecks: GWT-inspired agents perform better than standard recurrent architectures when working memory is limited, as the workspace forces efficient prioritization.[22]
• Temporal Integration: The global broadcast allows agents to process information over time more effectively than direct recurrent feedback, facilitating long-term planning and coherence.[22]
• Multimodal Fusion: The workspace serves as a natural location for resolving conflicts between sensory inputs, such as audio and video tracks, by forcing a single, coherent interpretation for the entire system.[20, 23]
Causal Discovery: Extracting the Mechanisms of Reality
For an agent to act with foresight, it must understand the causal mechanisms of its environment rather than just the statistical correlations. Causal discovery is the field dedicated to learning these structural relationships from data, moving beyond the "ladder of association" to the levels of intervention and counterfactuals.[25, 26, 27]
Statistical Dependence vs. Causal Directionality
Computational difference between correlation and causation lies in symmetry. Correlation is symmetric; if X is correlated with Y, then Y is correlated with X.[28, 29] Causation, however, is directional. If X causes Y (X→Y), a change in X will produce a change in Y, but a change in Y will not necessarily produce a change in X.[28, 29]
This distinction is formalized through the do-calculus.[27, 30] While P(Y∣X) represents the observational distribution (what we see), P(Y∣do(X)) represents the interventional distribution (what happens if we force X to take a value).[27, 31] Graphically, do(X) is implemented as "graph surgery," where all incoming arrows to X are removed, breaking its dependence on its own causes and allowing for the isolated estimation of its effect on Y.[32, 33, 34]
The PC Algorithm: Constraint-Based Structural Search
The PC algorithm, named after Peter Spirtes and Clark Glymour, is a classic constraint-based algorithm for discovering causal structures from observational data.[35, 36] It relies on the Causal Markov and Faithfulness assumptions to map conditional independencies to graph properties.[25, 36, 37]
The algorithm functions in three primary stages:
1. Skeleton Discovery: The algorithm starts with a complete undirected graph. It iteratively tests pairs of variables (X,Y) for conditional independence given subsets of their neighbors. If X⊥Y∣S for some set S, the edge between X and Y is removed, and S is stored as a separation set (Sepset).[35, 36]
2. V-Structure Identification: The algorithm searches for "uncoupled triples" X−Z−Y where X and Y are not adjacent. If the middle variable Z is not in the Sepset of (X,Y), the triple is oriented as a collider: X→Z←Y.[35, 38, 39]
3. Edge Orientation (Meek's Rules): The algorithm applies a set of sound logical rules to orient as many remaining undirected edges as possible without creating new V-structures or directed cycles.[36, 38, 40]
Meek's Rules for Maximal Orientation
Rule
Description
Causal Logic
Rule 1
If X→Y and Y−Z (where X,Z are not adjacent), orient Y→Z.
Prevents the creation of a new V-structure X→Y←Z.
Rule 2
If X→Y→Z and X−Z, orient X→Z.
Prevents the creation of a directed cycle X→Y→Z→X.
Rule 3
If X−Z→Y and X−W→Y (where Z,W are not adjacent), orient X→Y.
Prevents a new V-structure in more complex configurations.
Rule 4
Based on discriminating paths to resolve further ambiguities.
Leverages distant ancestral relationships to orient local edges.
The Role of Intervention in Identifying Direction
Observational data often leaves edges undirected because several graphs can be "Markov Equivalent"—they entail the same set of conditional independencies.[37, 40] Interventions are the definitive tool for resolving these ambiguities. For example, in the case of X−Y, an intervention on X will only change the distribution of Y if X→Y.[34, 36] If the distribution of Y remains invariant under the manipulation of X, then X cannot be a cause of Y. This principle of independent mechanisms—that a cause-effect mechanism is invariant to manipulations of other variables—allows agents to distinguish between directed causation and spurious correlation caused by unobserved confounders.[34, 36]
World Models: Planning in the Latent Imagination
The synthesis of discovery and structural knowledge manifests in World Models—predictive internal representations that allow agents to simulate potential futures and learn behaviors without real-world risk.[41, 42] World models transition the agent from purely model-free reinforcement learning to model-based foresight.
Latent State Prediction and Recurrent State-Space Models (RSSM)
Modern world models do not attempt to predict every pixel of the next frame. Instead, they operate in a compact latent space.[41, 43] The Recurrent State-Space Model (RSSM), used in agents like Dreamer, decomposes the latent state into deterministic and stochastic components.[44, 45]
• Deterministic State (h 
t
​
 ): A recurrent hidden state (often a GRU) that summarizes the history of observations and actions.[45, 46]
• Stochastic State (z 
t
​
 ): A latent variable sampled from a distribution (e.g., Gaussian or categorical) that represents the current observation's contribution.[47, 48]
The transition function predicts the next state s 
t+1
​
  based on the current state and action:
h 
t
​
 =f(h 
t−1
​
 ,z 
t−1
​
 ,a 
t−1
​
 )
z 
t
​
 ∼q(z 
t
​
 ∣h 
t
​
 ,o 
t
​
 ) (Posterior)
z
^
  
t
​
 ∼p( 
z
^
  
t
​
 ∣h 
t
​
 ) (Prior)
By minimizing the divergence between the prior and posterior, the world model learns to predict the next stochastic state using only its internal history, enabling the agent to "roll out" future trajectories entirely in its latent "imagination".[43, 44, 47]
Learning in Imagination: The Dreamer Algorithm
"Learning in imagination" is the process of deriving an optimal policy purely from predicted trajectories within the world model.[43, 44] In the Dreamer architecture, an actor and a critic are trained on these imagined trajectories. The key innovation is the use of analytic gradients.[43, 44, 49]
Because the world model (RSSM) is a differentiable neural network, the agent can calculate the gradient of the predicted future value with respect to the actor's actions by backpropagating through the imagined steps.[43] This allows for much faster and more stable behavior learning than derivative-free methods. The agent starts at a real state from its history and "dreams" forward 15 to 40 steps, updating its policy to maximize the expected sum of imagined rewards and the value of the final imagined state.[43, 44]
MuZero: Value Equivalence and Rule-Free Planning
MuZero represents the pinnacle of planning without explicit environment knowledge. Unlike models that try to reconstruct observations, MuZero is built on the "value equivalence" principle.[50, 51] It learns a model that predicts only the quantities directly relevant to decision-making: the reward, the policy (which action is best), and the value (how good is the position).[52, 53, 54]
MuZero consists of three networks:
1. Representation (h): Maps observations to a latent state s 
0
​
 .[53]
2. Dynamics (g): Predicts the next latent state s 
k+1
​
  and reward r 
k+1
​
  given s 
k
​
  and a 
k+1
​
 .[52, 53]
3. Prediction (f): Estimates the policy p 
k
​
  and value v 
k
​
  from a latent state s 
k
​
 .[52, 53]
MuZero plans by unrolling these networks within a Monte Carlo Tree Search (MCTS).[53, 54] It does not know the rules of chess or Go; instead, it has learned a "value equivalent" internal dynamics model where a sequence of actions in the latent space yields the same cumulative reward as the same actions in the real game.[50, 51, 55] This allows MuZero to achieve superhuman performance across diverse tasks—from Atari games to Go—using a single, unified configuration.[50, 53, 56]
Model Architecture
Planning Mechanism
Latent Objective
Key Advantage
World Models (Ha)
Controller on Latent
Observation Reconstruction
Unsupervised Feature Learning
Dreamer (Hafner)
Analytic Value Gradients
RSSM Consistency
Sample-efficient Imagination
MuZero
Monte Carlo Tree Search
Value Equivalence
Rules-free Precision Planning
Conclusion: The Integrated Frontier of Cognitive AI
The synthesis of these four domains establishes a robust framework for artificial general intelligence. Intrinsic motivation provides the initial drive, pushing agents to explore the "sweet spot" of learning progress and maintain control over their environment.[1, 16] Global Workspace Theory offers a template for integrating these exploratory experiences into a coherent amodal state, allowing for the flexible distribution of information across specialized modules.[20, 24] Causal discovery then structures this integrated data into mechanistic models, enabling the agent to reason about interventions and distinguish true cause-and-effect from spurious associations.[25, 27, 35] Finally, World Models provide the simulator where these causal mechanisms are tested and utilized for long-term planning.[43, 53]
As these technologies mature, the next generation of agents will not merely optimize for a scalar reward but will autonomously develop a structural understanding of their world, coordinate multimodal processing through cognitive bottlenecks, and act with the foresight provided by high-fidelity latent imagination. This convergence marks the transition from "learning by doing" to "learning by thinking," bringing artificial systems closer to the open-ended developmental capabilities of biological minds.
--------------------------------------------------------------------------------
1. (PDF) Intrinsic Motivation Systems for Autonomous Mental Development (2007) | Pierre-Yves Oudeyer | 1364 Citations - SciSpace, https://scispace.com/papers/intrinsic-motivation-systems-for-autonomous-mental-a9mds7ef42
2. Intrinsic Motivation Systems for Autonomous Mental Development, http://www.pyoudeyer.com/ims.pdf
3. Intrinsic Motivation Systems for Autonomous Mental Development - Cogprints - University of Southampton Web Archive, https://web-archive.southampton.ac.uk/cogprints.org/5473/index.html
4. Intrinsic Motivation Systems for Autonomous Mental Development - Computer Science, https://www.cs.swarthmore.edu/~meeden/DevelopmentalRobotics/iac07.pdf
5. Curiosity-driven Exploration by Self-supervised Prediction - Deepak Pathak, https://pathak22.github.io/noreward-rl/resources/icml17.pdf
6. Curiosity-driven Exploration by Self-supervised Prediction - Proceedings of Machine Learning Research, https://proceedings.mlr.press/v70/pathak17a/pathak17a.pdf
7. Curiosity-Driven Exploration by Self-Supervised Prediction - CVF Open Access, https://openaccess.thecvf.com/content_cvpr_2017_workshops/w5/papers/Pathak_Curiosity-Driven_Exploration_by_CVPR_2017_paper.pdf
8. Curiosity-driven Exploration by Self-supervised Prediction - arXiv, https://arxiv.org/pdf/1705.05363
9. Exploration by random network distillation, https://www.pure.ed.ac.uk/ws/files/181350841/Exploration_by_Random_BURDA_DoA211218_AFV.pdf
10. [1810.12894] Exploration by Random Network Distillation - arXiv, https://arxiv.org/abs/1810.12894
11. exploration by random network distillation - arXiv, https://arxiv.org/pdf/1810.12894
12. Transfer Learning with Random Network Distillation Theory & Reinforcement Learning - CS229, https://cs229.stanford.edu/proj2019spr/report/96.pdf
13. Montezuma's Revenge Solved by Go-Explore, a New Algorithm for Hard-Exploration Problems (Sets Records on Pitfall, Too) | Uber Blog, https://www.uber.com/blog/go-explore/
14. Slow Papers: Exploration by Random Network Distillation (Burda et al., 2018) | endtoendAI, https://v1.endtoend.ai/slowpapers/rnd/
15. Empowerment: A Universal Agent-Centric Measure of Control, https://uhra.herts.ac.uk/id/eprint/282/1/901241.pdf
16. Keep Your Options Open: An Information-Based Driving Principle for Sensorimotor Systems, https://pmc.ncbi.nlm.nih.gov/articles/PMC2607028/
17. Efficient Empowerment Estimation for Unsupervised Stabilization - EECS at Berkeley, https://www2.eecs.berkeley.edu/Pubs/TechRpts/2021/EECS-2021-109.pdf
18. Empowerment for Continuous Agent-Environment Systems - UT Austin Computer Science, https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/AB11-jung.pdf
19. Estimating the Empowerment of Language Model Agents - arXiv, https://arxiv.org/html/2509.22504v2
20. Global workspace theory - Wikipedia, https://en.wikipedia.org/wiki/Global_workspace_theory
21. Global workspace theory of consciousness: toward a cognitive neuroscience of human experience - PubMed, https://pubmed.ncbi.nlm.nih.gov/16186014/
22. Design and evaluation of a global workspace agent embodied in a realistic multimodal environment - Frontiers, https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2024.1352685/full
23. Deep Learning and the Global Workspace Theory, https://arxiv.org/abs/2012.10390
24. Deep learning and the Global Workspace Theory - PubMed, https://pubmed.ncbi.nlm.nih.gov/34001376/
25. Comprehensive Review and Empirical Evaluation of Causal Discovery Algorithms for Numerical Data - arXiv, https://arxiv.org/html/2407.13054v1
26. (PDF) From Correlation to Causation: Understanding Climate Change through Causal Analysis and LLM Interpretations - ResearchGate, https://www.researchgate.net/publication/387349999_From_Correlation_to_Causation_Understanding_Climate_Change_through_Causal_Analysis_and_LLM_Interpretations
27. What is Pearl's Causal Calculus? | Activeloop Glossary, https://www.activeloop.ai/resources/glossary/pearls-causal-calculus/
28. Correlation vs Causation: Understanding the Difference - R-bloggers, https://www.r-bloggers.com/2025/06/correlation-vs-causation-understanding-the-difference/
29. Beyond Correlation: The Power of Causal Inference - Craft AI, https://www.craft.ai/post/beyond-correlation-the-power-of-causal-inference-in-modern-data-science
30. Introduction to Causal Calculus - UBC Computer Science, https://www.cs.ubc.ca/labs/lci/mlrg/slides/doCalc.pdf
31. An Introduction to Causal Inference - PMC - PubMed Central, https://pmc.ncbi.nlm.nih.gov/articles/PMC2836213/
32. Do-Calculus: Causal Inference Rules - Emergent Mind, https://www.emergentmind.com/topics/do-calculus
33. Quantum Causality: Resolving Simpson's Paradox with 𝒟⁢𝒪-Calculus - arXiv, https://arxiv.org/html/2509.00744v1
34. Causation, Prediction, and Search - CMU School of Computer Science, https://www.cs.cmu.edu/afs/cs.cmu.edu/project/learn-43/lib/photoz/.g/web/.g/scottd/fullbook.pdf
        ▪ Carnegie Mellon University, https://www.cmu.edu/dietrich/philosophy/docs/scheines/spirtes5-2.doc
35. PC Algorithm, https://tyler-causality.streamlit.app/PC_Algorithm
36. Causation, Prediction, and Search, 2nd Edition | Request PDF - ResearchGate, https://www.researchgate.net/publication/227458517_Causation_Prediction_and_Search_2nd_Edition
37. Lab 4: Peter-Clark (PC) Algorithm - Yixin Zhu | PKU, https://yzhu.io/courses/core/labs/lab4/
38. On the completeness of orientation rules for causal discovery in the presence of latent confounders and selection bias - ResearchGate, https://www.researchgate.net/publication/30770452_On_the_completeness_of_orientation_rules_for_causal_discovery_in_the_presence_of_latent_confounders_and_selection_bias
39. Practical Algorithms for Orientations of Partially Directed Graphical Models - Proceedings of Machine Learning Research, https://proceedings.mlr.press/v213/luttermann23a/luttermann23a.pdf
40. [PDF] World Models - Semantic Scholar, https://www.semanticscholar.org/paper/World-Models-Ha-Schmidhuber/ff332c21562c87cab5891d495b7d0956f2d9228b
41. World Modeling in AI: What It Is and How It Works End-to-End - C# Corner, https://www.c-sharpcorner.com/article/world-modeling-in-ai-what-it-is-and-how-it-works-end-to-end/
42. Dream to Control: Learning Behaviors by Latent Imagination, https://arxiv.org/pdf/1912.01603
43. Dreamer: Learning Long-Horizon Behaviors through Latent Imagination - Medium, https://medium.com/@kdk199604/dreamer-learning-long-horizon-behaviors-through-latent-imagination-fa60327c2843
44. EN.520.637 Project Report: Training Bipedal Walker In Latent Space, http://lovinglavigne.com/dreamer.pdf
45. Mastering diverse control tasks through world models - PMC - PubMed Central, https://pmc.ncbi.nlm.nih.gov/articles/PMC12003158/
46. DreamerV3: Mastering Diverse Domains through World Models - The VITALab website, https://vitalab.github.io/article/2023/01/19/DreamerV3.html
47. LEARNING HIERARCHICAL WORLD MODELS WITH ADAPTIVE TEMPORAL ABSTRACTIONS FROM DISCRETE LATENT DYNAMICS - ICLR Proceedings, https://proceedings.iclr.cc/paper_files/paper/2024/file/13b45b44e26c353c64cba9529bf4724f-Paper-Conference.pdf
48. Dream to Control: Learning Behaviors by Latent Imagination - Danijar Hafner, https://danijar.com/project/dreamer/
49. Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model - Department of Computer Science (IDI), https://www.idi.ntnu.no/emner/it3105/materials/rl/muzero-2020.pdf
50. Proper Value Equivalence - OpenReview, https://openreview.net/pdf?id=aXbuWbta0V8
51. Visualizing MuZero Models - OpenReview, https://openreview.net/pdf/5291aeee6cdca5380d0e16d0c999ef187d0a00cf.pdf
52. MuZero: Mastering Go, chess, shogi and Atari without rules - Google DeepMind, https://deepmind.google/blog/muzero-mastering-go-chess-shogi-and-atari-without-rules/
53. How does MuZero learn without knowing the environment? - Milvus, https://milvus.io/ai-quick-reference/how-does-muzero-learn-without-knowing-the-environment
54. What model does MuZero learn? - arXiv, https://arxiv.org/html/2306.00840v3
55. Demystifying MuZero Planning: Interpreting the Learned Model - arXiv, https://arxiv.org/html/2411.04580v2