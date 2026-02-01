Cognitive Architectures for Autonomous Intelligence: A Synthesis of Intrinsic Motivation, Global Workspace Dynamics, Causal Discovery, and Generative World Models
1. Introduction: The Convergence of Cognitive Science and Artificial Intelligence
The pursuit of Artificial General Intelligence (AGI) has historically oscillated between symbolic, rule-based systems and connectionist, learning-based approaches. While the latter, particularly in the form of Deep Reinforcement Learning (RL), has achieved remarkable success in specialized domains—ranging from game playing to robotic manipulation—fundamental limitations remain. Standard model-free agents often struggle in environments characterized by sparse rewards, high dimensionality, and the requirement for long-term planning under uncertainty. Furthermore, the "black box" nature of deep neural networks frequently precludes the extraction of causal structures, limiting interpretability and robustness against distributional shifts.

To bridge these gaps, contemporary research is increasingly drawing upon principles from cognitive neuroscience, developmental psychology, and information theory. This report provides an exhaustive analysis of four critical pillars that are currently converging to define the next generation of cognitive architectures:

Intrinsic Motivation: Mechanisms that endow agents with internal drives for exploration and learning, independent of extrinsic rewards, mimicking biological curiosity and the drive for competence.   

Global Workspace Theory (GWT): A macro-architectural framework describing how specialized, modular neural processes compete for access to a limited-capacity "workspace," enabling information integration and conscious decision-making.   

Causal Discovery: The mathematical and algorithmic pursuit of inferring causal structure from observational data, moving beyond mere correlation to understanding intervention and counterfactuals.   

World Models: The development of agents that learn internal representations of their environment's dynamics, allowing them to simulate futures ("dream") and plan actions within a latent space rather than the high-dimensional physical world.   

By synthesizing these fields, we observe a trajectory toward agents that are not merely reactive stimulus-response machines but are active inferential engines—entities that construct structured internal models, query their environment to reduce uncertainty, deliberate on future courses of action within a unified cognitive workspace, and operate with a causal understanding of their world.

2. Intrinsic Motivation: The Engine of Agency
In classical Reinforcement Learning (RL), an agent's behavior is dictated by a reward function defined by the environment (e.g., points in a video game, distance accumulated by a robot). This dependency on extrinsic feedback poses severe challenges in real-world scenarios where rewards are sparse, delayed, or entirely absent. In contrast, biological entities exhibit a profound drive to explore and master their environments even in the absence of immediate survival-based rewards. This phenomenon, termed Intrinsic Motivation, is essential for autonomous mental development.

2.1 Theoretical Foundations: From Homeostasis to Competence
The psychological roots of intrinsic motivation lie in the distinction between drives that satisfy physiological deficits (homeostasis) and drives that seek to expand the organism's knowledge and capabilities (competence). Oudeyer and Kaplan (2007) formalized this for robotics, proposing that "intrinsic motivation systems" are necessary for open-ended development.   

They categorize these motivations into two primary classes:

Knowledge-Based (Predictive) Motivation: The drive to reduce uncertainty about the world. The agent seeks situations where its internal models are imperfect but improvable.

Competence-Based Motivation: The drive to maximize control over the environment. The agent seeks to produce specific effects or reach self-generated goals.

A critical innovation in this domain is the Intelligent Adaptive Curiosity (IAC) algorithm. Early formulations of curiosity simply rewarded agents for high prediction error. However, this led to the "white noise problem" (or Noisy TV problem), where agents would become fixated on fundamentally unpredictable stimuli (like static on a screen). Oudeyer and Kaplan introduced the concept of Learning Progress as the reward signal. Instead of rewarding raw error, the system rewards the derivative of the error—the reduction in prediction error over time. This pushes the agent toward the "Goldilocks zone" or the Zone of Proximal Development: situations that are neither too trivial (already mastered) nor too chaotic (unlearnable), but exactly complex enough to permit learning.   

2.2 The Intrinsic Curiosity Module (ICM)
Building on these foundations, Pathak et al. (2017) introduced the Intrinsic Curiosity Module (ICM), a practical deep learning architecture designed to scale curiosity to high-dimensional sensory inputs like images.   

2.2.1 The Challenge of High-Dimensional State Spaces
In visual environments, the state space (pixels) is vast. Predicting the next frame pixel-by-pixel is computationally expensive and prone to failure due to environmental stochasticity. If an agent observes a tree blowing in the wind, the movement of the leaves is hard to predict but irrelevant to the agent's actions. A curiosity mechanism based on pixel-prediction error would inaccurately reward the agent for staring at the tree.

2.2.2 The Inverse-Forward Model Architecture
To resolve this, ICM decouples the representation learning from the dynamics learning using a two-headed architecture :   

The Inverse Dynamics Model (Feature Learning): This component learns a feature space ϕ(s) that encodes only those aspects of the environment that the agent can influence. The model takes the current state s 
t
​
  and the next state s 
t+1
​
  (in feature space) and attempts to predict the action a 
t
​
  that caused the transition. The loss function for the inverse model is:

L 
inv
​
 (θ 
I
​
 )=−logp( 
a
^
  
t
​
 ∣ϕ(s 
t
​
 ),ϕ(s 
t+1
​
 );θ 
I
​
 )
Mechanism: By forcing the network to predict the action, the encoder ϕ effectively filters out environmental noise. If a leaf moves due to wind (and not the agent's action), that movement provides no information about a 
t
​
 . Therefore, the encoder learns to ignore it, resulting in a feature space ϕ that captures only "controllable" dynamics.

The Forward Dynamics Model (Prediction): Once the robust feature space ϕ is established, the forward model attempts to predict the next feature state  
ϕ
^
​
 (s 
t+1
​
 ) given the current features ϕ(s 
t
​
 ) and the action a 
t
​
 . The loss function for the forward model is: $$L_{fwd}(\theta_F) = \frac{1}{2} |

| \hat{\phi}(s_{t+1}) - \phi(s_{t+1}) ||^2$$

2.2.3 Computing Prediction Error as Reward
The intrinsic reward r 
i
t
​
  fed to the reinforcement learning agent is the prediction error of the forward model: $$ r_i^t = \eta \cdot |

| \hat{\phi}(s_{t+1}) - \phi(s_{t+1}) ||^2 $$ where η is a scaling factor. Because this error is computed in the learned feature space ϕ, the agent receives intrinsic rewards only for exploring interactions that are novel and consequential to its agency, effectively solving the Noisy TV problem in many contexts.   

2.3 Random Network Distillation (RND): Exploration via Distillation
While ICM relies on learning dynamics (predicting s 
t+1
​
  from s 
t
​
 ), Burda et al. (2018) proposed Random Network Distillation (RND), a method that achieves state-of-the-art exploration without explicitly modeling environmental dynamics. This approach proved particularly effective in Montezuma's Revenge, a game famous for its deadly traps and extremely sparse rewards.   

2.3.1 The RND Mechanism
RND defines novelty through the lens of prediction error, but uses a simpler, distinct architecture involving two neural networks that process the same observation x:

Component	Nature	Function
Target Network (f)	Fixed, Randomly Initialized	Maps observation x to a embedding y=f(x). This network is never trained.
Predictor Network ( 
f
^
​
 )	Trainable	Attempts to predict the output of the target network:  
y
^
​
 = 
f
^
​
 (x;θ).
The intrinsic reward is the Mean Squared Error (MSE) between the predictor and the target: $$ r_i = |

| \hat{f}(x) - f(x) ||^2 $$

2.3.2 Why RND Solves the Noisy TV Problem
The intuition is that neural networks generalize well to familiar data but produce high errors on novel data.

Familiar States: As the agent visits state x frequently, the Predictor network minimizes the error ∣∣ 
f
^
​
 (x)−f(x)∣∣ 
2
  via gradient descent. The intrinsic reward drops to zero.

Novel States: When the agent enters a new room in Montezuma's Revenge, the input x 
new
​
  is out-of-distribution for the Predictor. The Predictor fails to match the random embedding of the Target, generating a large error signal (high reward).

Stochasticity (Noisy TV): Unlike a forward dynamics model, RND does not predict the future; it predicts the current target embedding. If the agent stares at a screen of white noise, the Target network maps each noise frame to a random vector. The Predictor network attempts to map that same noise frame to that same vector. While the noise is random, the mapping f(x) is deterministic. The Predictor struggles to generalize to the infinite variations of white noise, but the expected error remains constant. Crucially, the agent does not see a reduction in error (learning progress) that would typically reinforce the behavior in dynamics-based methods. The reward signal becomes a constant background hum rather than a spike of "learning," preventing the agent from getting stuck.   

In Montezuma's Revenge, RND enabled agents to find over half the rooms in the game solely through intrinsic curiosity, without looking at the score.   

2.4 Empowerment: Information-Theoretic Agency
Distinct from curiosity (which seeks knowledge), Empowerment seeks agency. Proposed by Klyubin et al. (2005), it is an information-theoretic quantity that measures an agent's potential to influence its environment.   

2.4.1 The Computational Definition
Empowerment is defined as the channel capacity of the link between the agent's actuators (actions) and its sensors (perceptions). It answers the question: "How much information can I inject into the environment and read back?"

Mathematically, for a sequence of actions A of length n and a resulting future sensor state S 
t+n
​
 :

E=C(p(s∣a))= 
p(a)
max
​
 I(A;S 
t+n
​
 )
where I(A;S) is the Mutual Information:

I(A;S)= 
a,s
∑
​
 p(s∣a)p(a)log 
p(s)
p(s∣a)
​
 
2.4.2 Computing Empowerment
Computing empowerment involves:

Modeling the Channel: The environment is treated as a probabilistic channel defined by the transition distribution p(S 
t+n
​
 ∣A 
t...t+n−1
​
 ).   

Blahut-Arimoto Algorithm: Since channel capacity is the maximum mutual information over all possible source distributions p(a), one cannot simply measure it. One must find the optimal distribution of actions that maximizes differentiation in future states. The Blahut-Arimoto algorithm is an iterative method used to converge on this maximum.   

Insight: An agent maximizing empowerment will naturally avoid corners, dead-ends, or states where its actions have no effect (like being stuck in a pit). It will gravitate toward "hubs" in the state space—central locations where a single action can lead to a diverse array of outcomes. This serves as a universal, task-agnostic utility function for survival, as keeping options open is generally evolutionarily advantageous.   

3. Global Workspace Theory: Architecting Consciousness
Moving from the motivational drives of the agent to its cognitive architecture, Global Workspace Theory (GWT) provides a framework for understanding how modular, unconscious processes integrate to form a unified, conscious experience.

3.1 The Theater of the Mind: Baars and Dehaene
Originating with Bernard Baars (1988) and expanded by Stanislas Dehaene (2014), GWT posits a "theater architecture" of the mind.   

The Unconscious Audience: The vast majority of neural processing occurs in specialized, parallel modules (e.g., visual cortex, motor planning, language centers). These modules are encapsulated and operate automatically.

The Global Workspace (The Stage): This is a distributed network of long-range neurons (primarily in the prefrontal and parietal cortices) that interconnects the specialized modules. It has limited capacity.

The Spotlight of Attention: A selective mechanism determines which modular content enters the workspace.

Neural Mechanisms: Dehaene proposes that the entry of information into the workspace is characterized by a non-linear phase transition called ignition. When the signal from a local processor is strong enough (and attended to), it triggers a self-sustaining reverberation of activity across the global network. This effectively "broadcasts" the information, making it globally available to all other modules.   

3.2 Computational Implementations of the Workspace
In the realm of Artificial Intelligence, GWT offers a blueprint for architectures that require coordination between distinct modalities (e.g., vision, audio, text) without requiring O(N 
2
 ) connections between them.

3.2.1 VanRullen & Kanai: Deep Learning and GWT
VanRullen and Kanai (2021) formalized GWT using modern Deep Learning constructs, specifically focusing on the translation between latent spaces.   

Key Components:

Specialized Modules: Deep neural networks trained for specific tasks (e.g., a CNN for vision, a Transformer for language). Each has its own latent space z 
i
​
 .

Global Latent Workspace (GLW): A shared, amodal latent space z 
global
​
  that acts as the lingua franca of the system.

Selection Mechanism: An attention-based router. It uses a Query-Key-Value (QKV) mechanism similar to Transformers. The Workspace emits a Query; modules emit Keys based on their current activity (confidence or salience). The best match wins access.   

3.3 The Broadcast Mechanism and Modular Competition
The core computational operation in GWT is the Broadcast. In the VanRullen & Kanai model, this is not merely copying data; it is a neural translation process.   

Competition: Modules compete for access based on the "salience" or "relevance" of their content. This is computationally modeled as a softmax operation over the attention weights of the modules.

Ignition: Once a module (say, Vision) wins, its latent vector z 
vision
​
  is mapped into the global space: z 
global
​
 =Encoder 
V→G
​
 (z 
vision
​
 ).

Broadcast as Translation: The global vector z 
global
​
  is then fed into the decoders/translators of all other modules.

z 
motor
​
 =Decoder 
G→M
​
 (z 
global
​
 )
z 
lang
​
 =Decoder 
G→L
​
 (z 
global
​
 )
This allows a visual input to immediately influence motor planning or speech generation, even if those modules were never explicitly trained on vision data.

Cycle Consistency: To ensure the workspace captures meaningful semantics, the system is trained using a cycle-consistency loss. If a signal is translated from Vision to Global and back to Vision (V→G→V), the reconstruction error should be minimal. This forces the Global Workspace to retain the core semantic information of the input.   

4. Causal Discovery: Beyond Association
For a cognitive architecture to be robust, it must understand not just what happens (correlation) but why it happens (causation). This is the domain of Causal Discovery.

4.1 The Ladder of Causation
Judea Pearl (2000) defines a hierarchy of causal reasoning :   

Association (P(y∣x)): Seeing. "What does seeing x tell me about y?" This is the domain of standard machine learning.

Intervention (P(y∣do(x))): Doing. "If I force x to happen, what will happen to y?" This requires a causal model.

Counterfactuals (P(y 
x
​
 ∣x 
′
 ,y 
′
 )): Imagining. "What would have happened to y if I had done x instead, given that I actually did x 
′
  and observed y 
′
 ?"

Computationally: The difference between correlation and causation is the response to graph mutilation. To compute P(y∣do(x)), one removes all incoming edges to x in the causal graph (severing it from its parents) and sets its value. If y is a cause of x, setting x will not change y (unlike in correlation where observing x might predict y).   

4.2 The PC Algorithm: Recovering Structure
The PC Algorithm (Spirtes, Glymour, Scheines) is the standard constraint-based method for discovering causal graphs (DAGs) from observational data.   

Assumptions:

Causal Markov Condition: A node is independent of its non-descendants given its parents.

Faithfulness: The probabilistic independencies in the data perfectly match the graph structure (no "accidental" cancellations).

The Algorithm Steps:

Skeleton Discovery (Conditional Independence):

Start with a complete undirected graph connecting all variables.

Iteratively test for conditional independence: X⊥Y∣S.

Start with empty set S (marginal independence). If independent, delete the edge X−Y.

Increase cardinality of S (∣S∣=1,∣S∣=2,…). If X and Y become independent given any separating set S, remove the edge.

Result: An undirected skeleton where edges represent direct associations that cannot be explained away by other variables.   

Orientation (V-Structure Identification):

Identify "unshielded triples" X−Z−Y where X and Y are not connected.

The Collider Rule: Check the separating set S 
xy
​
  that removed the edge between X and Y.

If Z∈
/
S 
xy
​
 , then Z must be a collider: X→Z←Y.

Reasoning: In a chain (X→Z→Y) or fork (X←Z→Y), conditioning on Z blocks the path (makes X,Y independent). Only in a collider does conditioning on Z open the path (Berkson's Paradox). If they were independent without conditioning on Z, then Z functions as a collider.   

Propagation (Meek's Rules):

Orient remaining edges to avoid creating new colliders (which would have been found in step 2) or cycles.

Rule 1: If X→Y−Z and X,Z unconnected, orient Y→Z (avoids new collider).

Rule 2: If X→Y→Z and X−Z, orient X→Z (avoids cycle).   

4.3 Causal Representation Learning
Bernhard Schölkopf (2021) argues that the next step for AI is Causal Representation Learning—learning the high-level causal variables from low-level pixels.   

The Independent Causal Mechanisms (ICM) Principle: The causal generative process is composed of autonomous modules (mechanisms) that do not inform or influence each other. P(Cause) and P(Effect∣Cause) are independent.

Implication for AI: If the distribution of the cause changes (e.g., shifting the position of an object), the physical mechanism describing how light reflects off it (P(Image∣Object)) remains invariant. This invariance is the key to Out-of-Distribution (OOD) generalization. While standard deep learning overfits to the joint distribution P(X,Y), causal learning seeks the invariant mechanisms that hold true even when the environment changes.   

5. World Models: Planning in the Latent Imagination
The final component of the cognitive architecture is the ability to simulate the future. World Models allow agents to internalize the environment's dynamics, enabling "learning in imagination."

5.1 The Generative Vision: Ha & Schmidhuber
Ha and Schmidhuber (2018) proposed a tripartite architecture that explicitly separates perception, memory, and control.   

Module	Architecture	Function
Vision (V)	Variational Autoencoder (VAE)	Compresses the high-dimensional observation (frame) into a low-dimensional latent vector z. This forces the agent to learn a compact representation of the world.
Memory (M)	MDN-RNN (Mixture Density Network)	A predictive model that takes current z 
t
​
  and action a 
t
​
  to predict the distribution of the next state P(z 
t+1
​
 ). The MDN allows it to model stochastic environments (multiple possible futures).
Controller (C)	Simple Linear Layer	A small network that maps z 
t
​
  and the RNN hidden state h 
t
​
  to action a 
t
​
 .
Learning in Imagination: The key innovation is that the Controller can be trained entirely inside the "dream" generated by the Memory module. The RNN simulates a sequence of latent states z, and the Controller interacts with this hallucination. Because the simulation is vastly faster than real-time physics, the agent can learn policies using Evolution Strategies (CMA-ES) efficiently before ever acting in the real world.   

5.2 Dreaming with Gradients: The Dreamer Architecture
Hafner et al. (2020) introduced Dreamer, which addressed the limitations of the VAE-RNN approach (specifically the use of derivative-free evolution strategies) by enabling analytic gradient propagation through the dream.   

5.2.1 RSSM: The Recurrent State-Space Model
Dreamer utilizes an RSSM, which splits the state into:

Deterministic State (h 
t
​
 ): Modeled by an RNN (GRU), carrying the history context.

Stochastic State (s 
t
​
 ): A latent variable sampled from a distribution, allowing the model to handle uncertainty and branching futures.

5.2.2 Value Estimation and Backpropagation
Unlike Ha's model, Dreamer learns a Value Network v 
ψ
​
 (s) (predicting long-term reward) and an Actor Network q 
ϕ
​
 (s) (policy) in the latent space. Because the transition dynamics s 
t+1
​
 =f(s 
t
​
 ,a 
t
​
 ) are differentiable, Dreamer can "backpropagate through time" within the imagination.

Mechanism: The agent imagines a trajectory. It calculates the rewards and value estimates along this path. It then computes the gradient of the value with respect to the actions, updating the policy to maximize future value directly. This is akin to "thinking" about the consequences of actions and adjusting behavior based on the predicted outcome, rather than trial-and-error.   

5.3 Planning Without Rules: The MuZero Paradigm
While Dreamer and Ha's World Model rely on reconstructing the visual observation (via VAE/decoder) to ground their states, MuZero (Schrittwieser et al., 2020) demonstrated that reconstruction is unnecessary for planning. MuZero achieves superhuman performance in Go, Chess, and Atari without ever seeing the rules or trying to generate pixels.   

5.3.1 The Value Equivalence Principle
MuZero is built on the insight that the internal state s only needs to be good enough to predict three things relevant to planning:

Policy (p): The best next move.

Value (v): The winning probability or expected return.

Reward (r): The immediate points gained.

It does not need to predict the next board state or video frame. This avoids the computational burden of modeling irrelevant visual details (e.g., the texture of the grass in a soccer game).

5.3.2 The MuZero Architecture
MuZero uses three distinct networks :   

Representation Network (h): s 
0
​
 =h(o 
1
​
 ...o 
t
​
 ). Encodes history into the initial hidden state.

Dynamics Network (g): r 
k
​
 ,s 
k
​
 =g(s 
k−1
​
 ,a 
k
​
 ). Acts as the transition engine for the search.

Prediction Network (f): p 
k
​
 ,v 
k
​
 =f(s 
k
​
 ). Estimates value/policy for the search leaf nodes.

5.3.3 MCTS in Latent Space
MuZero performs Monte Carlo Tree Search (MCTS) using the learned Dynamics Network. The tree nodes are not actual game states but abstract hidden states s. The agent simulates trajectories through this abstract space, accumulating value estimates and refining its policy choice. This effectively allows MuZero to "learn the rules" of the game in its own latent language, tailored specifically for the task of winning.   

6. Synthesis: Toward Integrated Cognitive Architectures
The four topics discussed are not isolated islands; they are components of a nascent integrated cognitive architecture.

6.1 The Intersection of Motivation and Modeling
World Models (Dreamer/MuZero) require vast amounts of data to learn accurate dynamics. Intrinsic Motivation (ICM/RND) provides the exploration policy necessary to gather this data. An agent driven by curiosity will naturally explore the boundaries of its world, providing the World Model with the "edge cases" needed to robustly learn the environment's physics.   

6.2 The Causal Workspace
Global Workspace Theory provides the routing mechanism for these components. The "Ignition" event can be seen as the moment a prediction error (from the World Model) or a high-empowerment state (from Intrinsic Motivation) becomes salient enough to capture the system's resources. Furthermore, Causal Discovery offers a formal way to structure the World Model. Instead of a "black box" RSSM (as in Dreamer), future architectures may utilize Causal World Models—models where the latent variables correspond to independent causal mechanisms. This would allow the agent not just to predict future states (Dreaming) but to reason about interventions and counterfactuals ("What if I hadn't done that?"), a capability essential for robust planning in the real world.   

7. Conclusion
The path from specialized AI to general intelligence appears to lie in the convergence of these disciplines. Intrinsic motivation provides the drive to learn; World Models provide the map of the environment; Causal Discovery ensures the map reflects the structure of reality; and Global Workspace Theory provides the central executive to coordinate these processes. As these mechanisms are increasingly integrated, we move closer to engineering artificial agents that possess a form of functional consciousness—aware of their environment, driven to understand it, and capable of navigating its causal fabric.

Citations: 1   
Intrinsic Motivation Systems for Autonomous Mental Development - Pierre-Yves Oudeyer
Source icon
pyoudeyer.com/ims.pdf
Curiosity-Driven Exploration in Reinforcement Learning: An Adaptive Self-Supervised Learning Approach for Playing Action Games - MDPI
Source icon
mdpi.com/2073-431X/14/10/434
Conscious Processing and the Global Neuronal Workspace Hypothesis - PubMed Central
Source icon
pmc.ncbi.nlm.nih.gov/articles/PMC8770991
Deep learning and the Global Workspace Theory - PubMed
Source icon
pubmed.ncbi.nlm.nih.gov/34001376
Lab 4: Peter-Clark (PC) Algorithm - Yixin Zhu | PKU
Source icon
yzhu.io/courses/core/labs/lab4


pyoudeyer.com
Intrinsic Motivation Systems for Autonomous Mental Development - Pierre-Yves Oudeyer
Opens in a new window

mdpi.com
Curiosity-Driven Exploration in Reinforcement Learning: An Adaptive Self-Supervised Learning Approach for Playing Action Games - MDPI
Opens in a new window

pmc.ncbi.nlm.nih.gov
Conscious Processing and the Global Neuronal Workspace Hypothesis - PubMed Central
Opens in a new window

pubmed.ncbi.nlm.nih.gov
Deep learning and the Global Workspace Theory - PubMed
Opens in a new window

yzhu.io
Lab 4: Peter-Clark (PC) Algorithm - Yixin Zhu | PKU
Opens in a new window

research-collection.ethz.ch
Toward Causal Representation Learning - ETH Zurich Research Collection
Opens in a new window

papers.neurips.cc
Learning to Predict Without Looking Ahead: World Models Without Forward Prediction - NeurIPS
Opens in a new window

milvus.io
How does MuZero learn without knowing the environment? - Milvus
Opens in a new window

web.eecs.umich.edu
Intrinsic Motivation Systems for Autonomous Mental Development
Opens in a new window

web-archive.southampton.ac.uk
Intrinsic Motivation Systems for Autonomous Mental Development - Cogprints - University of Southampton Web Archive
Opens in a new window

emergentmind.com
Intrinsic Curiosity Modules - Emergent Mind
Opens in a new window

proceedings.mlr.press
Curiosity-driven Exploration by Self-supervised Prediction - Proceedings of Machine Learning Research
Opens in a new window

arxiv.org
Curiosity-driven Exploration by Self-supervised Prediction - arXiv
Opens in a new window

pure.ed.ac.uk
Exploration by random network distillation
Opens in a new window

arxiv.org
Exploration by Random Network Distillation
Opens in a new window

pmc.ncbi.nlm.nih.gov
Keep Your Options Open: An Information-Based Driving Principle for Sensorimotor Systems
Opens in a new window

uhra.herts.ac.uk
Empowerment: A Universal Agent-Centric Measure of Control
Opens in a new window

lizier.me
All Else Being Equal Be Empowered – Klyubin, Polani, Nahaniv - Joseph Lizier
Opens in a new window

alignmentforum.org
Empowerment is (almost) All We Need - AI Alignment Forum
Opens in a new window

en.wikipedia.org
Global workspace theory - Wikipedia
Opens in a new window

academic.oup.com
evolutionary origins of the Global Neuronal Workspace in vertebrates - Oxford Academic
Opens in a new window

arxiv.org
Hypothesis on the Functional Advantages of the Selection-Broadcast Cycle Structure: Global Workspace Theory and Dealing with a Real-Time World - arXiv
Opens in a new window

medium.com
Engineering Consciousness: Global Workspace Theory and Higher-Order Theories
Opens in a new window

arxiv.org
Deep Learning and the Global Workspace Theory - arXiv
Opens in a new window

activeloop.ai
What is Pearl's Causal Calculus? | Activeloop Glossary
Opens in a new window

cs.ubc.ca
Introduction to Causal Calculus - UBC Computer Science
Opens in a new window

ckassaad.github.io
Causal discovery
Opens in a new window

tyler-causality.streamlit.app
PC Algorithm
Opens in a new window

arxiv.org
[2102.11107] Towards Causal Representation Learning - arXiv
Opens in a new window

cl.cam.ac.uk
World Models - David Ha, Jürgen Schmidhuber
Opens in a new window

arxiv.org
[1803.10122] World Models - arXiv
Opens in a new window

researchgate.net
Dream to Control: Learning Behaviors by Latent Imagination - ResearchGate
Opens in a new window

openreview.net
DREAM TO CONTROL: LEARNING BEHAVIORS BY LATENT IMAGINATION - OpenReview
Opens in a new window

openreview.net
DREAM TO CONTROL: LEARNING BEHAVIORS BY LATENT IMAGINATION - OpenReview
Opens in a new window

arxiv.org
Mastering Atari, Go, Chess and Shogi by Planning with a Learned ...
Opens in a new window

en.wikipedia.org
MuZero - Wikipedia
Opens in a new window

idi.ntnu.no
Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model - Department of Computer Science (IDI)
Opens in a new window

deepmind.google
MuZero: Mastering Go, chess, shogi and Atari without rules - Google DeepMind
Opens in a new window

smartlabai.medium.com
World models — a reinforcement learning story | by SmartLab AI - Medium
Opens in a new window

worldmodels.github.io
World Models
Opens in a new window

semanticscholar.org
[PDF] Dreaming: Model-based Reinforcement Learning by Latent Imagination without Reconstruction | Semantic Scholar
Opens in a new window

arxiv.org
[1912.01603] Dream to Control: Learning Behaviors by Latent Imagination - arXiv
Opens in a new window

youtube.com
Dream to Control: Learning Behaviors by Latent Imagination - YouTube
Opens in a new window

reddit.com
[N] No rules, no problem: DeepMind's MuZero masters games while learning how to play them : r/MachineLearning - Reddit
Opens in a new window

yi-t.github.io
Toward Understanding State Representation Learning in MuZero: A Case Study in Linear Quadratic Gaussian Control - Yi Tian
Opens in a new window

arxiv.org
Demystifying MuZero Planning: Interpreting the Learned Model - arXiv
Opens in a new window

julian.ac
MuZero Intuition - Julian Schrittwieser
Opens in a new window

pathak22.github.io
Curiosity-driven Exploration by Self-supervised Prediction - Deepak Pathak
Opens in a new window

www2.eecs.berkeley.edu
Efficient Empowerment Estimation for Unsupervised Stabilization - EECS at Berkeley
Opens in a new window

docs.cleanrl.dev
Random Network Distillation (RND) - CleanRL
Opens in a new window

semanticscholar.org
[PDF] Exploration by Random Network Distillation - Semantic Scholar
Opens in a new window

v1.endtoend.ai
Slow Papers: Exploration by Random Network Distillation (Burda et al., 2018) | endtoendAI
Opens in a new window

cs.swarthmore.edu
Intrinsic Motivation Systems for Autonomous Mental Development - Computer Science
Opens in a new window

pmc.ncbi.nlm.nih.gov
What is Intrinsic Motivation? A Typology of Computational Approaches - PMC
Opens in a new window

frontiersin.org
Global Workspace Theory (GWT) and Prefrontal Cortex: Recent Developments - Frontiers
Opens in a new window

antoniocasella.eu
The Global Neuronal Workspace Model of Conscious Access: From Neuronal Architectures to Clinical Applications - Antonio Casella
Opens in a new window

pnas.org
A neuronal model of a global workspace in effortful cognitive tasks - PNAS
Opens in a new window

frontiersin.org
Hypothesis on the functional advantages of the selection-broadcast cycle structure: global workspace theory and dealing with a real-time world - Frontiers
Opens in a new window

researchgate.net
Deep learning and the Global Workspace Theory | Request PDF - ResearchGate
Opens in a new window

semanticscholar.org
[PDF] Deep learning and the Global Workspace Theory | Semantic Scholar
Opens in a new window

researchgate.net
Embodiment and the Inner Life: Cognition and Consciousness in the Space of Possible Minds - ResearchGate
Opens in a new window

books.google.com
Embodiment and the Inner Life: Cognition and Consciousness in the Space of Possible Minds - Google Books
Opens in a new window

digitalcommons.memphis.edu
Global Workspace Theory, Shanahan, and LIDA - University of Memphis Digital Commons
Opens in a new window

doc.ic.ac.uk
Embodiment and the inner life: Cognition and consciousness in the space of possible minds - Department of Computing - Imperial College London
Opens in a new window

selfawarepatterns.com
Global workspace theory: consciousness as brain wide information sharing - SelfAwarePatterns
Opens in a new window

opensource.salesforce.com
PC Algorithm for Tabular Causal Discovery - Open Source Projects Built at Salesforce
Opens in a new window

jmlr.org
Order-Independent Constraint-Based Causal Structure Learning
Opens in a new window

pmc.ncbi.nlm.nih.gov
An Introduction to Causal Inference - PMC - PubMed Central
Opens in a new window

inference.vc
ML beyond Curve Fitting: An Intro to Causal Inference and do-Calculus
Opens in a new window

reddit.com
If correlation doesn't imply causation, then what does? : r/slatestarcodex - Reddit
Opens in a new window

amazon.science
NeurIPS: Why causal-representation learning may be the future of AI - Amazon Science
Opens in a new window

reddit.com
[R] Towards Causal Representation Learning : r/MachineLearning - Reddit
Opens in a new window

arxiv.org
Towards Causal Representation Learning - arXiv
Opens in a new window

pmc.ncbi.nlm.nih.gov
The target trial framework for causal inference from observational data: Why and when is it helpful? - PMC - NIH
Opens in a new window

pmc.ncbi.nlm.nih.gov
Causal Inference Methods for Combining Randomized Trials and Observational Studies: A Review - PMC - PubMed Central
Opens in a new window

cambridge.org
Interventions and Causal Inference | Philosophy of Science | Cambridge Core
Opens in a new window

proceedings.neurips.cc
Causal discovery from observational and interventional data across multiple environments - NeurIPS
Opens in a new window

norc.org
The design versus the analysis of observational studies for causal effects: parallels with the design of randomized trials - NORC at the University of Chicago
Opens in a new window

arxiv.org
A fast PC algorithm for high dimensional causal discovery with multi-core PCs - arXiv
Opens in a new window

stat.cmu.edu
Discovering Causal Structure from Observations - Statistics & Data Science
Opens in a new window

medium.com
Introducing Conditional Independence and Causal Discovery | by Jakob Runge - Medium
Opens in a new window

proceedings.nips.cc
Bridging Imagination and Reality for Model-Based Deep Reinforcement Learning - NIPS
Opens in a new window

proceedings.neurips.cc
Model-Based Reinforcement Learning via Imagination with Derived Memory - NeurIPS
Opens in a new window

reddit.com
[D] As of 2020, how does model-based RL compare with model-free RL? What's the state of the art in model-based RL? : r/reinforcementlearning - Reddit
Opens in a new window

arxiv.org
What model does MuZero learn? - arXiv
Opens in a new window

youtube.com
Rufin VanRullen - "The Global Latent Workspace: a model of cognition with AI applications"
Opens in a new window

arxiv.org
[2012.10390] Deep Learning and the Global Workspace Theory - arXiv
Opens in a new window

escholarship.org
UC Merced - eScholarship.org
Opens in a new window
Deconstructing Cognitive and AI Frameworks
I am currently synthesizing the structural logic behind autonomous exploration and cognitive modeling. I have identified that the Intrinsic Curiosity Module distinguishes itself by calculating prediction errors within a latent feature space rather than raw sensory inputs, which effectively filters out irrelevant environmental noise. Similarly, I am thinking through how Random Network Distillation creates a dense reward signal in traditionally sparse environments by measuring the discrepancy between a fixed, random network and a trainable predictor network, essentially transforming novelty into a quantifiable objective.
Bridging Theory and Computation
I am working through the complexities of causal discovery and the computational realization of consciousness. A primary focus is resolving how specialized, unconscious modules compete for access to a global workspace and how the resulting 'broadcast' is implemented through shared latent representations in deep learning architectures. Furthermore, I am clarifying the mathematical distinctions in causal representation learning, specifically how the PC algorithm transitions from basic conditional independence tests to identifying the directional flow of causation, moving significantly beyond standard statistical correlation.
Investigating Model-Based Intelligence
Moving forward, I will be conducting a deep dive into the mechanics of world models to understand how agents can learn and plan entirely within an 'imagined' latent space. I intend to explore the specific implementation of rule-free planning and the information-theoretic calculation of empowerment, which measures an agent's potential influence over its future states. This will involve exploring themes related to mutual information, variational autoencoders, and the transition from purely predictive modeling to active environmental control.

Researching websites...

papers.neurips.cc
Learning to Predict Without Looking Ahead: World Models Without Forward Prediction - NeurIPS

smartlabai.medium.com
World models — a reinforcement learning story | by SmartLab AI - Medium

arxiv.org
[1803.10122] World Models - arXiv

worldmodels.github.io
World Models

cl.cam.ac.uk
World Models - David Ha, Jürgen Schmidhuber

researchgate.net
Dream to Control: Learning Behaviors by Latent Imagination - ResearchGate

openreview.net
DREAM TO CONTROL: LEARNING BEHAVIORS BY LATENT IMAGINATION - OpenReview

semanticscholar.org
[PDF] Dreaming: Model-based Reinforcement Learning by Latent Imagination without Reconstruction | Semantic Scholar

arxiv.org
[1912.01603] Dream to Control: Learning Behaviors by Latent Imagination - arXiv

youtube.com
Dream to Control: Learning Behaviors by Latent Imagination - YouTube

milvus.io
How does MuZero learn without knowing the environment? - Milvus

arxiv.org
[1911.08265] Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model

idi.ntnu.no
Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model - Department of Computer Science (IDI)

deepmind.google
MuZero: Mastering Go, chess, shogi and Atari without rules - Google DeepMind

reddit.com
[N] No rules, no problem: DeepMind's MuZero masters games while learning how to play them : r/MachineLearning - Reddit

yi-t.github.io
Toward Understanding State Representation Learning in MuZero: A Case Study in Linear Quadratic Gaussian Control - Yi Tian

arxiv.org
Demystifying MuZero Planning: Interpreting the Learned Model - arXiv

en.wikipedia.org
MuZero - Wikipedia

arxiv.org
What model does MuZero learn? - arXiv

julian.ac
MuZero Intuition - Julian Schrittwieser

mdpi.com
Curiosity-Driven Exploration in Reinforcement Learning: An Adaptive Self-Supervised Learning Approach for Playing Action Games - MDPI

emergentmind.com
Intrinsic Curiosity Modules - Emergent Mind

pathak22.github.io
Curiosity-driven Exploration by Self-supervised Prediction - Deepak Pathak

proceedings.mlr.press
Curiosity-driven Exploration by Self-supervised Prediction - Proceedings of Machine Learning Research

arxiv.org
Curiosity-driven Exploration by Self-supervised Prediction - arXiv

pmc.ncbi.nlm.nih.gov
Keep Your Options Open: An Information-Based Driving Principle for Sensorimotor Systems

uhra.herts.ac.uk
Empowerment: A Universal Agent-Centric Measure of Control

lizier.me
All Else Being Equal Be Empowered – Klyubin, Polani, Nahaniv - Joseph Lizier

alignmentforum.org
Empowerment is (almost) All We Need - AI Alignment Forum

www2.eecs.berkeley.edu
Efficient Empowerment Estimation for Unsupervised Stabilization - EECS at Berkeley

pure.ed.ac.uk
Exploration by random network distillation

docs.cleanrl.dev
Random Network Distillation (RND) - CleanRL

arxiv.org
[1810.12894] Exploration by Random Network Distillation - arXiv

semanticscholar.org
[PDF] Exploration by Random Network Distillation - Semantic Scholar

v1.endtoend.ai
Slow Papers: Exploration by Random Network Distillation (Burda et al., 2018) | endtoendAI

pyoudeyer.com
Intrinsic Motivation Systems for Autonomous Mental Development - Pierre-Yves Oudeyer

web.eecs.umich.edu
Intrinsic Motivation Systems for Autonomous Mental Development

cs.swarthmore.edu
Intrinsic Motivation Systems for Autonomous Mental Development - Computer Science

pmc.ncbi.nlm.nih.gov
What is Intrinsic Motivation? A Typology of Computational Approaches - PMC

web-archive.southampton.ac.uk
Intrinsic Motivation Systems for Autonomous Mental Development - Cogprints - University of Southampton Web Archive

frontiersin.org
Global Workspace Theory (GWT) and Prefrontal Cortex: Recent Developments - Frontiers

antoniocasella.eu
The Global Neuronal Workspace Model of Conscious Access: From Neuronal Architectures to Clinical Applications - Antonio Casella

pnas.org
A neuronal model of a global workspace in effortful cognitive tasks - PNAS

frontiersin.org
Hypothesis on the functional advantages of the selection-broadcast cycle structure: global workspace theory and dealing with a real-time world - Frontiers

pubmed.ncbi.nlm.nih.gov
Deep learning and the Global Workspace Theory - PubMed

researchgate.net
Deep learning and the Global Workspace Theory | Request PDF - ResearchGate

semanticscholar.org
[PDF] Deep learning and the Global Workspace Theory | Semantic Scholar

researchgate.net
Embodiment and the Inner Life: Cognition and Consciousness in the Space of Possible Minds - ResearchGate

books.google.com
Embodiment and the Inner Life: Cognition and Consciousness in the Space of Possible Minds - Google Books

digitalcommons.memphis.edu
Global Workspace Theory, Shanahan, and LIDA - University of Memphis Digital Commons

doc.ic.ac.uk
Embodiment and the inner life: Cognition and consciousness in the space of possible minds - Department of Computing - Imperial College London

pmc.ncbi.nlm.nih.gov
Conscious Processing and the Global Neuronal Workspace Hypothesis - PubMed Central

en.wikipedia.org
Global workspace theory - Wikipedia

arxiv.org
Hypothesis on the Functional Advantages of the Selection-Broadcast Cycle Structure: Global Workspace Theory and Dealing with a Real-Time World - arXiv

academic.oup.com
evolutionary origins of the Global Neuronal Workspace in vertebrates - Oxford Academic

selfawarepatterns.com
Global workspace theory: consciousness as brain wide information sharing - SelfAwarePatterns

yzhu.io
Lab 4: Peter-Clark (PC) Algorithm - Yixin Zhu | PKU

ckassaad.github.io
Causal discovery

opensource.salesforce.com
PC Algorithm for Tabular Causal Discovery - Open Source Projects Built at Salesforce

stat.cmu.edu
Discovering Causal Structure from Observations - Statistics & Data Science

jmlr.org
Order-Independent Constraint-Based Causal Structure Learning

activeloop.ai
What is Pearl's Causal Calculus? | Activeloop Glossary

pmc.ncbi.nlm.nih.gov
An Introduction to Causal Inference - PMC - PubMed Central

cs.ubc.ca
Introduction to Causal Calculus - UBC Computer Science

inference.vc
ML beyond Curve Fitting: An Intro to Causal Inference and do-Calculus

reddit.com
If correlation doesn't imply causation, then what does? : r/slatestarcodex - Reddit

research-collection.ethz.ch
Toward Causal Representation Learning - ETH Zurich Research Collection

amazon.science
NeurIPS: Why causal-representation learning may be the future of AI - Amazon Science

arxiv.org
[2102.11107] Towards Causal Representation Learning - arXiv

reddit.com
[R] Towards Causal Representation Learning : r/MachineLearning - Reddit

arxiv.org
Towards Causal Representation Learning - arXiv

pmc.ncbi.nlm.nih.gov
The target trial framework for causal inference from observational data: Why and when is it helpful? - PMC - NIH

pmc.ncbi.nlm.nih.gov
Causal Inference Methods for Combining Randomized Trials and Observational Studies: A Review - PMC - PubMed Central

cambridge.org
Interventions and Causal Inference | Philosophy of Science | Cambridge Core

proceedings.neurips.cc
Causal discovery from observational and interventional data across multiple environments - NeurIPS

norc.org
The design versus the analysis of observational studies for causal effects: parallels with the design of randomized trials - NORC at the University of Chicago
Refining Adaptive Curiosity
I have made significant progress in distinguishing between different forms of internal drive. While some frameworks focus on simple prediction errors, I am now synthesizing how more sophisticated systems prioritize actual learning progress—favoring environments that are neither too predictable nor too chaotic. I’ve also clarified why certain exploration methods excel in notoriously difficult environments like Montezuma’s Revenge; by utilizing deterministic prediction targets, they effectively ignore distracting environmental noise that might otherwise trap simpler curious agents.
Mapping Causal and Cognitive Architectures
I am currently refining my understanding of how causal structures are extracted from observational data through a multi-stage process involving connectivity mapping and the identification of directional 'colliders.' This aligns with my investigation into cognitive architectures, where I am analyzing how information becomes globally available through a non-linear process of activation. I've found that this 'broadcast' effectively allows localized, specialized modules to influence a unified, shared representation space, which is critical for complex decision-making.
Investigating Algorithmic Implementation
My next steps involve a targeted investigation into the precise mathematical routines used to calculate an agent’s potential influence over its future, specifically checking for the use of iterative algorithms to determine channel capacity. I will also be looking deeper into the integration of learned internal models with tree-search strategies, focusing on how systems can navigate possible futures without being explicitly programmed with the rules of their environment.

Researching websites...

arxiv.org
Exploration by Random Network Distillation

arxiv.org
Mastering Atari, Go, Chess and Shogi by Planning with a Learned ...

ckassaad.github.io
Causal discovery

uhra.herts.ac.uk
Empowerment: A Universal Agent-Centric Measure of Control

pubmed.ncbi.nlm.nih.gov
Deep learning and the Global Workspace Theory - PubMed
Quantifying Agency and Curiosity
I am synthesizing how agency can be mathematically quantified through the principle of empowerment, treating the interaction between an agent's actions and its perceptions as a communication channel. By calculating the capacity of this channel using iterative optimization, I can measure an agent's potential for influence. I have also clarified how modern curiosity frameworks solve the 'noisy TV' problem—where agents get trapped by unpredictable but irrelevant stimuli—by distilling intrinsic rewards from a fixed, random target that ignores non-deterministic environmental noise.
Architectural Competition and Directional Flow
I have been thinking through the 'ignition' mechanism in cognitive architectures, where a non-linear surge in neural activity allows a local processor to overcome inhibitory signals and broadcast its information globally. This transition from local to global accessibility is what allows specialized modules to collaborate on complex tasks. Simultaneously, I am refining the logic of causal orientation, specifically how to identify 'colliders' by determining if a common consequence is excluded from the set of variables that otherwise renders two factors independent.
Differentiating Synthetic Reasoning
I am currently distinguishing between two fundamental ways agents utilize their internal models: learning behaviors by backpropagating through imagined trajectories versus using those same models for real-time lookahead planning. While one approach optimizes a policy within a dream-like simulation, the other uses the model to navigate potential futures during active decision-making. My next actions will focus on verifying the precise attention-based mechanisms used for module selection and finalizing the logical rules that propagate directional flow across an entire causal network.

Researching websites...

tyler-causality.streamlit.app
PC Algorithm

arxiv.org
A fast PC algorithm for high dimensional causal discovery with multi-core PCs - arXiv

stat.cmu.edu
Discovering Causal Structure from Observations - Statistics & Data Science

medium.com
Introducing Conditional Independence and Causal Discovery | by Jakob Runge - Medium

ckassaad.github.io
Causal discovery

openreview.net
DREAM TO CONTROL: LEARNING BEHAVIORS BY LATENT IMAGINATION - OpenReview

proceedings.nips.cc
Bridging Imagination and Reality for Model-Based Deep Reinforcement Learning - NIPS

proceedings.neurips.cc
Model-Based Reinforcement Learning via Imagination with Derived Memory - NeurIPS

reddit.com
[D] As of 2020, how does model-based RL compare with model-free RL? What's the state of the art in model-based RL? : r/reinforcementlearning - Reddit

arxiv.org
What model does MuZero learn? - arXiv

medium.com
Engineering Consciousness: Global Workspace Theory and Higher-Order Theories

youtube.com
Rufin VanRullen - "The Global Latent Workspace: a model of cognition with AI applications"

arxiv.org
Deep Learning and the Global Workspace Theory - arXiv

arxiv.org
[2012.10390] Deep Learning and the Global Workspace Theory - arXiv

escholarship.org
UC Merced - eScholarship.org
Decoding the Computational Global Workspace
I have successfully synthesized the computational implementation of cognitive broadcasting, moving beyond the theater metaphor to a concrete architecture of multimodal latent translation. I've found that this process relies on creating an amodal workspace where specialized modules map their specific data into a shared representational space, ensuring coherence through cycle-consistency. This mechanism allows information to be globally accessible, effectively enabling localized networks to collaborate on a unified internal state and resolve competing inputs through non-linear surges in activation.
Synthesizing Worlds and Imaginative Planning
I am now clearly distinguishing between how agents utilize internal simulations to learn versus how they use them to plan. I have resolved the conceptual overlap between world models by identifying that while some frameworks optimize behaviors by backpropagating value gradients through an imagined latent rollout, others integrate a learned transition model directly into real-time lookahead tree-search routines. This distinction is vital for understanding how autonomous systems can navigate complex environments without prior knowledge of the underlying rules, using their internal models to predict future rewards and state transitions entirely in imagination.
Finalizing the Logic of Causality and Control
My final stage of synthesis involves structuring the complete sequence of causal discovery, from the initial pruning of connections based on statistical independence to the propagation of directional flow using fixed logical rules. I am currently organizing the final details regarding how interventions—mathematically represented as modifications to a causal graph—allow a system to differentiate between simple observation and true causal influence. I am now prepared to finalize the mapping of these directional relationships to complete the broader picture of how agents build and use internal models of the world.
