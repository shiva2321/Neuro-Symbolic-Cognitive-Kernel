# NSCK Enhancement Implementation Plan 2026

**Document Purpose:** Detailed implementation roadmap with agent task assignments for system improvements

**Based On:** RESEARCH_SYNTHESIS_2026.md

**Timeline:** 12 weeks (3 phases)

**Last Updated:** February 11, 2026

---

## Implementation Strategy

### Phased Approach
- **Phase 1 (Weeks 1-4):** HIGH priority improvements
- **Phase 2 (Weeks 5-8):** MEDIUM priority improvements  
- **Phase 3 (Weeks 9-12):** LOW priority optimizations

### Success Criteria
- All improvements must have automated tests
- Benchmark performance vs baseline
- Document before/after metrics
- No regressions in existing capabilities

---

## Phase 1: Core Architecture Improvements (Weeks 1-4)

### 1.1 Task-Adaptive VSA Encoding

**Goal:** Implement dynamic encoding strategies that adapt based on task requirements

**Impact:** +15% accuracy in classification and reasoning tasks

**Agent:** `vsa-core-agent`

**Tasks:**
1. **Design Encoding Strategy Manager**
   ```python
   # Location: nsck-demo/python/hypervec_shim.py
   
   class EncodingStrategyManager:
       """Manages task-adaptive encoding strategies for VSA"""
       
       STRATEGIES = {
           'learning': {
               'correlation_target': 0.75,  # High correlation for generalization
               'noise_level': 0.05,
               'bundling_threshold': 0.6
           },
           'symbolic': {
               'correlation_target': 0.50,  # Orthogonal for separability
               'noise_level': 0.0,
               'bundling_threshold': 0.8
           },
           'hybrid': {
               'correlation_target': 0.62,  # Balanced
               'noise_level': 0.02,
               'bundling_threshold': 0.7
           }
       }
       
       def __init__(self, default_strategy='hybrid'):
           self.current_strategy = default_strategy
           self.strategy_history = []
       
       def set_strategy(self, strategy_name, context=None):
           """Switch encoding strategy based on task requirements"""
           if strategy_name not in self.STRATEGIES:
               raise ValueError(f"Unknown strategy: {strategy_name}")
           
           self.current_strategy = strategy_name
           self.strategy_history.append({
               'strategy': strategy_name,
               'context': context,
               'timestamp': time.time()
           })
       
       def encode_with_strategy(self, concept, strategy=None):
           """Encode concept using current or specified strategy"""
           strategy = strategy or self.current_strategy
           params = self.STRATEGIES[strategy]
           
           # Generate base encoding
           encoding = self._base_encode(concept)
           
           # Apply strategy-specific transformations
           if params['correlation_target'] > 0.6:
               # Learning mode: Add controlled noise for generalization
               encoding = self._add_controlled_noise(
                   encoding, params['noise_level']
               )
           else:
               # Symbolic mode: Maximize orthogonality
               encoding = self._orthogonalize(encoding)
           
           return encoding
       
       def auto_select_strategy(self, task_type):
           """Automatically select strategy based on task"""
           if task_type in ['classification', 'prediction', 'learning']:
               return 'learning'
           elif task_type in ['reasoning', 'planning', 'rules']:
               return 'symbolic'
           else:
               return 'hybrid'
   ```

2. **Integrate with Cognitive Engine**
   ```python
   # Location: nsck-demo/python/cognitive_engine.py
   
   class CognitiveEngine:
       def __init__(self):
           # ... existing init ...
           self.encoding_manager = EncodingStrategyManager()
       
       def decide(self, state_hv, action_hvs, task_context):
           # Auto-select encoding strategy
           if 'rule_reasoning' in task_context:
               self.encoding_manager.set_strategy('symbolic')
           elif 'neural_learning' in task_context:
               self.encoding_manager.set_strategy('learning')
           
           # ... rest of decide logic ...
   ```

3. **Create Comprehensive Tests**
   ```python
   # Location: nsck-demo/tests/test_adaptive_encoding.py
   
   def test_learning_strategy_high_correlation():
       """Test that learning strategy produces correlated encodings"""
       manager = EncodingStrategyManager()
       manager.set_strategy('learning')
       
       # Encode related concepts
       dog_hv = manager.encode_with_strategy('dog')
       puppy_hv = manager.encode_with_strategy('puppy')
       
       similarity = compute_similarity(dog_hv, puppy_hv)
       assert similarity > 0.7, "Learning strategy should produce high correlation"
   
   def test_symbolic_strategy_orthogonal():
       """Test that symbolic strategy produces orthogonal encodings"""
       manager = EncodingStrategyManager()
       manager.set_strategy('symbolic')
       
       # Encode distinct concepts
       left_hv = manager.encode_with_strategy('LEFT')
       right_hv = manager.encode_with_strategy('RIGHT')
       
       similarity = compute_similarity(left_hv, right_hv)
       assert 0.45 < similarity < 0.55, "Symbolic strategy should be orthogonal"
   
   def test_auto_strategy_selection():
       """Test automatic strategy selection"""
       manager = EncodingStrategyManager()
       
       assert manager.auto_select_strategy('classification') == 'learning'
       assert manager.auto_select_strategy('planning') == 'symbolic'
       assert manager.auto_select_strategy('unknown') == 'hybrid'
   
   def test_strategy_impact_on_classification():
       """Benchmark: Learning strategy should improve classification"""
       # ... run classification task with both strategies ...
       # Assert learning strategy achieves 10-15% better accuracy
   ```

4. **Benchmark Performance**
   ```bash
   # Create benchmark script
   python nsck-demo/python/benchmark_encoding_strategies.py \
       --tasks classification,reasoning \
       --iterations 100 \
       --output results/encoding_benchmark.json
   ```

**Success Metrics:**
- Learning strategy: 0.7-0.8 correlation for related concepts ✓
- Symbolic strategy: ~0.5 correlation for distinct concepts ✓
- Classification accuracy: +10-15% with learning strategy ✓
- Reasoning accuracy: +12% with symbolic strategy ✓
- Tests: 10+ tests covering all strategies ✓

**Deliverables:**
- [ ] `EncodingStrategyManager` class implemented
- [ ] Integration with `CognitiveEngine`
- [ ] 10+ comprehensive tests
- [ ] Performance benchmarks documented
- [ ] Migration guide for existing code

**Estimated Effort:** 5 days

---

### 1.2 Distributed Global Workspace

**Goal:** Replace single bottleneck workspace with dynamic distributed workspaces

**Impact:** +30-40% throughput, -25% latency

**Agent:** `workspace-agent`

**Tasks:**
1. **Design Distributed Workspace Architecture**
   ```python
   # Location: nsck-demo/python/distributed_workspace.py
   
   class DistributedGlobalWorkspace:
       """Multiple parallel global workspaces with meta-level arbitration"""
       
       def __init__(self, num_workspaces=3):
           self.workspaces = [
               GlobalWorkspace(id=i) for i in range(num_workspaces)
           ]
           self.context_router = ContextRouter()
           self.meta_arbiter = MetaArbiter()
           
       def compete(self, coalitions, context):
           """
           Parallel competition across distributed workspaces
           
           Args:
               coalitions: List of Coalition objects
               context: Current cognitive context
               
           Returns:
               Winner coalition or None
           """
           # Phase 1: Context-based partitioning
           partitions = self.context_router.partition_coalitions(
               coalitions, context, num_partitions=len(self.workspaces)
           )
           
           # Phase 2: Parallel competition (can be threaded)
           winners = []
           for workspace, partition in zip(self.workspaces, partitions):
               if partition:  # Skip empty partitions
                   winner = workspace.compete(partition)
                   if winner:
                       winners.append(winner)
           
           # Phase 3: Meta-level arbitration
           if len(winners) == 0:
               return None
           elif len(winners) == 1:
               return winners[0]
           else:
               return self.meta_arbiter.resolve(winners, context)
   
   class ContextRouter:
       """Routes coalitions to appropriate workspaces based on context"""
       
       def partition_coalitions(self, coalitions, context, num_partitions):
           """Partition coalitions by domain/modality/urgency"""
           partitions = [[] for _ in range(num_partitions)]
           
           for coalition in coalitions:
               # Assign to workspace based on coalition properties
               workspace_id = self._select_workspace(coalition, context)
               partitions[workspace_id].append(coalition)
           
           return partitions
       
       def _select_workspace(self, coalition, context):
           """Select appropriate workspace for coalition"""
           # Partition by domain
           if 'perception' in coalition.tags:
               return 0  # Perceptual workspace
           elif 'reasoning' in coalition.tags:
               return 1  # Reasoning workspace
           else:
               return 2  # General workspace
   
   class MetaArbiter:
       """Resolves conflicts between workspace winners"""
       
       def resolve(self, winners, context):
           """Meta-level competition between winners"""
           # Check for conflicts
           if self._detect_conflict(winners):
               return self._resolve_conflict(winners, context)
           
           # No conflict: Select highest activation
           return max(winners, key=lambda w: w.activation)
       
       def _detect_conflict(self, winners):
           """Detect if winners recommend incompatible actions"""
           actions = [w.action for w in winners]
           return len(set(actions)) > 1
       
       def _resolve_conflict(self, winners, context):
           """Resolve conflicting recommendations"""
           # Use context-based priority
           priorities = {
               'safety': 3,
               'goal': 2,
               'exploration': 1
           }
           
           scored_winners = []
           for winner in winners:
               priority = priorities.get(winner.category, 0)
               score = winner.activation * (1 + 0.2 * priority)
               scored_winners.append((score, winner))
           
           return max(scored_winners, key=lambda x: x[0])[1]
   ```

2. **Integrate with Cognitive Engine**
   ```python
   # Location: nsck-demo/python/cognitive_engine.py
   
   # Add configuration flag
   USE_DISTRIBUTED_WORKSPACE = True
   
   class CognitiveEngine:
       def __init__(self, config=None):
           # ... existing init ...
           if config and config.get('distributed_workspace', False):
               self.workspace = DistributedGlobalWorkspace(num_workspaces=3)
           else:
               self.workspace = GlobalWorkspace()
       
       def decide(self, state_hv, action_hvs, context):
           # ... prepare coalitions ...
           
           # Compete (now using distributed workspace)
           winner = self.workspace.compete(coalitions, context)
           
           # ... rest of logic ...
   ```

3. **Performance Testing**
   ```python
   # Location: nsck-demo/tests/test_distributed_workspace.py
   
   def test_parallel_competition():
       """Test that parallel workspaces process coalitions correctly"""
       dws = DistributedGlobalWorkspace(num_workspaces=3)
       
       # Create coalitions for different domains
       coalitions = [
           Coalition(tags=['perception'], activation=0.8),
           Coalition(tags=['reasoning'], activation=0.9),
           Coalition(tags=['general'], activation=0.7),
       ]
       
       winner = dws.compete(coalitions, context={})
       assert winner is not None
       assert winner.activation == 0.9
   
   def test_throughput_improvement():
       """Benchmark: Distributed should be 30-40% faster"""
       single_ws = GlobalWorkspace()
       distributed_ws = DistributedGlobalWorkspace(num_workspaces=3)
       
       # Generate 1000 coalition sets
       coalition_sets = generate_random_coalition_sets(1000)
       
       # Benchmark single workspace
       single_time = benchmark_workspace(single_ws, coalition_sets)
       
       # Benchmark distributed workspace
       distributed_time = benchmark_workspace(distributed_ws, coalition_sets)
       
       improvement = (single_time - distributed_time) / single_time
       assert improvement > 0.30, f"Expected 30%+ improvement, got {improvement}"
   
   def test_meta_arbiter_conflict_resolution():
       """Test meta-level conflict resolution"""
       arbiter = MetaArbiter()
       
       # Create conflicting winners
       winner1 = Coalition(action='LEFT', activation=0.8, category='exploration')
       winner2 = Coalition(action='RIGHT', activation=0.75, category='safety')
       
       # Safety should win despite lower activation
       resolved = arbiter.resolve([winner1, winner2], context={})
       assert resolved.action == 'RIGHT'
   ```

**Success Metrics:**
- Throughput: +30-40% vs single workspace ✓
- Latency: -25% average decision time ✓
- Correctness: 100% decision accuracy maintained ✓
- Scalability: Handles 10× more coalitions ✓
- Tests: 8+ tests covering all scenarios ✓

**Deliverables:**
- [ ] `DistributedGlobalWorkspace` implementation
- [ ] `ContextRouter` and `MetaArbiter` classes
- [ ] Integration with `CognitiveEngine`
- [ ] 8+ comprehensive tests
- [ ] Performance benchmarks
- [ ] Configuration guide

**Estimated Effort:** 7 days

---

### 1.3 Generalization-Preserved Learning (GPL)

**Goal:** Implement GPL to reduce catastrophic forgetting from 8% to 3%

**Impact:** 62% reduction in forgetting, +5% memory overhead

**Agent:** `continual-learning-agent`

**Tasks:**
1. **Implement Hyperbolic Embedding**
   ```python
   # Location: nsck-demo/python/hyperbolic_learning.py
   
   import torch
   import numpy as np
   
   class HyperbolicEmbedding:
       """Embed tasks in hyperbolic space for distance preservation"""
       
       def __init__(self, dim=128, curvature=-1.0):
           self.dim = dim
           self.curvature = curvature
       
       def embed_task(self, task_data):
           """
           Embed task in hyperbolic space (Poincaré ball model)
           
           Args:
               task_data: Task dataset or representative samples
               
           Returns:
               Hyperbolic embedding vector
           """
           # Extract task statistics
           feature_mean = self._compute_feature_mean(task_data)
           feature_cov = self._compute_feature_covariance(task_data)
           
           # Map to Euclidean space
           euclidean_embedding = np.concatenate([
               feature_mean.flatten(),
               feature_cov.flatten()[:self.dim - len(feature_mean.flatten())]
           ])
           
           # Project to Poincaré ball
           hyperbolic_embedding = self._exponential_map(euclidean_embedding)
           
           return hyperbolic_embedding
       
       def hyperbolic_distance(self, embed1, embed2):
           """Compute hyperbolic distance in Poincaré ball"""
           delta = np.linalg.norm(embed1 - embed2)
           norm1 = np.linalg.norm(embed1)
           norm2 = np.linalg.norm(embed2)
           
           # Poincaré distance formula
           numerator = 2 * delta**2
           denominator = (1 - norm1**2) * (1 - norm2**2)
           
           distance = np.arccosh(1 + numerator / denominator)
           return distance
       
       def _exponential_map(self, euclidean_vec):
           """Map from tangent space to Poincaré ball"""
           norm = np.linalg.norm(euclidean_vec)
           if norm < 1e-10:
               return euclidean_vec
           
           return np.tanh(np.sqrt(-self.curvature) * norm / 2) * \
                  euclidean_vec / (np.sqrt(-self.curvature) * norm)
   
   class GeneralizationPreservedLearning:
       """GPL: Continual learning with generalization preservation"""
       
       def __init__(self, model, lambda_gpl=0.1):
           self.model = model
           self.lambda_gpl = lambda_gpl
           self.task_embeddings = {}
           self.hyperbolic_space = HyperbolicEmbedding()
       
       def learn_task(self, task_id, task_data, num_epochs=10):
           """Learn new task while preserving generalization"""
           
           # Embed current task
           current_embedding = self.hyperbolic_space.embed_task(task_data)
           
           # Training loop
           for epoch in range(num_epochs):
               for batch in task_data:
                   # Forward pass
                   loss = self.model.compute_loss(batch)
                   
                   # Compute preservation constraint
                   preservation_loss = self._compute_preservation_constraint(
                       current_embedding
                   )
                   
                   # Combined loss
                   total_loss = loss + self.lambda_gpl * preservation_loss
                   
                   # Backward pass
                   total_loss.backward()
                   self.model.optimizer.step()
           
           # Store task embedding
           self.task_embeddings[task_id] = current_embedding
       
       def _compute_preservation_constraint(self, current_embedding):
           """Constraint to preserve distances in hyperbolic space"""
           if not self.task_embeddings:
               return 0.0
           
           constraint = 0.0
           for task_id, past_embedding in self.task_embeddings.items():
               # Compute desired distance preservation
               distance = self.hyperbolic_space.hyperbolic_distance(
                   current_embedding, past_embedding
               )
               
               # Penalize if distance changes too much
               # (encourage parameter updates that maintain distances)
               constraint += distance ** 2
           
           return constraint / len(self.task_embeddings)
   ```

2. **Integrate with Continual Learning Module**
   ```python
   # Location: nsck-demo/python/continual_learning.py
   
   class ContinualLearner:
       def __init__(self, use_gpl=True):
           # ... existing init ...
           self.use_gpl = use_gpl
           if use_gpl:
               self.gpl = GeneralizationPreservedLearning(
                   self.model, lambda_gpl=0.1
               )
       
       def learn_task(self, task_id, task_data):
           if self.use_gpl:
               self.gpl.learn_task(task_id, task_data)
           else:
               # Original EWC-based learning
               self._ewc_learn_task(task_id, task_data)
   ```

3. **Comprehensive Testing**
   ```python
   # Location: nsck-demo/tests/test_gpl.py
   
   def test_gpl_reduces_forgetting():
       """Test that GPL reduces forgetting vs baseline"""
       
       # Baseline: EWC only
       baseline_learner = ContinualLearner(use_gpl=False)
       baseline_learner.learn_task('task_a', task_a_data)
       baseline_learner.learn_task('task_b', task_b_data)
       baseline_forgetting = baseline_learner.evaluate_task('task_a')
       
       # GPL
       gpl_learner = ContinualLearner(use_gpl=True)
       gpl_learner.learn_task('task_a', task_a_data)
       gpl_learner.learn_task('task_b', task_b_data)
       gpl_forgetting = gpl_learner.evaluate_task('task_a')
       
       # GPL should reduce forgetting by ~62%
       improvement = (baseline_forgetting - gpl_forgetting) / baseline_forgetting
       assert improvement > 0.55, f"Expected 55%+ improvement, got {improvement}"
   
   def test_hyperbolic_embedding():
       """Test hyperbolic embedding produces valid embeddings"""
       hyp = HyperbolicEmbedding(dim=128)
       
       task_embedding = hyp.embed_task(generate_task_data())
       
       # Check in Poincaré ball (norm < 1)
       norm = np.linalg.norm(task_embedding)
       assert norm < 1.0, "Embedding should be in Poincaré ball"
   
   def test_distance_preservation():
       """Test that GPL preserves task distances"""
       gpl = GeneralizationPreservedLearning(model)
       
       # Learn three tasks
       gpl.learn_task('A', task_a_data)
       gpl.learn_task('B', task_b_data)
       gpl.learn_task('C', task_c_data)
       
       # Compute initial distances
       dist_ab_initial = gpl.hyperbolic_space.hyperbolic_distance(
           gpl.task_embeddings['A'], gpl.task_embeddings['B']
       )
       
       # Learn fourth task
       gpl.learn_task('D', task_d_data)
       
       # Re-compute distance A-B
       dist_ab_after = gpl.hyperbolic_space.hyperbolic_distance(
           gpl.task_embeddings['A'], gpl.task_embeddings['B']
       )
       
       # Distance should be preserved
       distance_change = abs(dist_ab_after - dist_ab_initial)
       assert distance_change < 0.1, "Distance preservation violated"
   ```

**Success Metrics:**
- Forgetting: 3% vs 8% baseline (62% reduction) ✓
- Memory overhead: +5% (embeddings only) ✓
- Training time: +8% overhead acceptable ✓
- Task sequence: Tested on 5+ sequential tasks ✓
- Tests: 12+ tests covering all aspects ✓

**Deliverables:**
- [ ] `HyperbolicEmbedding` class
- [ ] `GeneralizationPreservedLearning` class
- [ ] Integration with `ContinualLearner`
- [ ] 12+ comprehensive tests
- [ ] Benchmark vs EWC baseline
- [ ] Ablation studies

**Estimated Effort:** 6 days

---

### 1.4 Meta-Analogical Transfer

**Goal:** Enable transfer of entire reasoning strategies, not just individual rules

**Impact:** +25% zero-shot performance, 3× fewer examples needed

**Agent:** `transfer-learning-agent`

**Tasks:**
1. **Design Meta-Analogy Framework**
   ```python
   # Location: nsck-demo/python/meta_analogy.py
   
   class ReasoningStrategy:
       """Abstract reasoning strategy extracted from multiple episodes"""
       
       def __init__(self, domain):
           self.domain = domain
           self.retrieval_heuristics = []
           self.mapping_preferences = []
           self.inference_rules = []
           self.meta_patterns = []
       
       def to_dict(self):
           return {
               'domain': self.domain,
               'retrieval_heuristics': self.retrieval_heuristics,
               'mapping_preferences': self.mapping_preferences,
               'inference_rules': self.inference_rules,
               'meta_patterns': self.meta_patterns
           }
   
   class MetaAnalogicalTransfer:
       """Transfer reasoning strategies across domains"""
       
       def __init__(self, analogy_engine):
           self.analogy_engine = analogy_engine
           self.reasoning_strategies = {}
           self.strategy_effectiveness = {}
       
       def extract_strategy(self, domain, analogy_episodes):
           """
           Extract abstract reasoning strategy from multiple episodes
           
           Args:
               domain: Source domain name
               analogy_episodes: List of analogy-making episodes
               
           Returns:
               ReasoningStrategy object
           """
           strategy = ReasoningStrategy(domain)
           
           # Extract retrieval heuristics
           strategy.retrieval_heuristics = self._extract_retrieval_patterns(
               analogy_episodes
           )
           
           # Extract mapping preferences
           strategy.mapping_preferences = self._extract_mapping_patterns(
               analogy_episodes
           )
           
           # Extract inference rules
           strategy.inference_rules = self._extract_inference_patterns(
               analogy_episodes
           )
           
           # Extract meta-level patterns
           strategy.meta_patterns = self._extract_meta_patterns(
               analogy_episodes
           )
           
           # Store strategy
           self.reasoning_strategies[domain] = strategy
           
           return strategy
       
       def transfer_strategy(self, source_domain, target_domain, target_task):
           """
           Transfer reasoning strategy from source to target domain
           
           Args:
               source_domain: Source domain name
               target_domain: Target domain name
               target_task: Task in target domain
               
           Returns:
               Transferred strategy adapted to target domain
           """
           if source_domain not in self.reasoning_strategies:
               raise ValueError(f"No strategy for domain: {source_domain}")
           
           source_strategy = self.reasoning_strategies[source_domain]
           target_strategy = ReasoningStrategy(target_domain)
           
           # Transfer retrieval heuristics
           target_strategy.retrieval_heuristics = self._transfer_heuristics(
               source_strategy.retrieval_heuristics,
               source_domain,
               target_domain
           )
           
           # Transfer mapping preferences
           target_strategy.mapping_preferences = self._transfer_preferences(
               source_strategy.mapping_preferences,
               source_domain,
               target_domain
           )
           
           # Transfer inference rules
           target_strategy.inference_rules = self._transfer_rules(
               source_strategy.inference_rules,
               source_domain,
               target_domain
           )
           
           # Adapt meta-patterns
           target_strategy.meta_patterns = self._adapt_meta_patterns(
               source_strategy.meta_patterns,
               target_task
           )
           
           return target_strategy
       
       def _extract_retrieval_patterns(self, episodes):
           """Extract common retrieval patterns across episodes"""
           patterns = []
           
           for episode in episodes:
               # What features drove retrieval?
               retrieval_cues = episode.get('retrieval_cues', [])
               
               for cue in retrieval_cues:
                   # Abstract the cue
                   abstract_cue = self._abstract_retrieval_cue(cue)
                   patterns.append(abstract_cue)
           
           # Find frequent patterns
           return self._find_frequent_patterns(patterns, min_frequency=0.3)
       
       def _extract_mapping_patterns(self, episodes):
           """Extract common mapping preferences"""
           mapping_types = []
           
           for episode in episodes:
               # What types of mappings succeeded?
               mappings = episode.get('successful_mappings', [])
               
               for mapping in mappings:
                   mapping_type = self._classify_mapping_type(mapping)
                   mapping_types.append(mapping_type)
           
           return self._rank_by_frequency(mapping_types)
       
       def _extract_inference_patterns(self, episodes):
           """Extract common inference patterns"""
           inference_rules = []
           
           for episode in episodes:
               # What inferences were made?
               inferences = episode.get('inferences', [])
               
               for inference in inferences:
                   abstract_rule = self._abstract_inference(inference)
                   inference_rules.append(abstract_rule)
           
           return self._consolidate_rules(inference_rules)
       
       def _extract_meta_patterns(self, episodes):
           """Extract meta-level patterns (patterns about patterns)"""
           meta_patterns = []
           
           # When to use retrieval vs mapping vs inference?
           for episode in episodes:
               episode_type = self._classify_episode_type(episode)
               success_rate = episode.get('success', 0)
               
               meta_patterns.append({
                   'type': episode_type,
                   'success': success_rate,
                   'context': episode.get('context', {})
               })
           
           return meta_patterns
       
       def apply_strategy(self, strategy, task):
           """Apply reasoning strategy to a new task"""
           
           # Use strategy heuristics for retrieval
           relevant_cases = self._retrieve_with_strategy(
               task, strategy.retrieval_heuristics
           )
           
           # Use strategy preferences for mapping
           mappings = self._map_with_strategy(
               task, relevant_cases, strategy.mapping_preferences
           )
           
           # Use strategy rules for inference
           inferences = self._infer_with_strategy(
               mappings, strategy.inference_rules
           )
           
           return inferences
   ```

2. **Integrate with Analogy Engine**
   ```python
   # Location: nsck-demo/python/analogy.py
   
   class AnalogyEngine:
       def __init__(self):
           # ... existing init ...
           self.meta_transfer = MetaAnalogicalTransfer(self)
           self.analogy_episodes = []
       
       def transfer_with_strategy(self, source_domain, target_domain, target_task):
           """Use meta-analogical transfer"""
           
           # Extract or retrieve strategy
           if source_domain not in self.meta_transfer.reasoning_strategies:
               # Extract strategy from recorded episodes
               domain_episodes = [
                   ep for ep in self.analogy_episodes
                   if ep['domain'] == source_domain
               ]
               self.meta_transfer.extract_strategy(source_domain, domain_episodes)
           
           # Transfer strategy
           target_strategy = self.meta_transfer.transfer_strategy(
               source_domain, target_domain, target_task
           )
           
           # Apply strategy
           result = self.meta_transfer.apply_strategy(target_strategy, target_task)
           
           return result
   ```

3. **Testing**
   ```python
   # Location: nsck-demo/tests/test_meta_analogy.py
   
   def test_strategy_extraction():
       """Test extraction of reasoning strategy from episodes"""
       meta_transfer = MetaAnalogicalTransfer(analogy_engine)
       
       # Generate analogy episodes
       episodes = generate_snake_analogy_episodes(num=20)
       
       # Extract strategy
       strategy = meta_transfer.extract_strategy('snake', episodes)
       
       assert len(strategy.retrieval_heuristics) > 0
       assert len(strategy.mapping_preferences) > 0
       assert len(strategy.inference_rules) > 0
   
   def test_strategy_transfer():
       """Test transfer of strategy across domains"""
       meta_transfer = MetaAnalogicalTransfer(analogy_engine)
       
       # Extract strategy from Snake
       snake_strategy = meta_transfer.extract_strategy(
           'snake', snake_episodes
       )
       
       # Transfer to Maze
       maze_strategy = meta_transfer.transfer_strategy(
           'snake', 'maze', maze_task
       )
       
       # Verify strategy adapted to Maze domain
       assert maze_strategy.domain == 'maze'
       assert len(maze_strategy.inference_rules) > 0
   
   def test_zero_shot_improvement():
       """Benchmark: Meta-transfer should improve zero-shot by 25%"""
       
       # Baseline: Regular analogical transfer
       baseline_engine = AnalogyEngine()
       baseline_performance = baseline_engine.zero_shot_eval('maze')
       
       # Meta-transfer
       meta_engine = AnalogyEngine()
       meta_engine.meta_transfer.extract_strategy('snake', snake_episodes)
       meta_performance = meta_engine.transfer_with_strategy(
           'snake', 'maze', maze_task
       )
       
       improvement = (meta_performance - baseline_performance) / baseline_performance
       assert improvement > 0.20, f"Expected 20%+ improvement, got {improvement}"
   ```

**Success Metrics:**
- Zero-shot performance: +25% ✓
- Sample efficiency: 3× fewer examples needed ✓
- Transfer to distant domains: Successful ✓
- Strategy extraction: Captures key patterns ✓
- Tests: 15+ tests covering all components ✓

**Deliverables:**
- [ ] `MetaAnalogicalTransfer` class
- [ ] `ReasoningStrategy` representation
- [ ] Integration with `AnalogyEngine`
- [ ] 15+ comprehensive tests
- [ ] Benchmarks vs baseline transfer
- [ ] Documentation with examples

**Estimated Effort:** 8 days

---

### 1.5 Dual-Process Architecture

**Goal:** Explicit System 1 (fast) and System 2 (slow) reasoning paths

**Impact:** +15% accuracy, +40% error detection, 5× faster on easy cases

**Agent:** `architecture-agent`

**Tasks:**
1. **Design Dual-Process System**
   ```python
   # Location: nsck-demo/python/dual_process.py
   
   class System1:
       """Fast, intuitive, automatic processing"""
       
       def __init__(self, snn, cached_rules):
           self.snn = snn  # Fast neural system
           self.cached_rules = cached_rules  # Compiled rules
           self.response_time_threshold = 0.1  # 100ms
       
       def decide(self, state):
           """Fast decision based on pattern matching"""
           start_time = time.time()
           
           # Check cached rules first (fastest)
           for rule in self.cached_rules:
               if rule.matches(state) and rule.confidence > 0.8:
                   return {
                       'action': rule.action,
                       'confidence': rule.confidence,
                       'method': 'cached_rule',
                       'time': time.time() - start_time
                   }
           
           # Fall back to SNN (fast but less reliable)
           snn_action, snn_confidence = self.snn.predict(state)
           
           return {
               'action': snn_action,
               'confidence': snn_confidence,
               'method': 'snn',
               'time': time.time() - start_time
           }
   
   class System2:
       """Slow, deliberate, analytical processing"""
       
       def __init__(self, planner, causal_reasoner, world_model):
           self.planner = planner
           self.causal_reasoner = causal_reasoner
           self.world_model = world_model
       
       def decide(self, state, context):
           """Deliberative decision with full reasoning"""
           start_time = time.time()
           
           # Causal analysis
           causal_chain = self.causal_reasoner.analyze(state)
           
           # World model simulation
           predictions = self.world_model.simulate_actions(state)
           
           # Planning
           plan = self.planner.plan(state, context['goal'])
           
           # Select best action based on analysis
           action = self._select_best_action(
               causal_chain, predictions, plan
           )
           
           return {
               'action': action,
               'confidence': 0.9,  # High confidence (deliberative)
               'method': 'deliberative',
               'reasoning': {
                   'causal': causal_chain,
                   'predictions': predictions,
                   'plan': plan
               },
               'time': time.time() - start_time
           }
   
   class DualProcessCognition:
       """Coordinate System 1 and System 2"""
       
       def __init__(self, system1, system2, metacognition):
           self.system1 = system1
           self.system2 = system2
           self.metacognition = metacognition
           
           # Statistics
           self.s1_count = 0
           self.s2_count = 0
           self.errors_caught = 0
       
       def decide(self, state, context):
           """Dual-process decision making"""
           
           # Always try System 1 first (fast)
           s1_response = self.system1.decide(state)
           
           # Check if System 2 needed
           needs_deliberation = self._needs_deliberation(
               s1_response, state, context
           )
           
           if not needs_deliberation:
               # Fast path: Use System 1
               self.s1_count += 1
               return s1_response
           
           # Slow path: Use System 2
           self.s2_count += 1
           s2_response = self.system2.decide(state, context)
           
           # Check for conflict
           if self._detect_conflict(s1_response, s2_response):
               self.errors_caught += 1
               return self._arbitrate(s1_response, s2_response, state, context)
           
           return s2_response
       
       def _needs_deliberation(self, s1_response, state, context):
           """Determine if System 2 deliberation is needed"""
           
           # Low confidence → deliberate
           if s1_response['confidence'] < 0.7:
               return True
           
           # High stakes → deliberate
           if context.get('stakes', 'normal') == 'high':
               return True
           
           # Metacognitive conflict detection
           if self.metacognition.detect_conflict():
               return True
           
           # Novel situation → deliberate
           if self.metacognition.novelty_score(state) > 0.8:
               return True
           
           return False
       
       def _detect_conflict(self, s1_response, s2_response):
           """Detect if S1 and S2 disagree"""
           return s1_response['action'] != s2_response['action']
       
       def _arbitrate(self, s1_response, s2_response, state, context):
           """Resolve conflict between S1 and S2"""
           
           # In general, prefer System 2 (more reliable)
           # But consider confidence and context
           
           if context.get('time_pressure', False):
               # Time pressure → favor S1
               return s1_response
           
           if s2_response.get('reasoning'):
               # S2 has explicit reasoning → trust it
               return s2_response
           
           # Default: S2 wins
           return s2_response
       
       def get_statistics(self):
           """Return dual-process statistics"""
           total = self.s1_count + self.s2_count
           return {
               'system1_usage': self.s1_count / total if total > 0 else 0,
               'system2_usage': self.s2_count / total if total > 0 else 0,
               'errors_caught': self.errors_caught,
               'error_rate': self.errors_caught / total if total > 0 else 0
           }
   ```

2. **Integrate with Cognitive Engine**
   ```python
   # Location: nsck-demo/python/cognitive_engine.py
   
   class CognitiveEngine:
       def __init__(self, use_dual_process=True):
           # ... existing init ...
           
           if use_dual_process:
               # Initialize dual-process architecture
               system1 = System1(self.snn, self.rule_learner.tenured_rules)
               system2 = System2(self.planner, self.causal_reasoner, self.world_model)
               self.dual_process = DualProcessCognition(
                   system1, system2, self.metacognition
               )
           else:
               self.dual_process = None
       
       def decide(self, state_hv, action_hvs, context):
           if self.dual_process:
               # Use dual-process decision making
               response = self.dual_process.decide(state_hv, context)
               return response['action']
           else:
               # Original decision logic
               return self._original_decide(state_hv, action_hvs)
   ```

3. **Testing**
   ```python
   # Location: nsck-demo/tests/test_dual_process.py
   
   def test_fast_path_for_easy_cases():
       """Test that System 1 handles easy cases quickly"""
       dual_proc = DualProcessCognition(system1, system2, metacognition)
       
       # Easy case: High confidence cached rule
       state = generate_easy_state()
       response = dual_proc.decide(state, context={'stakes': 'normal'})
       
       assert response['method'] == 'cached_rule'
       assert response['time'] < 0.1  # < 100ms
       assert dual_proc.s1_count == 1
       assert dual_proc.s2_count == 0
   
   def test_slow_path_for_hard_cases():
       """Test that System 2 handles hard cases"""
       dual_proc = DualProcessCognition(system1, system2, metacognition)
       
       # Hard case: Low confidence, high stakes
       state = generate_hard_state()
       response = dual_proc.decide(state, context={'stakes': 'high'})
       
       assert response['method'] == 'deliberative'
       assert 'reasoning' in response
       assert dual_proc.s2_count == 1
   
   def test_error_detection():
       """Test that S2 catches S1 errors"""
       dual_proc = DualProcessCognition(system1, system2, metacognition)
       
       # Create situation where S1 makes wrong decision
       state = generate_tricky_state()
       response = dual_proc.decide(state, context={})
       
       # S2 should have caught error
       assert dual_proc.errors_caught > 0
   
   def test_accuracy_improvement():
       """Benchmark: Dual-process should improve accuracy by 15%"""
       
       # Baseline: S1 only
       baseline_accuracy = benchmark_system1_only(test_set)
       
       # Dual-process
       dual_accuracy = benchmark_dual_process(test_set)
       
       improvement = (dual_accuracy - baseline_accuracy) / baseline_accuracy
       assert improvement > 0.12, f"Expected 12%+ improvement, got {improvement}"
   
   def test_speed_on_easy_cases():
       """Benchmark: 5× faster on easy cases"""
       
       easy_cases = generate_easy_test_set(1000)
       
       # S2 only (deliberative)
       s2_time = benchmark_system2_only(easy_cases)
       
       # Dual-process (should use S1 fast path)
       dual_time = benchmark_dual_process(easy_cases)
       
       speedup = s2_time / dual_time
       assert speedup > 4.0, f"Expected 4×+ speedup, got {speedup}×"
   ```

**Success Metrics:**
- Overall accuracy: +15% ✓
- Error detection: +40% ✓
- Speed on easy cases: 5× faster ✓
- S1 usage: 70-80% of cases ✓
- S2 usage: 20-30% (hard cases) ✓
- Tests: 12+ comprehensive tests ✓

**Deliverables:**
- [ ] `System1`, `System2`, `DualProcessCognition` classes
- [ ] Integration with `CognitiveEngine`
- [ ] 12+ comprehensive tests
- [ ] Performance benchmarks
- [ ] Configuration options
- [ ] Documentation

**Estimated Effort:** 8 days

---

## Phase 1 Summary

**Total Duration:** 4 weeks

**Total Tests:** 57+ new tests

**Expected Improvements:**
- Decision quality: +15-20%
- Throughput: +30-40%
- Continual learning: 62% less forgetting
- Transfer learning: +25% zero-shot
- System responsiveness: 5× faster (easy cases)

**Next Phase:** Medium priority improvements (Weeks 5-8)

---

## Phase 2: Targeted Enhancements (Weeks 5-8)

### 2.1 Context-Sensitive Competition
- Multi-criteria coalition competition
- +20% decision quality
- Estimated effort: 3 days

### 2.2 Robust Policy Optimization
- Safety-aware policy optimization
- +18% safety rule retention
- Estimated effort: 4 days

### 2.3 NeuroNAS
- Hardware-aware SNN optimization
- 84% energy reduction
- Estimated effort: 7 days

### 2.4 Category Theory Mapping
- Formal analogical mapping
- +35% mapping correctness
- Estimated effort: 6 days

### 2.5 Emotion-Guided Attention
- Emotional modulation of workspace
- +30% goal achievement
- Estimated effort: 3 days

**Phase 2 Total:** 23 days work (parallelizable)

---

## Phase 3: Optimizations (Weeks 9-12)

### 3.1 Vector Acceleration
- SIMD optimization
- 12× speedup
- Estimated effort: 3 days

### 3.2 Self-Synthesized Rehearsal
- Synthetic experience generation
- 90% memory savings
- Estimated effort: 4 days

### 3.3 Surrogate Gradient Enhancement
- Deeper SNN training
- Match ANN accuracy
- Estimated effort: 5 days

### 3.4 Adversarial Robustness
- Temporal encoding for robustness
- 2× adversarial resistance
- Estimated effort: 4 days

### 3.5 Sparse Projection Optimization
- Adaptive sparsity
- +8% accuracy
- Estimated effort: 2 days

**Phase 3 Total:** 18 days work

---

## Testing & Validation Strategy

### Unit Tests
- Each component: 5-15 tests
- Coverage target: 90%+
- Run continuously during development

### Integration Tests
- Inter-component interactions
- End-to-end cognitive cycles
- Run before each milestone

### Benchmarks
- Performance vs baseline
- Before/after comparisons
- Document all metrics

### Ablation Studies
- Test each component independently
- Measure contribution
- Identify critical components

---

## Documentation Updates

### Technical Documentation
- [ ] Update `docs/ARCHITECTURE.md` with new components
- [ ] Update `docs/FORMULAS_AND_PROOFS.md` with new algorithms
- [ ] Create `docs/DUAL_PROCESS_GUIDE.md`
- [ ] Create `docs/GPL_IMPLEMENTATION.md`
- [ ] Update API documentation

### User Documentation
- [ ] Update README with new capabilities
- [ ] Create migration guide
- [ ] Update configuration examples
- [ ] Add troubleshooting guide

### Research Documentation
- [ ] Document all benchmarks
- [ ] Create performance comparison report
- [ ] Document ablation studies
- [ ] Create visual diagrams

---

## Risk Mitigation

### Technical Risks
1. **Performance Regression**
   - Mitigation: Comprehensive benchmarking
   - Rollback plan: Feature flags for all new components

2. **Integration Complexity**
   - Mitigation: Phased integration, extensive testing
   - Rollback plan: Modular design allows isolation

3. **Resource Requirements**
   - Mitigation: Profile early, optimize as needed
   - Alternative: Make components optional

### Schedule Risks
1. **Underestimation**
   - Mitigation: 20% time buffer built in
   - Contingency: De-prioritize LOW priority items

2. **Dependency Delays**
   - Mitigation: Parallel work streams
   - Contingency: Stub interfaces, implement later

---

## Success Criteria

### Must Have (Phase 1)
- [ ] All HIGH priority improvements implemented
- [ ] 90%+ test coverage
- [ ] Performance improvements validated
- [ ] Documentation updated
- [ ] No regressions

### Should Have (Phase 2)
- [ ] All MEDIUM priority improvements implemented
- [ ] Comprehensive benchmarks
- [ ] Migration tools
- [ ] User guides

### Nice to Have (Phase 3)
- [ ] All LOW priority optimizations
- [ ] Advanced analytics
- [ ] Visual monitoring tools

---

## Conclusion

This implementation plan provides a systematic, research-backed approach to enhancing the NSCK cognitive architecture. Each improvement is grounded in peer-reviewed research with documented performance benefits. The phased approach ensures manageable complexity while delivering incremental value.

**Next Steps:**
1. Review and approve plan
2. Assign agents to tasks
3. Begin Phase 1 implementation
4. Track progress weekly
5. Adjust as needed based on results
