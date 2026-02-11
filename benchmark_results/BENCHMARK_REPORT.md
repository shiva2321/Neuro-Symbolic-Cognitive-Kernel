# NSCK Benchmark Report

**Generated**: 2026-02-06T14:09:35.025774

## Summary

- **Total**: 5
- **Passed**: 0
- **Failed**: 5

## Results

| Benchmark | Status | Duration | Key Metric |
|-----------|--------|----------|------------|
| Transfer Learning (Snake+Maze -> Pong) | [FAIL] | 7.72s | snake_score: 2.3000 |
| Zero-Shot Generalization (New Game) | [FAIL] | 5.05s | snake_score: 2.1000 |
| Continual Learning (Anti-Forgetting) | [FAIL] | 2.30s | baseline_snake_score: 3.6000 |
| Efficiency (Latency/Memory/Throughput) | [FAIL] | 0.64s | min_latency_ms: 0.2709 |
| Explainability (Completeness & Accuracy) | [FAIL] | 0.07s | explanation_rate: 1.0000 |

## Detailed Results

### Transfer Learning (Snake+Maze -> Pong)

**Status**: FAIL
**Duration**: 7.72s

**Metrics**:

- snake_score: 2.3000 (threshold: 5.0)
- maze_score: 0.0000 (threshold: 0.5)
- pong_score: 5.0000 (threshold: N/A)
- random_pong_score: 1.4000 (threshold: N/A)
- transfer_ratio: 3.5714 (threshold: 2.0)

### Zero-Shot Generalization (New Game)

**Status**: FAIL
**Duration**: 5.05s

**Metrics**:

- snake_score: 2.1000 (threshold: N/A)
- maze_score: 0.0000 (threshold: N/A)
- trained_score: 1.0500 (threshold: 3.0)
- collector_score: 2.5000 (threshold: N/A)
- random_score: 0.9000 (threshold: N/A)
- zeroshot_ratio: 2.7778 (threshold: 1.5)

### Continual Learning (Anti-Forgetting)

**Status**: FAIL
**Duration**: 2.30s

**Metrics**:

- baseline_snake_score: 3.6000 (threshold: N/A)
- task_b_score: 0.0000 (threshold: 0.3)
- after_snake_score: 1.6000 (threshold: N/A)
- forgetting_rate: 0.5556 (threshold: 0.1)
- retention_rate: 0.4444 (threshold: N/A)

### Efficiency (Latency/Memory/Throughput)

**Status**: FAIL
**Duration**: 0.64s

**Metrics**:

- min_latency_ms: 0.2709 (threshold: N/A)
- max_latency_ms: 15.9380 (threshold: N/A)
- mean_latency_ms: 5.9172 (threshold: N/A)
- p50_latency_ms: 6.0714 (threshold: N/A)
- p95_latency_ms: 13.8325 (threshold: 100.0)
- p99_latency_ms: 15.9380 (threshold: N/A)
- throughput_dps: 168.9069 (threshold: 10.0)
- current_memory_mb: 0.3962 (threshold: N/A)
- peak_memory_mb: 0.4080 (threshold: 500.0)

### Explainability (Completeness & Accuracy)

**Status**: FAIL
**Duration**: 0.07s

**Metrics**:

- explanation_rate: 1.0000 (threshold: 0.95)
- completeness_rate: 0.0000 (threshold: 0.8)
- action_mentioned_rate: 1.0000 (threshold: 0.9)
- source_rule_rate: 0.0000 (threshold: N/A)
- source_causal_rate: 0.0000 (threshold: N/A)
- source_episodic_rate: 0.0000 (threshold: N/A)
- source_transfer_rate: 0.0000 (threshold: N/A)
