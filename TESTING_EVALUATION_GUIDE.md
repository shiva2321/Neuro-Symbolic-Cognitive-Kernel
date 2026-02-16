# NSCK AI Model - Complete Testing & Evaluation Guide

## Overview

This directory contains a comprehensive testing and evaluation framework for the NSCK AI Model. The framework provides rigorous testing across multiple dimensions including cognitive capabilities, performance metrics, real-world scenarios, and continuous monitoring.

## Components

### 1. Comprehensive Benchmark System
**File**: `nsck_ai_model/comprehensive_benchmark.py`

A complete benchmarking suite that tests:
- **Multi-domain training**: 8 knowledge domains (physics, biology, history, technology, geography, arts, mathematics, causal reasoning)
- **Cognitive capabilities**: Context retention, counter-factual reasoning, cross-domain transfer, continuous coherence, knowledge integration, reasoning depth
- **Performance metrics**: Latency (tiny to very large queries), memory usage, storage efficiency, throughput
- **Image understanding**: Visual comprehension and description (when available)
- **Real-world scenarios**: Educational Q&A, technical support, general conversation, creative writing

**Usage**:
```bash
# Run the comprehensive benchmark
python -m nsck_ai_model.comprehensive_benchmark --output-dir ./benchmark_results

# Results will be saved to:
# - benchmark_results/FINAL_REPORT.md (human-readable report)
# - benchmark_results/FINAL_REPORT.json (structured data)
# - benchmark_results/training_summary.json
# - benchmark_results/cognitive_tests.json
# - benchmark_results/performance_benchmarks.json
# - benchmark_results/real_world_scenarios.json
```

### 2. Telemetry Monitoring System
**File**: `nsck_ai_model/telemetry_monitor.py`

Real-time monitoring and telemetry collection:
- **Query logging**: Every query with full trace, confidence, latency
- **Training sessions**: Learning curves, improvement rates, resource usage
- **Behavioral analysis**: Confidence trends, latency patterns, response quality
- **Anomaly detection**: Low confidence, high latency, short responses
- **Pattern recognition**: Query types, learning efficiency

**Usage**:
```python
from nsck_ai_model.ai_engine import NSCKAIEngine
from nsck_ai_model.telemetry_monitor import TelemetryMonitor, MonitoredEngine

# Create engine and monitor
engine = NSCKAIEngine()
monitor = TelemetryMonitor(output_dir="./telemetry_logs")
monitored_engine = MonitoredEngine(engine, monitor)

# Use monitored engine (automatically logs everything)
monitor.start_training_session("my_session")
monitored_engine.train_on_text("Some training text")
monitor.end_training_session()

result = monitored_engine.chat("What did you learn?")

# Generate reports
monitor.generate_monitoring_report()
monitor.export_query_log()
```

### 3. Complete Evaluation Suite
**File**: `run_complete_evaluation.py`

Integrated evaluation that runs everything:
- Comprehensive benchmark
- Telemetry monitoring
- Extended training (optional)
- Stress testing
- Final consolidated report

**Usage**:
```bash
# Standard evaluation (fast, ~10-15 seconds)
python run_complete_evaluation.py --output-dir ./complete_evaluation

# Extended evaluation (more training data, ~2-5 minutes)
python run_complete_evaluation.py --output-dir ./complete_evaluation --extended

# Results structure:
# complete_evaluation/
#   ├── COMPLETE_EVALUATION_REPORT.md
#   ├── COMPLETE_EVALUATION_REPORT.json
#   ├── benchmark/
#   │   ├── FINAL_REPORT.md
#   │   ├── training_summary.json
#   │   ├── cognitive_tests.json
#   │   └── performance_benchmarks.json
#   └── telemetry/
#       ├── monitoring_report.json
#       ├── query_log.json
#       └── standard_training.json
```

## Key Metrics Explained

### Cognitive Capabilities
- **Context Retention**: Ability to maintain information across conversation turns
- **Counter-factual Reasoning**: Ability to reason about hypothetical scenarios
- **Cross-Domain Transfer**: Applying knowledge from one domain to another
- **Continuous Coherence**: How long responses stay coherent
- **Knowledge Integration**: Combining multiple facts into unified understanding
- **Reasoning Depth**: Multi-step logical reasoning capability

### Performance Metrics
- **Latency**: Response time for various query sizes (tiny: <5ms, large: ~20ms)
- **Throughput**: Queries per second (typical: 1000+ QPS)
- **Memory**: RAM usage during training and inference
- **Storage**: Number of concepts, relations, episodes stored

### Quality Metrics
- **Confidence**: Model's self-assessed confidence (0.0-1.0)
- **Response Length**: Average characters in responses
- **Source Diversity**: How many different sources contribute to responses
- **Anomaly Rate**: Percentage of queries with unusual behavior

## Current Results Summary

Based on the comprehensive evaluation:

### Strengths ✓
1. **Fast training**: Processes samples quickly (<50ms average)
2. **Memory efficient**: Uses VSA-based representations (10,240-bit hypervectors)
3. **Full traceability**: Every response includes 11-stage thought trace
4. **Novel architecture**: No neural networks, transformers, or matrix multiplication
5. **Excellent throughput**: 6500+ queries per second
6. **High reliability**: <5% anomaly rate

### Weaknesses ⚠️
1. **Limited training**: Would benefit from more diverse training data
2. **Cognitive limitations**: Failed context retention and counter-factual reasoning tests
3. **Response generation**: Currently uses simple sentence retrieval and assembly
4. **Multi-modal**: Image understanding needs more development

### Performance Characteristics
- **Training Speed**: ~50ms per sample
- **Query Latency**: 2-5ms average, up to 30ms for very large queries
- **Throughput**: 6500+ QPS (burst), 280+ QPS (sustained conversation)
- **Memory Efficiency**: Minimal overhead, ~0MB growth during 100 query burst
- **Cognitive Pass Rate**: 66.7% (4/6 tests passed)
- **Real-World Success**: 100% on standard scenarios

## Evaluation Workflow

### Quick Test (5 minutes)
```bash
# 1. Run basic benchmark
python -m nsck_ai_model.comprehensive_benchmark --output-dir ./quick_test

# 2. Review results
cat quick_test/FINAL_REPORT.md
```

### Standard Evaluation (15 minutes)
```bash
# Run complete evaluation
python run_complete_evaluation.py --output-dir ./standard_eval

# Review summary
cat standard_eval/COMPLETE_EVALUATION_REPORT.md

# Explore detailed results
cat standard_eval/benchmark/FINAL_REPORT.md
cat standard_eval/telemetry/monitoring_report.json
```

### Extended Evaluation (1-2 hours)
```bash
# Run with extended training data
python run_complete_evaluation.py --output-dir ./extended_eval --extended

# This includes:
# - 40+ training samples across multiple domains
# - Comprehensive cognitive testing
# - Long-running stress tests
# - Detailed telemetry analysis
```

## Interpreting Results

### Pass/Fail Criteria

**Cognitive Tests**: Pass if score ≥ 50%
- Context Retention: Correctly answers ≥50% of context questions
- Counter-factual: Recognizes hypothetical scenarios
- Cross-domain: Applies knowledge across domains (confidence ≥30%)
- Coherence: Maintains coherence for ≥60% of conversation
- Integration: Combines ≥40% of related facts
- Reasoning: Correctly performs ≥50% of logical inferences

**Performance Tests**: Pass if meets thresholds
- Latency: <50ms for medium queries
- Throughput: >10 QPS sustained
- Memory: <100MB growth over 1000 queries

### Good vs Excellent

**Good System**:
- 60-80% cognitive pass rate
- 50-100 QPS throughput
- <10% anomaly rate
- 50-70% average confidence

**Excellent System**:
- >80% cognitive pass rate
- >100 QPS throughput
- <5% anomaly rate
- >70% average confidence

## Continuous Monitoring

For production deployments, use the telemetry monitor:

```python
# Production setup
monitor = TelemetryMonitor(output_dir="/var/log/nsck_ai/telemetry")
monitored_engine = MonitoredEngine(engine, monitor)

# Periodic reporting (e.g., hourly cron job)
def generate_hourly_report():
    report = monitor.generate_monitoring_report()
    # Send alerts if anomaly rate > threshold
    if report['anomalies']['total_anomalies'] > 100:
        send_alert("High anomaly rate detected")
```

## Troubleshooting

### Low Cognitive Scores
- **Cause**: Insufficient training data
- **Solution**: Run with `--extended` flag or train on more diverse data
- **Check**: `training_summary.json` for samples processed

### High Latency
- **Cause**: Large query size or system load
- **Solution**: Normal for queries >200 chars; profile if persistent
- **Check**: `performance_benchmarks.json` for latency breakdown

### High Anomaly Rate
- **Cause**: Queries outside training distribution
- **Solution**: Expand training domains, review anomaly types
- **Check**: `monitoring_report.json` -> anomalies -> by_type

### Memory Growth
- **Cause**: Episode accumulation without cleanup
- **Solution**: Normal for VSA; consider episode pruning for very long sessions
- **Check**: `performance_benchmarks.json` -> memory section

## Next Steps

After running evaluation:

1. **Review Reports**: Read FINAL_REPORT.md for honest assessment
2. **Identify Gaps**: Check weaknesses section for improvement areas
3. **Expand Training**: Add more data in weak domains
4. **Iterate**: Re-run evaluation to measure improvement
5. **Monitor Production**: Use telemetry for ongoing assessment

## Dependencies

```bash
# Install requirements
pip install numpy psutil networkx flask

# Or use requirements.txt
pip install -r requirements.txt
```

## License & Citation

Part of the NSCK (Neural-Symbolic Cognitive Kernel) project.
See main repository README for license and citation information.

---

**Questions or Issues?**
- Check existing tests: `nsck_ai_model/tests/`
- Review architecture: `nsck_ai_model/ARCHITECTURE.md`
- See examples: `nsck_ai_model/train.py`, `nsck_ai_model/dashboard.py`
