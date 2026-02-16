# NSCK AI Model - Quick Testing Reference

## One-Command Tests

### Quick Health Check (5 seconds)
```bash
python -c "
from nsck_ai_model.ai_engine import NSCKAIEngine
engine = NSCKAIEngine()
engine.train_on_text('Water is H2O.')
result = engine.chat('What is water?')
print('✓ System operational')
print(f'Response: {result[\"response\"]}')
print(f'Confidence: {result[\"confidence\"]:.2%}')
"
```

### Complete Benchmark (10 seconds)
```bash
python -m nsck_ai_model.comprehensive_benchmark --output-dir ./quick_benchmark
cat quick_benchmark/FINAL_REPORT.md
```

### Full Evaluation (15 seconds)
```bash
python run_complete_evaluation.py --output-dir ./eval_results
cat eval_results/COMPLETE_EVALUATION_REPORT.md
```

### Extended Evaluation (2-5 minutes)
```bash
python run_complete_evaluation.py --output-dir ./eval_extended --extended
```

## Test What You Want

### Test Cognitive Capabilities
```bash
python -c "
from nsck_ai_model.comprehensive_benchmark import ComprehensiveBenchmark
b = ComprehensiveBenchmark()
b.run_comprehensive_training()
results = b.test_cognitive_capabilities()
print(f'Pass Rate: {results[\"pass_rate\"]:.0%}')
"
```

### Test Performance Only
```bash
python -c "
from nsck_ai_model.comprehensive_benchmark import ComprehensiveBenchmark
b = ComprehensiveBenchmark()
b.engine = __import__('nsck_ai_model.ai_engine', fromlist=['NSCKAIEngine']).NSCKAIEngine()
results = b.benchmark_performance()
print(f'Latency: {results[\"latency\"][\"medium\"][\"avg_ms\"]:.2f}ms')
print(f'Throughput: {results[\"throughput\"][\"queries_per_second\"]:.1f} QPS')
"
```

### Monitor Training
```bash
python -c "
from nsck_ai_model.ai_engine import NSCKAIEngine
from nsck_ai_model.telemetry_monitor import TelemetryMonitor, MonitoredEngine

engine = NSCKAIEngine()
monitor = TelemetryMonitor('./my_logs')
monitored = MonitoredEngine(engine, monitor)

monitor.start_training_session()
for text in ['fact 1', 'fact 2', 'fact 3']:
    monitored.train_on_text(text)
monitor.end_training_session()

report = monitor.generate_monitoring_report()
print(f'Trained {report[\"overall_stats\"][\"total_training_samples\"]} samples')
"
```

## Quick Metrics

### Current System Stats
```bash
python -c "
from nsck_ai_model.ai_engine import NSCKAIEngine
import json
engine = NSCKAIEngine()
stats = engine.get_system_stats()
print(json.dumps({
    'concepts': stats['knowledge']['total_concepts'],
    'relations': stats['knowledge']['total_relations'],
    'episodes': stats['knowledge']['total_episodes']
}, indent=2))
"
```

### Stress Test
```bash
python -c "
from nsck_ai_model.ai_engine import NSCKAIEngine
import time
engine = NSCKAIEngine()
engine.train_on_text('AI is awesome.')

# Burst test
start = time.time()
for _ in range(100):
    engine.chat('What is AI?')
qps = 100 / (time.time() - start)
print(f'Throughput: {qps:.0f} QPS')
"
```

## Common Scenarios

### Train and Test
```bash
python -c "
from nsck_ai_model.ai_engine import NSCKAIEngine

engine = NSCKAIEngine()

# Train
facts = [
    'The sky is blue.',
    'Grass is green.',
    'The sun is bright.'
]
for fact in facts:
    engine.train_on_text(fact)

# Test
queries = ['What color is the sky?', 'Tell me about grass.']
for q in queries:
    r = engine.chat(q)
    print(f'Q: {q}')
    print(f'A: {r[\"response\"]} (confidence: {r[\"confidence\"]:.0%})')
    print()
"
```

### Continuous Monitoring
```bash
python -c "
from nsck_ai_model.telemetry_monitor import TelemetryMonitor, MonitoredEngine
from nsck_ai_model.ai_engine import NSCKAIEngine

monitor = TelemetryMonitor('./monitor_demo')
engine = MonitoredEngine(NSCKAIEngine(), monitor)

# Use normally
engine.train_on_text('Some fact.')
for _ in range(10):
    engine.chat('Question?')

# Get stats
stats = monitor.get_real_time_stats()
print(f'Queries: {stats[\"total_queries\"]}')
print(f'Avg Confidence: {stats[\"avg_confidence\"]:.0%}')
print(f'Avg Latency: {stats[\"avg_latency_ms\"]:.1f}ms')
"
```

## File Locations

After running tests, find results at:

```
./benchmark_results/           # From comprehensive_benchmark.py
  ├── FINAL_REPORT.md          # Main report
  ├── training_summary.json     # Training metrics
  ├── cognitive_tests.json      # Test results
  └── performance_benchmarks.json

./complete_evaluation/          # From run_complete_evaluation.py
  ├── COMPLETE_EVALUATION_REPORT.md
  ├── benchmark/               # Benchmark sub-results
  └── telemetry/               # Monitoring data

./telemetry_logs/              # From telemetry_monitor.py
  ├── monitoring_report.json   # Overall report
  ├── query_log.json           # All queries
  └── [session_id].json        # Training sessions
```

## Interpreting Results

### Good Results
- Cognitive pass rate: >60%
- Average confidence: >60%
- Average latency: <20ms
- Anomaly rate: <10%
- Throughput: >50 QPS

### Excellent Results
- Cognitive pass rate: >80%
- Average confidence: >70%
- Average latency: <10ms
- Anomaly rate: <5%
- Throughput: >100 QPS

### Current System
- Cognitive: 66.7% ✅
- Confidence: 63.9% ✅
- Latency: 3.4ms ✅
- Anomalies: 0% ✅
- Throughput: 6,574 QPS ✅

## Troubleshooting

### Import Errors
```bash
pip install numpy networkx flask psutil
```

### Slow Performance
```bash
# Check if using Python fallback (slower)
python -c "import sys; sys.path.insert(0, 'nsck-demo'); from python.core.vsa.hypervec_shim import HyperVector; print('Backend:', 'Rust' if hasattr(HyperVector, 'from_rust') else 'Python')"
```

### Memory Issues
```bash
# Monitor memory usage
python -c "
import psutil
from nsck_ai_model.ai_engine import NSCKAIEngine
import gc

process = psutil.Process()
before = process.memory_info().rss / 1024 / 1024

engine = NSCKAIEngine()
for _ in range(100):
    engine.train_on_text('Sample text.')
gc.collect()

after = process.memory_info().rss / 1024 / 1024
print(f'Memory used: {after - before:.1f} MB')
"
```

## Next Steps

1. **Review Results**: Read `COMPREHENSIVE_TESTING_REPORT.md`
2. **Identify Issues**: Check weaknesses in reports
3. **Add Training**: Provide more diverse data
4. **Re-test**: Run evaluation again
5. **Compare**: Track improvement over time

---

For complete documentation, see:
- `COMPREHENSIVE_TESTING_REPORT.md` - Full analysis
- `TESTING_EVALUATION_GUIDE.md` - Detailed guide
- `benchmark_results/FINAL_REPORT.md` - Benchmark results
- `complete_evaluation/COMPLETE_EVALUATION_REPORT.md` - Full evaluation
