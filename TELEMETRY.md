# NSCK Telemetry & Monitoring

Comprehensive guide to monitoring, logging, and telemetry systems across NSCK projects.

---

## Overview

The NSCK system provides extensive telemetry for monitoring cognitive processes, performance metrics, and system health in real-time.

### Available Monitoring Systems

1. **Unified Dashboard** - Web-based real-time monitoring
2. **Telemetry Monitor** - AI Model query tracking and analysis
3. **Structured Logging** - Multi-level categorized logging
4. **Performance Metrics** - Continuous performance tracking
5. **Anomaly Detection** - Automatic issue identification

---

## Unified Dashboard Telemetry

### Access
```bash
python launch_dashboard.py
# Open: http://localhost:5000
```

### Real-Time Metrics

**System Metrics:**
- Memory usage (current and historical)
- CPU utilization
- Query throughput (QPS)
- Response latency

**Cognitive Metrics:**
- Active concepts count
- Relations extracted
- Episodes stored
- Emotion state
- Attention focus
- Working memory load

**Game Learning Metrics:**
- Score progression
- Win rate over time
- Learning curve visualization
- Policy size growth

### Structured Logging

**Log Categories:**
```
SYSTEM     - Startup, shutdown, configuration
COGNITIVE  - Memory operations, reasoning steps
GAME       - Game interactions, actions, rewards
LEARNING   - Knowledge acquisition, rule learning
ERROR      - Exceptions, failures, warnings
```

**Severity Levels:**
```
DEBUG      - Detailed debugging information
INFO       - General informational messages
WARNING    - Warning messages (non-critical)
ERROR      - Error messages (critical)
CRITICAL   - Critical system failures
```

**Log Format:**
```json
{
  "timestamp": "2026-02-16T02:18:31.755Z",
  "category": "COGNITIVE",
  "severity": "INFO",
  "message": "Concept 'gravity' stored with similarity 0.94",
  "metadata": {
    "concept": "gravity",
    "similarity": 0.94,
    "module": "semantic_memory"
  }
}
```

### Export Capabilities

**Export Formats:**

1. **JSON** - Machine-readable structured data
   ```bash
   curl http://localhost:5000/api/logs/export/json > session.json
   ```

2. **CSV** - Spreadsheet-compatible format
   ```bash
   curl http://localhost:5000/api/logs/export/csv > session.csv
   ```

3. **TXT** - Human-readable report
   ```bash
   curl http://localhost:5000/api/logs/export/txt > session.txt
   ```

**Session Persistence:**
- Logs automatically saved to: `/logs/nsck_session_YYYYMMDD_HHMMSS.log`
- Rotated daily
- Retained for 30 days (configurable)

---

## AI Model Telemetry Monitor

### Overview

The `telemetry_monitor.py` system provides comprehensive tracking for the NSCK AI Model.

### Usage

```python
from nsck_ai_model.telemetry_monitor import TelemetryMonitor

# Initialize
monitor = TelemetryMonitor()

# Track query
monitor.log_query(
    query="What is gravity?",
    response="Gravity is a force...",
    confidence=0.89,
    latency_ms=3.4,
    thought_trace={...}
)

# Generate reports
summary = monitor.generate_summary()
print(summary)
```

### Tracked Metrics

**Query Metrics:**
- Query text and response
- Confidence score
- Processing latency (per stage)
- Thought trace (11 stages)
- Concepts activated
- Relations traversed

**Training Metrics:**
- Samples processed
- Concepts learned
- Relations extracted
- Training time
- Learning curve data

**Behavioral Patterns:**
- Query type distribution
- Topic frequency
- Confidence trends
- Response length distribution

**Anomaly Detection:**
```python
# Low confidence queries
anomalies = monitor.detect_anomalies(
    low_confidence_threshold=0.5,
    high_latency_threshold=50.0,
    short_response_threshold=20
)
```

### Real-Time Monitoring

```python
# Start continuous monitoring
monitor.start_monitoring(interval=5.0)  # Check every 5 seconds

# Get current status
status = monitor.get_realtime_status()
print(f"QPS: {status['qps']}")
print(f"Avg Latency: {status['avg_latency_ms']}ms")
print(f"Memory: {status['memory_mb']}MB")
```

### Report Generation

```python
# Generate comprehensive report
report = monitor.generate_report(
    include_queries=True,
    include_training=True,
    include_anomalies=True,
    include_visualizations=True
)

# Save report
with open('telemetry_report.json', 'w') as f:
    json.dump(report, f, indent=2)
```

---

## Performance Tracking

### Metrics Collected

**Hypervector Operations:**
```python
{
  "operation": "bind",
  "time_us": 1.8,
  "backend": "rust",
  "vector_size": 10240
}
```

**Memory Operations:**
```python
{
  "operation": "store_concept",
  "time_ms": 0.12,
  "memory_kb": 10,
  "concept_count": 476
}
```

**Query Processing:**
```python
{
  "query_id": "q_12345",
  "total_time_ms": 3.43,
  "stages": {
    "encoding": 0.45,
    "retrieval": 0.82,
    "reasoning": 0.51,
    "composition": 1.23
  }
}
```

### Continuous Monitoring

**System Health:**
- Memory usage trends
- CPU utilization patterns
- Disk I/O
- Network activity (for APIs)

**Performance Degradation Detection:**
- Latency increase alerts
- Memory leak detection
- Throughput drop warnings
- Error rate spikes

---

## Dashboard API Endpoints

### Metrics Endpoints

**GET /api/stats**
```json
{
  "concepts": 476,
  "relations": 1290,
  "episodes": 3421,
  "queries_processed": 10543
}
```

**GET /api/metrics/performance**
```json
{
  "avg_latency_ms": 3.43,
  "p95_latency_ms": 8.2,
  "p99_latency_ms": 12.8,
  "qps": 6574,
  "uptime_hours": 24.3
}
```

**GET /api/metrics/cognitive**
```json
{
  "attention_focus": "gravity_concepts",
  "emotion_state": {"valence": 0.6, "arousal": 0.4},
  "working_memory_load": 0.45,
  "active_reasoning": true
}
```

### Logging Endpoints

**GET /api/logs**
```json
{
  "logs": [
    {
      "timestamp": "2026-02-16T02:18:31.755Z",
      "category": "COGNITIVE",
      "severity": "INFO",
      "message": "..."
    }
  ],
  "count": 1543,
  "page": 1
}
```

**GET /api/logs/export/{format}**
- Formats: `json`, `csv`, `txt`
- Returns: Complete session logs in specified format

---

## Alerting System

### Alert Types

**Performance Alerts:**
- Latency > 50ms (WARNING)
- Latency > 100ms (ERROR)
- Memory growth > 10% (WARNING)
- QPS drop > 50% (WARNING)

**Quality Alerts:**
- Confidence < 0.3 (WARNING)
- Response length < 10 chars (WARNING)
- Error rate > 5% (ERROR)

**System Alerts:**
- Memory usage > 80% (WARNING)
- Memory usage > 95% (CRITICAL)
- Disk space < 1GB (WARNING)
- CPU usage > 90% (WARNING)

### Alert Configuration

```python
# Configure alerts
monitor.configure_alerts(
    latency_threshold_ms=50.0,
    confidence_threshold=0.3,
    memory_threshold_pct=80.0,
    alert_callback=send_alert_email
)
```

---

## Visualization

### Real-Time Dashboards

**Unified Dashboard Features:**
- Live metrics charts
- Learning curve visualization
- Memory usage graphs
- Cognitive state displays
- System health indicators

**Chart Types:**
1. **Line Charts** - Time-series data (latency, throughput)
2. **Bar Charts** - Categorical data (query types, topics)
3. **Heatmaps** - Activity patterns over time
4. **Gauge Charts** - Current state metrics (CPU, memory)
5. **Network Graphs** - Concept relationships

### Export Visualizations

```bash
# Generate performance chart
python -c "
from nsck_ai_model.telemetry_monitor import TelemetryMonitor
monitor = TelemetryMonitor()
monitor.load_session('session_20260216.log')
monitor.plot_performance(save_path='performance.png')
"
```

---

## Analysis Tools

### Query Analysis

```python
# Analyze query patterns
patterns = monitor.analyze_query_patterns()
print(f"Most common topics: {patterns['top_topics']}")
print(f"Avg confidence: {patterns['avg_confidence']}")
print(f"Peak QPS time: {patterns['peak_qps_time']}")
```

### Learning Analysis

```python
# Analyze learning progress
progress = monitor.analyze_learning_progress()
print(f"Concepts learned: {progress['concepts_learned']}")
print(f"Learning rate: {progress['learning_rate']}")
print(f"Convergence estimate: {progress['convergence_eta']}")
```

### Anomaly Analysis

```python
# Detailed anomaly analysis
anomalies = monitor.analyze_anomalies()
for anomaly in anomalies:
    print(f"Type: {anomaly['type']}")
    print(f"Severity: {anomaly['severity']}")
    print(f"Affected queries: {anomaly['query_count']}")
    print(f"Recommendation: {anomaly['recommendation']}")
```

---

## Integration Examples

### Custom Monitoring Script

```python
#!/usr/bin/env python3
"""Custom monitoring script for production deployment."""

import time
from nsck_ai_model.telemetry_monitor import TelemetryMonitor

def main():
    monitor = TelemetryMonitor()
    
    # Start monitoring
    while True:
        # Get current metrics
        status = monitor.get_realtime_status()
        
        # Check thresholds
        if status['avg_latency_ms'] > 50:
            alert("High latency detected", status)
        
        if status['confidence'] < 0.5:
            alert("Low confidence queries", status)
        
        # Wait before next check
        time.sleep(5)

def alert(message, data):
    """Send alert (implement your notification system)."""
    print(f"ALERT: {message}")
    print(f"Data: {data}")

if __name__ == "__main__":
    main()
```

### Prometheus Integration

```python
from prometheus_client import Gauge, Counter, Histogram
import time

# Define metrics
query_latency = Histogram('nsck_query_latency_seconds', 
                          'Query processing latency')
query_count = Counter('nsck_queries_total', 
                     'Total queries processed')
confidence_score = Gauge('nsck_confidence_score',
                        'Average confidence score')

# Instrument your code
with query_latency.time():
    response = engine.process_query(query)
    
query_count.inc()
confidence_score.set(response['confidence'])
```

---

## Best Practices

### Monitoring Strategy

1. **Always Enable Telemetry** in production
2. **Set Appropriate Alert Thresholds** based on your SLAs
3. **Regular Health Checks** (every 5-10 seconds)
4. **Retain Logs** for at least 30 days
5. **Export Critical Sessions** for analysis

### Performance Optimization

1. **Monitor Latency Trends** - Identify performance degradation early
2. **Track Memory Growth** - Detect leaks before they impact service
3. **Analyze Query Patterns** - Optimize for common queries
4. **Review Anomalies Daily** - Fix issues proactively

### Privacy & Security

1. **Sanitize Logs** - Remove PII before storage
2. **Encrypt Session Data** - Protect sensitive information
3. **Access Control** - Limit telemetry access to authorized personnel
4. **Compliance** - Follow GDPR, HIPAA, or other regulations

---

## Troubleshooting

### High Latency

```python
# Diagnose high latency
trace = monitor.get_slow_queries()
for query in trace:
    print(f"Query: {query['text']}")
    print(f"Total time: {query['latency_ms']}ms")
    print(f"Bottleneck: {query['slowest_stage']}")
```

### Low Confidence

```python
# Analyze low confidence queries
low_conf = monitor.get_low_confidence_queries()
for query in low_conf:
    print(f"Query: {query['text']}")
    print(f"Confidence: {query['confidence']}")
    print(f"Reason: {query['analysis']['reason']}")
    print(f"Suggestion: {query['analysis']['suggestion']}")
```

### Memory Issues

```python
# Check memory usage
memory_report = monitor.get_memory_report()
print(f"Current: {memory_report['current_mb']}MB")
print(f"Peak: {memory_report['peak_mb']}MB")
print(f"Growth rate: {memory_report['growth_rate_mb_per_hour']}")
print(f"Top consumers: {memory_report['top_consumers']}")
```

---

## References

- [Dashboard Guide](docs/DASHBOARD_GUIDE.md) - Complete dashboard documentation
- [Performance Metrics](PERFORMANCE.md) - Detailed performance benchmarks
- [Testing Documentation](TESTING.md) - Test infrastructure
- [Architecture](docs/ARCHITECTURE.md) - System architecture

---

**Last Updated:** February 16, 2026  
**Version:** 1.0  
**Status:** Production Ready

