#!/usr/bin/env python3
"""
Advanced Telemetry and Monitoring System for NSCK AI Model
===========================================================

This module provides real-time monitoring, telemetry collection, and
analysis of the NSCK AI model during training and inference.

Features:
- Real-time performance monitoring
- Detailed reasoning trace analysis
- Memory and resource profiling
- Learning curve tracking
- Behavioral pattern analysis
- Anomaly detection
"""

import os
import sys
import time
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict, deque
import numpy as np

# Ensure nsck_ai_model is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class QueryTelemetry:
    """Detailed telemetry for a single query."""
    timestamp: str
    query: str
    response: str
    confidence: float
    latency_ms: float
    
    # Cognitive trace
    trace_steps: List[Dict[str, Any]] = field(default_factory=list)
    
    # Resource usage
    memory_usage_mb: float = 0.0
    cpu_percent: float = 0.0
    
    # Cognitive metrics
    concepts_activated: int = 0
    episodes_recalled: int = 0
    causal_chains_fired: int = 0
    emotion_state: str = ""
    novelty_score: float = 0.0
    
    # Response quality
    response_length: int = 0
    unique_tokens: int = 0
    source_diversity: float = 0.0  # How diverse the sources are


@dataclass
class TrainingSession:
    """Telemetry for a training session."""
    session_id: str
    start_time: str
    end_time: Optional[str] = None
    
    # Training data
    samples_processed: int = 0
    total_training_time_s: float = 0.0
    
    # Learning metrics
    concepts_learned: List[int] = field(default_factory=list)  # Per batch
    relations_learned: List[int] = field(default_factory=list)
    learning_rate: List[float] = field(default_factory=list)  # Concepts per second
    
    # Resource metrics
    memory_snapshots: List[float] = field(default_factory=list)
    cpu_snapshots: List[float] = field(default_factory=list)


class TelemetryMonitor:
    """
    Real-time monitoring and telemetry collection system.
    """
    
    def __init__(self, output_dir: str = "./telemetry_logs"):
        """Initialize telemetry monitor."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Telemetry storage
        self.query_log: deque = deque(maxlen=10000)
        self.training_sessions: List[TrainingSession] = []
        self.current_session: Optional[TrainingSession] = None
        
        # Statistics
        self.stats = {
            "total_queries": 0,
            "total_training_samples": 0,
            "avg_confidence": 0.0,
            "avg_latency_ms": 0.0,
            "concept_growth": [],
            "relation_growth": [],
        }
        
        # Behavioral patterns
        self.query_patterns: Dict[str, int] = defaultdict(int)
        self.confidence_history: deque = deque(maxlen=1000)
        self.latency_history: deque = deque(maxlen=1000)
        
        # Anomalies
        self.anomalies: List[Dict[str, Any]] = []
        
        # Setup logging
        self.logger = logging.getLogger("telemetry_monitor")
        
        # Session file
        self.session_file = os.path.join(output_dir, "current_session.json")
        
    def start_training_session(self, session_id: str = None) -> str:
        """Start a new training session."""
        if session_id is None:
            session_id = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_session = TrainingSession(
            session_id=session_id,
            start_time=datetime.now().isoformat(),
        )
        
        self.logger.info(f"Started training session: {session_id}")
        return session_id
    
    def end_training_session(self):
        """End the current training session."""
        if self.current_session:
            self.current_session.end_time = datetime.now().isoformat()
            self.training_sessions.append(self.current_session)
            
            # Save session data
            self._save_session(self.current_session)
            
            self.logger.info(f"Ended training session: {self.current_session.session_id}")
            self.current_session = None
    
    def log_training_step(self, 
                         samples_processed: int,
                         concepts_added: int,
                         relations_added: int,
                         elapsed_s: float):
        """Log a training step."""
        if not self.current_session:
            self.start_training_session()
        
        session = self.current_session
        session.samples_processed += samples_processed
        session.total_training_time_s += elapsed_s
        session.concepts_learned.append(concepts_added)
        session.relations_learned.append(relations_added)
        
        # Calculate learning rate (concepts per second)
        learning_rate = concepts_added / elapsed_s if elapsed_s > 0 else 0
        session.learning_rate.append(learning_rate)
        
        # Update global stats
        self.stats["total_training_samples"] += samples_processed
        self.stats["concept_growth"].append(concepts_added)
        self.stats["relation_growth"].append(relations_added)
    
    def log_query(self, 
                  query: str, 
                  result: Dict[str, Any],
                  trace: Optional[Dict[str, Any]] = None) -> QueryTelemetry:
        """Log a query and its result."""
        # Extract telemetry
        telemetry = QueryTelemetry(
            timestamp=datetime.now().isoformat(),
            query=query,
            response=result.get('response', ''),
            confidence=result.get('confidence', 0.0),
            latency_ms=result.get('latency_ms', 0.0),
            response_length=len(result.get('response', '')),
            emotion_state=result.get('emotion', {}).get('emotion', 'neutral'),
        )
        
        # Add trace information if available
        if trace:
            telemetry.trace_steps = trace.get('steps', [])
        
        # Add to log
        self.query_log.append(telemetry)
        
        # Update statistics
        self.stats["total_queries"] += 1
        self.confidence_history.append(telemetry.confidence)
        self.latency_history.append(telemetry.latency_ms)
        
        # Update averages
        self.stats["avg_confidence"] = np.mean(list(self.confidence_history))
        self.stats["avg_latency_ms"] = np.mean(list(self.latency_history))
        
        # Detect anomalies
        self._detect_anomalies(telemetry)
        
        # Pattern analysis
        self._analyze_query_pattern(query)
        
        return telemetry
    
    def _detect_anomalies(self, telemetry: QueryTelemetry):
        """Detect anomalous behavior."""
        anomalies = []
        
        # Check for abnormally low confidence
        if telemetry.confidence < 0.2 and len(telemetry.response) > 10:
            anomalies.append({
                "type": "low_confidence",
                "confidence": telemetry.confidence,
                "query": telemetry.query,
                "timestamp": telemetry.timestamp,
            })
        
        # Check for abnormally high latency
        if len(self.latency_history) > 10:
            avg_latency = np.mean(list(self.latency_history))
            std_latency = np.std(list(self.latency_history))
            
            if telemetry.latency_ms > avg_latency + 3 * std_latency:
                anomalies.append({
                    "type": "high_latency",
                    "latency_ms": telemetry.latency_ms,
                    "avg_latency_ms": avg_latency,
                    "query": telemetry.query,
                    "timestamp": telemetry.timestamp,
                })
        
        # Check for empty or very short responses
        if len(telemetry.response) < 5 and len(telemetry.query) > 10:
            anomalies.append({
                "type": "short_response",
                "response_length": len(telemetry.response),
                "query_length": len(telemetry.query),
                "query": telemetry.query,
                "timestamp": telemetry.timestamp,
            })
        
        if anomalies:
            self.anomalies.extend(anomalies)
            self.logger.warning(f"Detected {len(anomalies)} anomalies in query")
    
    def _analyze_query_pattern(self, query: str):
        """Analyze query patterns for insights."""
        # Simple categorization
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["what", "which", "who", "where", "when"]):
            self.query_patterns["factual"] += 1
        elif any(word in query_lower for word in ["how", "why", "explain"]):
            self.query_patterns["explanatory"] += 1
        elif "?" in query:
            self.query_patterns["question"] += 1
        else:
            self.query_patterns["statement"] += 1
    
    def get_real_time_stats(self) -> Dict[str, Any]:
        """Get real-time statistics."""
        recent_queries = list(self.query_log)[-100:] if self.query_log else []
        
        stats = {
            "total_queries": self.stats["total_queries"],
            "total_training_samples": self.stats["total_training_samples"],
            "avg_confidence": self.stats["avg_confidence"],
            "avg_latency_ms": self.stats["avg_latency_ms"],
            "recent_confidence": np.mean([q.confidence for q in recent_queries]) if recent_queries else 0,
            "recent_latency_ms": np.mean([q.latency_ms for q in recent_queries]) if recent_queries else 0,
            "query_patterns": dict(self.query_patterns),
            "anomaly_count": len(self.anomalies),
        }
        
        # Add training session stats
        if self.current_session:
            stats["current_session"] = {
                "samples_processed": self.current_session.samples_processed,
                "training_time_s": self.current_session.total_training_time_s,
                "avg_learning_rate": np.mean(self.current_session.learning_rate) if self.current_session.learning_rate else 0,
            }
        
        return stats
    
    def get_behavioral_analysis(self) -> Dict[str, Any]:
        """Analyze behavioral patterns."""
        recent_queries = list(self.query_log)[-1000:] if self.query_log else []
        
        if not recent_queries:
            return {"status": "insufficient_data"}
        
        # Confidence trends
        confidences = [q.confidence for q in recent_queries]
        confidence_trend = "improving" if len(confidences) > 1 and confidences[-1] > confidences[0] else "declining"
        
        # Latency trends
        latencies = [q.latency_ms for q in recent_queries]
        latency_trend = "improving" if len(latencies) > 1 and latencies[-1] < latencies[0] else "stable"
        
        # Response quality
        response_lengths = [q.response_length for q in recent_queries]
        
        return {
            "confidence": {
                "mean": np.mean(confidences),
                "std": np.std(confidences),
                "min": np.min(confidences),
                "max": np.max(confidences),
                "trend": confidence_trend,
            },
            "latency": {
                "mean": np.mean(latencies),
                "std": np.std(latencies),
                "min": np.min(latencies),
                "max": np.max(latencies),
                "trend": latency_trend,
            },
            "response_quality": {
                "avg_length": np.mean(response_lengths),
                "min_length": np.min(response_lengths),
                "max_length": np.max(response_lengths),
            },
            "query_distribution": dict(self.query_patterns),
        }
    
    def get_learning_curve_analysis(self) -> Dict[str, Any]:
        """Analyze learning curves from training sessions."""
        if not self.training_sessions:
            return {"status": "no_training_data"}
        
        all_learning_rates = []
        all_concepts = []
        
        for session in self.training_sessions:
            all_learning_rates.extend(session.learning_rate)
            all_concepts.extend(session.concepts_learned)
        
        return {
            "total_sessions": len(self.training_sessions),
            "learning_rate": {
                "mean": np.mean(all_learning_rates) if all_learning_rates else 0,
                "std": np.std(all_learning_rates) if all_learning_rates else 0,
                "trend": all_learning_rates[-10:] if len(all_learning_rates) >= 10 else all_learning_rates,
            },
            "concepts_per_session": {
                "mean": np.mean(all_concepts) if all_concepts else 0,
                "total": sum(all_concepts),
            },
        }
    
    def get_anomaly_report(self) -> Dict[str, Any]:
        """Get anomaly detection report."""
        if not self.anomalies:
            return {"status": "no_anomalies", "count": 0}
        
        # Group anomalies by type
        by_type = defaultdict(list)
        for anomaly in self.anomalies:
            by_type[anomaly['type']].append(anomaly)
        
        return {
            "total_anomalies": len(self.anomalies),
            "by_type": {
                atype: {
                    "count": len(anomalies),
                    "recent": anomalies[-5:],
                }
                for atype, anomalies in by_type.items()
            },
        }
    
    def generate_monitoring_report(self, output_file: str = None) -> Dict[str, Any]:
        """Generate comprehensive monitoring report."""
        report = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "output_directory": self.output_dir,
            },
            "overall_stats": self.get_real_time_stats(),
            "behavioral_analysis": self.get_behavioral_analysis(),
            "learning_curves": self.get_learning_curve_analysis(),
            "anomalies": self.get_anomaly_report(),
            "training_sessions": [
                {
                    "session_id": s.session_id,
                    "samples_processed": s.samples_processed,
                    "training_time_s": s.total_training_time_s,
                    "total_concepts_learned": sum(s.concepts_learned),
                    "avg_learning_rate": np.mean(s.learning_rate) if s.learning_rate else 0,
                }
                for s in self.training_sessions
            ],
        }
        
        # Save report
        if output_file is None:
            output_file = os.path.join(self.output_dir, "monitoring_report.json")
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Saved monitoring report to {output_file}")
        
        return report
    
    def _save_session(self, session: TrainingSession):
        """Save training session data."""
        filename = os.path.join(self.output_dir, f"{session.session_id}.json")
        with open(filename, 'w') as f:
            json.dump(asdict(session), f, indent=2, default=str)
    
    def export_query_log(self, output_file: str = None):
        """Export query log to JSON."""
        if output_file is None:
            output_file = os.path.join(self.output_dir, "query_log.json")
        
        log_data = [asdict(q) for q in self.query_log]
        
        with open(output_file, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
        
        self.logger.info(f"Exported {len(log_data)} queries to {output_file}")
    
    def visualize_metrics(self) -> str:
        """Generate text-based visualization of key metrics."""
        stats = self.get_real_time_stats()
        behavior = self.get_behavioral_analysis()
        
        viz = []
        viz.append("=" * 70)
        viz.append("TELEMETRY MONITOR - REAL-TIME METRICS")
        viz.append("=" * 70)
        viz.append("")
        
        viz.append("Overall Statistics")
        viz.append("-" * 70)
        viz.append(f"  Total Queries:         {stats['total_queries']}")
        viz.append(f"  Training Samples:      {stats['total_training_samples']}")
        viz.append(f"  Avg Confidence:        {stats['avg_confidence']:.2%}")
        viz.append(f"  Avg Latency:           {stats['avg_latency_ms']:.2f}ms")
        viz.append(f"  Anomalies Detected:    {stats['anomaly_count']}")
        viz.append("")
        
        if behavior.get('status') != 'insufficient_data':
            viz.append("Behavioral Analysis")
            viz.append("-" * 70)
            viz.append(f"  Confidence Trend:      {behavior['confidence']['trend']}")
            viz.append(f"  Latency Trend:         {behavior['latency']['trend']}")
            viz.append(f"  Avg Response Length:   {behavior['response_quality']['avg_length']:.0f} chars")
            viz.append("")
            
            viz.append("Query Distribution")
            viz.append("-" * 70)
            for qtype, count in behavior['query_distribution'].items():
                viz.append(f"  {qtype:20s} {count:5d}")
            viz.append("")
        
        viz.append("=" * 70)
        
        return "\n".join(viz)


class MonitoredEngine:
    """
    Wrapper around NSCKAIEngine that automatically logs telemetry.
    """
    
    def __init__(self, engine, monitor: TelemetryMonitor):
        """Initialize monitored engine."""
        self.engine = engine
        self.monitor = monitor
    
    def train_on_text(self, text: str):
        """Train on text with monitoring."""
        start_time = time.time()
        
        # Get before stats
        stats_before = self.engine.get_system_stats()
        concepts_before = stats_before['knowledge']['total_concepts']
        relations_before = stats_before['knowledge']['total_relations']
        
        # Train
        result = self.engine.train_on_text(text)
        
        # Get after stats
        elapsed = time.time() - start_time
        stats_after = self.engine.get_system_stats()
        concepts_added = stats_after['knowledge']['total_concepts'] - concepts_before
        relations_added = stats_after['knowledge']['total_relations'] - relations_before
        
        # Log to monitor
        self.monitor.log_training_step(
            samples_processed=1,
            concepts_added=concepts_added,
            relations_added=relations_added,
            elapsed_s=elapsed,
        )
        
        return result
    
    def chat(self, query: str) -> Dict[str, Any]:
        """Chat with monitoring."""
        result = self.engine.chat(query)
        
        # Log to monitor
        self.monitor.log_query(
            query=query,
            result=result,
            trace=result.get('trace'),
        )
        
        return result
    
    def __getattr__(self, name):
        """Forward all other attributes to the underlying engine."""
        return getattr(self.engine, name)


# ── Main Entry Point ────────────────────────────────────────────────────────

def main():
    """Demo of telemetry monitoring system."""
    from nsck_ai_model.ai_engine import NSCKAIEngine
    
    # Create engine and monitor
    engine = NSCKAIEngine()
    monitor = TelemetryMonitor(output_dir="./telemetry_demo")
    monitored_engine = MonitoredEngine(engine, monitor)
    
    print("=" * 70)
    print("TELEMETRY MONITORING DEMO")
    print("=" * 70)
    
    # Start training session
    monitor.start_training_session("demo_session")
    
    # Train on some data
    training_texts = [
        "The Sun is a star at the center of the Solar System.",
        "Earth orbits the Sun once per year.",
        "The Moon orbits Earth approximately every 27 days.",
        "Mars is the fourth planet from the Sun.",
        "Venus is the hottest planet in the Solar System.",
    ]
    
    print("\nTraining on sample data...")
    for text in training_texts:
        monitored_engine.train_on_text(text)
    
    monitor.end_training_session()
    
    # Run some queries
    queries = [
        "What is the Sun?",
        "Tell me about Earth.",
        "What orbits what?",
        "How hot is Venus?",
    ]
    
    print("\nRunning test queries...")
    for query in queries:
        result = monitored_engine.chat(query)
        print(f"  Q: {query}")
        print(f"  A: {result['response'][:60]}...")
    
    # Generate report
    print("\n" + "=" * 70)
    print("TELEMETRY REPORT")
    print("=" * 70)
    
    report = monitor.generate_monitoring_report()
    print(f"\nProcessed {report['overall_stats']['total_training_samples']} training samples")
    print(f"Answered {report['overall_stats']['total_queries']} queries")
    print(f"Average confidence: {report['overall_stats']['avg_confidence']:.2%}")
    print(f"Average latency: {report['overall_stats']['avg_latency_ms']:.2f}ms")
    
    # Export logs
    monitor.export_query_log()
    
    print(f"\nResults saved to: {monitor.output_dir}")
    print("\nDone!")


if __name__ == "__main__":
    main()
